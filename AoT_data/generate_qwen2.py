import copy
import os
import json
import re
import tqdm
import torch
import argparse
import datetime
import requests
from PIL import Image
from io import BytesIO
from transformers import TextStreamer
import datasets
import sys
# sys.path.insert(0, "../../seva")

from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info

import torchvision.transforms as transforms
import random
from PIL import ImageFilter


def convert_dict_to_tensor(results, device):
    part_tensor = json.dumps(results)
    part_tensor = torch.Tensor([ord(part_tensor[i]) for i in range(len(part_tensor))]).long().to(device)
    return part_tensor

# PROMPT='''Given an image, a question and an answer. Your task is to analyze the problem, combine the picture content, and deduce the answer step by step.
# Question: <question>
# Answer: <answer>
# Please deduce the Answer step by step:'''

# PROMPT='''There is a question about this image, which is "<question>". The correct answer to the question is "<answer>". Why? Please provide a concise and direct step-by-step reasoning in the format like: 'Step (1), ... Step (2), ...'. Make sure to keep the number of steps as few as possible, and provide the correct answer in the final step.'''
PROMPT='''There is a question about this image, which is "<question>". Please provide a concise and direct step-by-step reasoning in the format like: 'Step 1, ... Step 2, ...'. Make sure to keep the number of steps as few as possible, and provide the correct answer in the final step.'''

PROMPT_input = '''Question: <question>
Choices: <choices>'''

def add_diffusion_noise(image_tensor, noise_step=500):
    num_steps = 1000  # Number of diffusion steps

    # decide beta in each step
    betas = torch.linspace(-6,6,num_steps)
    betas = torch.sigmoid(betas) * (0.5e-2 - 1e-5) + 1e-5

    # decide alphas in each step
    alphas = 1 - betas
    alphas_prod = torch.cumprod(alphas, dim=0)
    alphas_prod_p = torch.cat([torch.tensor([1]).float(), alphas_prod[:-1]],0) # p for previous
    alphas_bar_sqrt = torch.sqrt(alphas_prod)
    one_minus_alphas_bar_log = torch.log(1 - alphas_prod)
    one_minus_alphas_bar_sqrt = torch.sqrt(1 - alphas_prod)

    def q_x(x_0,t):
        noise = torch.randn_like(x_0)
        alphas_t = alphas_bar_sqrt[t]
        alphas_1_m_t = one_minus_alphas_bar_sqrt[t]
        return (alphas_t*x_0 + alphas_1_m_t*noise)

    noise_delta = int(noise_step) # from 0-999
    noisy_image = image_tensor.clone()
    image_tensor_cd = q_x(noisy_image,noise_step) 

    return image_tensor_cd

import random

def select_random_element(lst):
    element = random.choice(lst)
    while len(element) == 0:
        element = random.choice(lst)
    return element


def main(args):
    # Model
    local_rank = int(os.environ["LOCAL_RANK"])
    device = f"cuda:{local_rank}"
    # model = Qwen2VLForConditionalGeneration.from_pretrained(
    # args.model_path, torch_dtype="auto", device_map=device,
    #     ).eval()
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map=device,
    )
    processor = AutoProcessor.from_pretrained(args.model_path,use_fast=True)
    conv_mode = "llava_v1"

    ## get question file
    image_file_list = open(args.image_file_list)

    lines = list(image_file_list.readlines())
    # lines = json.load(image_file_list)

    rank, word_size = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    step = len(lines) // word_size + 1
    start, end = rank * step, (rank + 1) * step
    end = min(end, len(lines))
    results = []
    if int(os.environ["RANK"]) == 0:
        print("generating answers...")

    results = []
    # keywords = ["uncertain", "Uncertain", "insufficient", "Insufficient", "cannot be determined", "not provide", "not possible"]
    rank_output_file = f'{args.res_file}_rank_{rank}.jsonl'
    if os.path.exists(rank_output_file):
        with open(rank_output_file, 'r') as f:
            processed_lines = sum(1 for _ in f)  # 计算已处理的行数
    else:
        processed_lines = 0  # 如果文件不存在，说明还没有处理任何样本
    # Open the file in append mode to ensure data is written progressively
    with open(rank_output_file, "a", encoding='utf-8') as f:
        for line in tqdm.tqdm(lines[start+processed_lines:end]):
    # for line in tqdm.tqdm(lines[start:end]):
    # for line in tqdm.tqdm(range(start,end)):
            data = json.loads(line)
            # data = line
            '''for m3cot'''
            # data = lines[line]
            # if data['image'] is None:continue
            # formatted_list = [f"{chr(65 + i)}. {description}" for i, description in enumerate(data['choices'])]
            # gt = data['answer']
            # index = next((i for i, option in enumerate(formatted_list) if option.startswith(f'{gt}. ')), None)
            # all_answers = data['choices']
            # question = data['question']
            # image = data['image']
            '''for sqa'''
            # image_path = data["image_path"].replace('/workspace/tanwentao1/self-reward/SeVa-main/ReasoningData/scienceqa/','/public/data0/NLP/users/tanwentao1/52/project/SeVa-main/ReasoningData/scienceqa/')
            # image = Image.open(image_path).convert("RGB")
            # all_answers = copy.deepcopy(data['choices'])
            # if args.aug:
            #     del all_answers[data['answer']]
            #     mislead_answer = random.sample(all_answers,1)[0]
            #     input_text = PROMPT.replace('<question>',data['question']).replace('<answer>',mislead_answer)
            # else:
            #     input_text = PROMPT.replace('<question>',data['question']).replace('<answer>',all_answers[data['answer']])

            '''for mathv360k'''
            image_path = os.path.join(args.image_path, data['image'])
            image = Image.open(image_path).convert("RGB")
            if args.aug:
                pil_to_tensor = transforms.ToTensor()
                tensor_to_pil = transforms.ToPILImage()
                horizontal_flip = transforms.RandomHorizontalFlip(p=0.5)
                vertical_flip = transforms.RandomVerticalFlip(p=0.5)
                Erasing =  transforms.RandomErasing(p=0.5, scale=(0.02, 0.2), ratio=(0.3, 3.3), value=0, inplace=False)
                image = horizontal_flip(pil_to_tensor(image)) 
                image = vertical_flip(image) 
                image = tensor_to_pil(Erasing(image))
            question = data['conversations'][0]['value'].split('\n')
            question = ' '.join(question[2:])
            all_answers = data['conversations'][0]['value'].split('Choices:\n')[-1]
            all_answers = all_answers.split('\n')
            gt = data['conversations'][1]['value'].split('The answer is ')[-1]

            index = next((i for i, option in enumerate(all_answers) if option.startswith(f'({gt})')), None)
            try:
                if args.aug:
                    del all_answers[index]
                    
                    mislead_answer = random.sample(all_answers,1)[0]
                
                    input_text = PROMPT.replace('<question>',question).replace('<answer>',mislead_answer)
                    pre = f'The answer is {mislead_answer}.\n'
                else:
                    input_text = PROMPT.replace('<question>',question).replace('<answer>',all_answers[index])
                    pre = f'The answer is {all_answers[index]}.\n'
                    del all_answers[index]
                    
                    mislead_answer = random.sample(all_answers,1)[0]
            except:
                continue
            # print(input_text)
            # import IPython
            # IPython.embed()
            '''for test time prompt'''
            # image_path = os.path.join(args.image_path, data['image'])
            # image = Image.open(image_path).convert("RGB")
            # query = data['conversations'][0]['value']
            # hint_pattern = re.compile(r"(Hint: )(.+?)( at the end\.)", re.DOTALL)
            # query = hint_pattern.sub(
            #     r"\1Please first give a simple rationale and then \2\3",
            #     query
            # )
            # input_text = query
            '''for multi image input'''
            # image_path = os.path.join(args.image_path, data['image'])
            # image = Image.open(image_path).convert("RGB")
            # question = data['conversations'][0]['value'].split('\n')[2].replace('Question: ','')
            # all_answers = data['conversations'][0]['value'].split('Choices:\n')[-1]
            # all_answers = all_answers.split('\n')
            # gt = data['conversations'][1]['value'].split('The answer is ')[-1]
            # input_text = f'There is a question about the image, which is "{question}". To answer the question, which region should I focus on? Please provide the bbox coordinates.'
            
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image","image": image},
                        {"type": "text", "text": input_text},
                        # {"type": "image","image": image},
                    ],
                }
            ]

            # Preparation for inference
            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )

            if args.aug:
                inputs['pixel_values'] = add_diffusion_noise(inputs['pixel_values'],600)

            inputs = inputs.to(model.device)
            generated_ids = model.generate(**inputs,
                                        do_sample=True,
                                        # repetition_penalty=1.05,
                                        temperature=0.7,
                                        top_p=0.9,
                                        max_new_tokens=512)
            generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            generated_text = output_text[0]
            
            new_data = copy.deepcopy(data)
            new_data['rationale'] = generated_text
            # results.append(new_data)
            f.write(json.dumps(new_data) + '\n')
            f.flush()
            
    # device = f"cuda:{torch.cuda.current_device()}"
    # # convert dictionary -> tensor for gather all results in all ranks
    # part_tensor = convert_dict_to_tensor(results, device)
    # shape_tensor = torch.tensor(part_tensor.shape, device=device)
    # shape_list = [shape_tensor.clone() for _ in range(int(os.environ["WORLD_SIZE"]))]
    # torch.distributed.all_gather(shape_list, shape_tensor)

    # # gather tensor
    # max_shape = max(shape_list)
    # part_tensor_pad = torch.zeros(max_shape).to(device)
    # part_tensor_pad[:part_tensor.shape[0]] = part_tensor
    # tensor_list = [part_tensor_pad.clone() for _ in range(int(os.environ["WORLD_SIZE"]))]
    # torch.distributed.all_gather(tensor_list, part_tensor_pad)

    # if int(os.environ["RANK"]) == 0:
    #     results_all_rank = []
    #     for tensor, shape in zip(tensor_list, shape_list):
    #         t = tensor.long()[:shape]
    #         _data = "".join([chr(t[i].item()) for i in range(t.shape[0])])
    #         _data = json.loads(_data)
    #         results_all_rank.extend(_data)
    #     # sort according to question_id
    #     # results_all_rank = sorted(results_all_rank, key=lambda x:x["question_id"])
    #     res_file = args.res_file
    #     save_dir = args.save_dir
    #     if not os.path.exists(save_dir):
    #         os.makedirs(save_dir)
    #     with open(os.path.join(save_dir, res_file), "w") as f:
    #         for res in results_all_rank:
    #             f.write(json.dumps(res)+'\n')
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--res_file", type=str, default="generate.jsonl")
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")
    
    parser.add_argument('--augmentations', nargs='+', type=str, default=[])
    parser.add_argument("--noise_step", default=500, type=int)
    parser.add_argument("--image_file_list", default=None, type=str)
    
    parser.add_argument("--image_path", type=str, required=True)
    parser.add_argument('--aug', action='store_true', help="Enable augmentation")
    args = parser.parse_args()
    
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    args.dist_backend = "nccl"
    print(
        "| distributed init (rank {}, world {}): {}".format(
            int(os.environ["LOCAL_RANK"]), int(os.environ["WORLD_SIZE"]), "env://"
        ),
        flush=True,
    )
    torch.distributed.init_process_group(
        backend="nccl",
        init_method="env://",
        world_size=int(os.environ["WORLD_SIZE"]),
        rank=int(os.environ["LOCAL_RANK"]),
        timeout=datetime.timedelta(
            days=365
        ),  # allow auto-downloading and de-compressing
    )
    
    args = parser.parse_args()
    main(args)