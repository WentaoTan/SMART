import json
import os
import random
# import nltk
import re
import argparse
from collections import Counter
from PIL import Image
import json
from nltk import ngrams
from collections import Counter

def extract_steps(input_string):
    # Split the input string into individual steps using regex
    # This regex looks for "Step" followed by a number and a colon
    steps = re.split(r'(Step \(\d+\))', input_string)
    
    # Combine the step indicators with their corresponding content
    # We use a list comprehension to merge step labels with their contents
    step_list = [steps[i] + steps[i + 1].strip() for i in range(1, len(steps) - 1, 2)]
    
    return step_list

def has_high_probability_bigrams(text, threshold=0.1):
    # 将文本分割为单词
    words = nltk.word_tokenize(text)
    
    # 创建二元组
    bigrams = list(nltk.bigrams(words))
    
    # 统计二元组出现次数
    bigram_counts = Counter(bigrams)
    
    # 计算总的二元组数量
    total_bigrams = len(bigrams)
    
    # 检查每个二元组的出现概率
    for bigram, count in bigram_counts.items():
        probability = count / total_bigrams  # 计算概率
        if probability > threshold:  # 检查是否超过阈值
            # print(text)
            # print("高概率二元组:", bigram, "出现次数:", count, "概率:", probability)
            return True  # 找到一个高概率的二元组
    
    return False  # 没有高概率的二元组

def has_repeated_ngrams(text, n, max_count=3):
    """检查文本中是否存在重复超过阈值的n-gram"""
    words = text.split()
    if len(words) < n:
        return False
    n_grams = ngrams(words, n)
    counter = Counter(n_grams)
    return any(count > max_count for count in counter.values())

def main(args):
    choosen_file = f"{args.filename}_chosen.jsonl"
    rejected_file = f"{args.filename}_rejected.jsonl"
    # rejected_why_file = f"{args.filename}_rejected_why.jsonl"

    choosen_lines = open(choosen_file, "r", encoding='utf-8').readlines()
    rejected_lines = open(rejected_file, "r", encoding='utf-8').readlines()
    # rejected_why_lines = open(rejected_why_file, "r", encoding='utf-8').readlines()

    message = []

    for cline, rline in zip(choosen_lines, rejected_lines):
        cline = json.loads(cline)
        rline = json.loads(rline)
        
        assert cline['image'] == rline['image']
        cline['rationale'] = cline['AoT']
        rline['rationale'] = rline['AoT']
        if len(cline['rationale']) < 10 or len(rline['rationale']) < 10:
            continue
        args.n = 2
        if has_repeated_ngrams(cline['rationale'], args.n):
            continue  # 跳过包含重复n-gram的样本
        question = cline['conversations'][0]['value'].split('\n')
        question = ' '.join(question[2:])
        all_answers = cline['conversations'][0]['value'].split('Choices:\n')[-1]
        all_answers = all_answers.split('\n')
        random.shuffle(all_answers)
        
        gt = cline['conversations'][1]['value'].split('The answer is ')[-1]
        for ans in all_answers:
            if f'({gt})' in ans:
                wrong = ans
        
        if f'({gt})' not in '\n'.join(cline['rationale'].split('\n')[1:]):continue
        if f'({gt})' in '\n'.join(rline['rationale'].split('\n')[1:]):continue
        if 'however' in cline['rationale'].lower():continue
        new_item = {
                "conversations": [
                    {
                        "from": "human",
                        "value": cline['conversations'][0]['value']
                    }
                ],
                "chosen": {
                    "from": "gpt",
                    "value": cline['rationale']
                },
                "rejected": {
                    "from": "gpt",
                    "value": rline['rationale']
                },
                "images": [
                    cline['image']
                ]
            }


        message.append(new_item)
        

    json.dump(message, open(f"{args.filename}_noScores_aug.json", "w", encoding='utf-8'),indent=4,)
    print(len(choosen_lines))
    print(len(message))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some jsonl files.")
    parser.add_argument('--filename', type=str, required=True, help='Base filename to process')
    
    args = parser.parse_args()
    main(args)
