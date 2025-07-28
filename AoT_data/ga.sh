

MODEL=Qwen2-VL-7B
IMG_PATH=datasets--Zhiqiang007--MathV360K/data_images
IMG_DATA=datasets--Zhiqiang007--MathV360K/choices.jsonl
QUES_DATA=datasets--Zhiqiang007--MathV360K/choices.jsonl
OUTPUT_DATA=aot

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node 8 --master_port 29502 generate_qwen2.py \
    --model-path $MODEL \
    --image_file_list $QUES_DATA \
    --image_path $IMG_PATH \
    --save_dir ./ \
    --res_file ${OUTPUT_DATA}_rejected.jsonl \
    --aug

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node 8 --master_port 29502 generate_qwen2.py \
    --model-path $MODEL \
    --image_file_list $QUES_DATA \
    --image_path $IMG_PATH \
    --save_dir ./ \
    --res_file ${OUTPUT_DATA}_chosen.jsonl \

cat ${OUTPUT_DATA}_rejected.jsonl_rank* > ${OUTPUT_DATA}_rejected.jsonl
cat ${OUTPUT_DATA}_chosen.jsonl_rank* > ${OUTPUT_DATA}_chosen.jsonl

python generate_pairs.py --filename ${OUTPUT_DATA}

# rm -r ${OUTPUT_DATA}_rejected.jsonl_rank*
# rm -r ${OUTPUT_DATA}_chosen.jsonl_rank*
