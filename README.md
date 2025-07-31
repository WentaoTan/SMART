# SMART

Code for *From Answers to Rationales: Self-Aligning Multimodal Reasoning with Answer-Oriented Chain-of-Thought*

---

## Environment Setup

This project’s training code is based on the popular [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory).  
Please follow the installation instructions provided in the repository to set up the environment.

---

## Data Generation

1. Download the MathV360K dataset from [Hugging Face](https://huggingface.co/datasets/Zhiqiang007/MathV360K).  
2. Extract the multiple-choice questions from the dataset and save them separately as a JSON file.  
3. Modify the parameters in the `ga.sh` script located inside the `AoT_data` directory according to your setup.  
4. Run the `ga.sh` script. This will generate output data similar to `aot_noScores.json`.

---

## Training

Please refer to the `qwen2vl_full_dpo.yaml` training script for the training configuration and instructions.

---

## Acknowledgement

We would like to thank the authors and contributors of the [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) and the creators of the [MathV360K](https://huggingface.co/datasets/Zhiqiang007/MathV360K) dataset for their valuable resources.

---

## Contact

For any questions or inquiries, please contact:  
- Email: [ftwentaotan@mail.scut.edu.cn](mailto:ftwentaotan@mail.scut.edu.cn)  
- Email: [731584671@qq.com](mailto:731584671@qq.com)
  
现在欢迎用中英文双contact我啦！
