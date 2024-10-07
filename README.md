# CoT-Influx
This repository contains the code of CoT-Influx introduced in our work: "[Fewer is More: Boosting LLM Reasoning with Reinforced Context Pruning](https://arxiv.org/abs/2312.08901)", published in EMNLP Main Conference 2024.

<div align=center>
<img width=90% src="CoT-Influx.png"/>
</div>

## Abstract

Motivated by the observation that adding more concise CoT examples in the prompt can improve LLM reasoning performance, we propose **CoT-Influx**, which employs a coarse-to-fine pruner to maximize the input of effective and concise CoT examples. The pruner first selects as many crucial CoT examples as possible and then prunes unimportant tokens to fit the context window. 

- CoT-Influx significantly outperforms various prompting baselines across various LLMs (LLaMA2-7B, 13B, 70B) and 5 math datasets, achieving up to 4.55% absolute improvements. 
- Without any fine-tuning, LLaMA2-70B with CoT-Influx surpasses GPT-3.5 and a wide range of larger LLMs (PaLM, Minerva 540B, etc.) on the GSM8K.

## ⚒️ TODO

- [x] [Preparation](#Preparation)
- [x] [Citation](#Citation)
- [x] [Acknowledgements](#Acknowledgements)

## Preparation

### Requirements

- PyTorch 2.0.1+ 
```shell
pip install -r requirements.txt
```

### Preparing for pruner training data 

- Download pruner weights via link provided in [Models](#models).
- Check our prompt evolution scripts for MRD^3 in `./mrd3/*`

## Run

### Pruner training on MRD$^3$

To be released soon

### Evaluation on math reasoning dataset
To evaluate the few-shot reasoning performance of LLaMA2-7B with CoT-Influx on GSM8K, run the following command
```
CUDA_VISIBLE_DEVICES=0 python example_retrieval_pruner.py \
--base_model meta-llama/Llama-2-7b-hf --pruner_model ./pruner_ckpt/llama2_13b.pth \
--candidate_set ./mrd3/score_revise_difficulty_mrd3.json \
--method few_shot_cot --cot_shot_length 24 --add_16shot \
2>&1 | tee -a ./logs/llama2-7b-gsm8k.log
```

## CoT-Influx Pruner Models and Evaluation Logs (w/ pruned prompt)

| Model    | EM (\%) on GSM8K | Pruner weights  | Evaluation logs |
|:-------:|:--------:|:--------:|:--------:|
| `LLaMA2-7B`  | **15.92**  |[link]() |  - |
| `LLaMA2-13B` | **32.27**  |[link]()  |  [link](./log/DeiT-T-W4A4.log) |
| `LLaMA2-70B` | **59.59**  |[link]()  |  [link](./log/DeiT-T-W3A3.log) |


## Citation

    @article{huang2023fewer,
        title={Fewer is More: Boosting LLM Reasoning with Reinforced Context Pruning},
        author={Huang, Xijie and Zhang, Li Lyna and Cheng, Kwang-Ting and Yang, Mao},
        journal={arXiv preprint arXiv:2312.08901},
        year={2023}
        }

## Contact

Xijie HUANG (huangxijie1108 at gmail.com or xhuangbs at connect.ust.hk) 

Li Lyna Zhang (lzhani at microsoft.com)