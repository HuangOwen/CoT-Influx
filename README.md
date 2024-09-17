# CoT-Influx
PyTorch implementation of paper "Fewer is More: Boosting LLM Reasoning with Reinforced Context Pruning"


<div align=center>
<img width=90% src="CoT-Influx.png"/>
</div>

## Abstract

Large Language Models (LLMs) have shown impressive capabilities, yet they still struggle with math reasoning. In this work, we propose **CoT-Influx**, a novel approach that pushes the boundary of few-shot Chain-of-Thoughts (CoT) learning to improve LLM mathematical reasoning. Motivated by the observation that adding more concise CoT examples in the prompt can improve LLM reasoning performance, CoT-Influx employs a coarse-to-fine pruner to maximize the input of effective and concise CoT examples. The pruner first selects as many crucial CoT examples as possible and then prunes unimportant tokens to fit the context window. As a result, by enabling more CoT examples with double the context window size in tokens, CoT-Influx significantly outperforms various prompting baselines across various LLMs (LLaMA2-7B, 13B, 70B) and 5 math datasets, achieving up to 4.55% absolute improvements. Remarkably, without any fine-tuning, LLaMA2-70B with CoT-Influx surpasses GPT-3.5 and a wide range of larger LLMs (PaLM, Minerva 540B, etc.) on the GSM8K. CoT-Influx is a plug-and-play module for LLMs, adaptable in various scenarios.

## Preparation

### Requirements

- PyTorch 2.0.1+ 
```shell
pip install -r requirements.txt
```

### Preparing for pruner training data (Optional)

- Download pruner weights via link provided in [Models](#models).
- Check our prompt evolution scripts for MRD^3

## Run

### Pruner training on MRD$^3$


- Training a pruner with LLaMA2-13B

```
CUDA_VISIBLE_DEVICES=0,1,2,3 python train_VVTQ.py \
```

### Evaluation on math reasoning dataset
```
CUDA_VISIBLE_DEVICES=0 python train_VVTQ.py \
```

## Models

| Model    | EM (\%) on GSM8K | Pruner weights  | Evaluation logs |
|:-------:|:--------:|:--------:|:--------:|
| `LLaMA2-7B`  | **15.92**  |[link]() |  - |
| `LLaMA2-13B` | **32.27**  |[link]()  |  [link](./log/DeiT-T-W4A4.log) |
| `LLaMA2-70B` | **59.59**  |[link]()  |  [link](./log/DeiT-T-W3A3.log) |


## Citation

    @article{huang2023boosting,
        title={Boosting LLM Reasoning: Push the Limits of Few-shot Learning with Reinforced In-Context Pruning},
        author={Huang, Xijie and Zhang, Li Lyna and Cheng, Kwang-Ting and Yang, Mao},
        journal={arXiv preprint arXiv:2312.08901},
        year={2023}
        }

## Contact

Xijie HUANG (huangxijie1108 at gmail.com or xhuangbs at connect.ust.hk) 

Li Lyna Zhang (lzhani at microsoft.com)