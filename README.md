# CoT-Influx
PyTorch implementation of paper "Fewer is More: Boosting LLM Reasoning with Reinforced Context Pruning"


<div align=center>
<img width=90% src="CoT-Influx.png"/>
</div>

## Abstract

Large language models (LLMs) have shown impressive capabilities in various tasks, yet they still struggle with math reasoning. Despite efforts to optimize Chain-of-Thoughts (CoT) prompts and fine-tune LLMs, the potential of few-shot learning remains unexplored.  In this work, we propose **CoT-Max**, a novel approach  pushing the boundaries of few-shot CoT learning to improve LLM math reasoning capabilities. CoT-Max addresses the challenges of the selection of useful examples and limited number of examples due to restricted context window length.  Inspired by our observation that natural language inputs contain many redundancy, we propose a coarse-to-fine pruner as a plug-and-play module for LLMs, which first identifies crucial CoT examples from a large batch and then further prunes unimportant tokens. To train the pruner, we collect a math reasoning dataset with diverse difficulty  and steps, introduce a reward  to measure both the input's effectiveness for math reasoning and token length constraints, and propose a novel training approach with reinforcement learning. As a result, CoT-Max significantly outperforms CoT and few-shot prompting baselines across various LLMs   (LLaMA2-7B, 13B, 70B) and 5 mathematical  datasets, achieving up  to 4.55\% absolute improvements. Remarkably, without any fine-tuning, LLaMA2-70B with CoT-Max surpasses GPT-3.5 and a wide range of larger LLMs (PaLM, Minerva, etc.) on the GSM8K.

## Preparation

### Requirements

- PyTorch 2.0.1+ 
```shell
pip install -r requirements.txt
```

### Preparing for pruner training data (Optional)

- Download pruner weights via link provided in [Models](#models).
- 

## Run

### Pruner training on MRD^3


- Training a pruner with LLaMA2-13B

```
CUDA_VISIBLE_DEVICES=0,1,2,3 python train_VVTQ.py \
--dist-url 'tcp://127.0.0.1:10001' \
--dist-backend 'nccl' \
--multiprocessing-distributed --world-size 1 --rank 0 \
--model deit_tiny_patch16_224_quant --batch-size 512 --lr 5e-4 \
--warmup-epochs 0 --min-lr 0 --wbits 4 --abits 4 --reg \
--softlabel_path ./FKD_soft_label_500_crops_marginal_smoothing_k_5 \
--finetune [path to full precision baseline model] \
--save_checkpoint_path ./DeiT-T-4bit --log ./log/DeiT-T-4bit.log\
--data [imagenet-folder with train and val folders]
```

### Evaluation on math reasoning dataset
```
CUDA_VISIBLE_DEVICES=0 python train_VVTQ.py \
--model deit_tiny_patch16_224_quant --batch-size 512 --wbits 4 --abits 4 \
--resume [path to W4A4 DeiT-T ckpt] --evaluate --log ./log/DeiT-T-W4A4.log \
--data [imagenet-folder with train and val folders]
```

## Models

| Model    | EM (\%) on GSM8K | Pruner weights  | Evaluation logs |
|:-------:|:--------:|:--------:|:--------:|
| `LLaMA2-7B`  | **15.92**  |[link](https://hkustconnect-my.sharepoint.com/:f:/g/personal/xhuangbs_connect_ust_hk/EoJbm6qBXoRNpOZbvv-Z4u0BSKji09RPWfFhSVjZ4Swmag?e=M3vf0h) |  - |
| `LLaMA2-13B` | **32.27**  |[link](https://hkustconnect-my.sharepoint.com/:f:/g/personal/xhuangbs_connect_ust_hk/EmsOhPWW83tIqeB8_bHoJwkBHQIlgFyDs45WvQtBdZ80iA?e=zjSkXC)  |  [link](./log/DeiT-T-W4A4.log) |
| `LLaMA2-70B` | **59.59**  |[link](https://hkustconnect-my.sharepoint.com/:f:/g/personal/xhuangbs_connect_ust_hk/El-PoCPkBLxJoIt1q5QAZPsBR6r03LtD6GT0E_JsEOa8UQ?e=jFaRmR)  |  [link](./log/DeiT-T-W3A3.log) |


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