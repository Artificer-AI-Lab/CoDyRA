# CoDyRA: Adaptive Rank, Reduced Forgetting

[![arXiv](https://img.shields.io/badge/arXiv-2412.01004-b31b1b.svg)](https://arxiv.org/abs/2412.01004)

> Official implementation of **Adaptive Rank, Reduced Forgetting: Knowledge Retention in Continual Learning Vision-Language Models with Dynamic Rank-Selective LoRA** (CoDyRA). 

---

## Abstract
Continual learning (CL) aims to accumulate knowledge from sequential tasks without catastrophic forgetting. Vision–language models like CLIP, with strong generalization, are widely used for CL. Existing methods often adapt isolated PTM components, adding inference complexity and limiting PTM improvement, or rely on replay, stored information, or assumptions, incurring high costs and limited applicability. To advance models as continual learners, we explore CL via natural, efficient PTM updates instead of complex task-specific additions. We thus study continual low-rank learning and systematically analyze how LoRA ranks and placements affect learning and forgetting. We find that a relatively higher-rank LoRA improves task learning (i.e., plasticity) but increases forgetting, while a relatively lower-rank LoRA reduces forgetting (i.e., stability) but limits adaptation. Crucially, we find a plasticity–stability balance tied to rank across parameters and tasks, with moderately small ranks maximizing CL benefits. Motivated by this, we propose Continual Dynamic Rank-Selective LoRA (CoDyRA), which continually updates PTMs with LoRA adapters of adaptively optimized rank. While the new-task objective drives learning, CoDyRA adaptively minimizes ranks with sparsity-promoting regularization to reduce interference and forgetting, achieving a plasticity–stability balance tailored to different parameters and tasks. Adaptively selected and minimized LoRA ranks keep the updated model closer to its previous state while learning new tasks. CoDyRA enables efficient CL as a sequence of LoRA-based tasks without storing past data, task information, or relying on assumptions. It preserves the original model architecture and deployment pipeline, adding no inference overhead. Extensive experiments show CoDyRA improves new representations while retaining old knowledge, achieving state-of-the-art results.


## Overview

![CoDyRA overview diagram](figs/overview.png)

## Quick Start

### 1. Environment
```bash
conda create -n codyra python=3.12 -y
conda activate codyra
# Install PyTorch that matches your CUDA setup, e.g.
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
# Install project dependencies
pip install -r requirements.txt
```

### 2. Data
- Set `--data_dir` to the root directory that should hold all benchmarks (Aircraft, Caltech101, DTD, EuroSAT, Oxford Flowers, Food-101, MNIST, Oxford Pets, Stanford Cars, SUN397).
- Please refer to the following guides for setting up datasets:
[CoOp](https://github.com/KaiyangZhou/CoOp/blob/main/DATASETS.md)


## Running CoDyRA

```
bash runner_codyra.sh
```

## License
CoDyRA is released under the Apache License 2.0. See [`LICENSE`](LICENSE) for details.

## Citation

```bibtex
@article{lu2024adaptive,
  title   = {Adaptive Rank, Reduced Forgetting: Knowledge Retention in Continual Learning Vision-Language Models with Dynamic Rank-Selective LoRA},
  author  = {Lu, Haodong and Zhao, Chongyang and Xue, Jason and Yao, Lina and Moore, Kristen and Gong, Dong},
  journal = {arXiv preprint arXiv:2412.01004},
  year    = {2024}
}
```

## Acknowledgement

Our repo benefits from [MoE-Adapters](https://github.com/JiazuoYu/MoE-Adapters4CL) and [RAIL](https://github.com/linghan1997/Regression-based-Analytic-Incremental-Learning). We thank them for their wonderful works.
