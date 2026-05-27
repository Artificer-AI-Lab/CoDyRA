# Take Only What You Need: Adaptive Rank Minimization as Forgetting Regularizer in Continual Learning

[![arXiv](https://img.shields.io/badge/arXiv-2412.01004-b31b1b.svg)](https://arxiv.org/abs/2412.01004)

> Official implementation of **CoDyRA**.
>
> [Haodong Lu](https://jeff024.github.io/), Chongyang Zhao, Jason Xue, Lina Yao, Kristen Moore, [Dong Gong](https://donggong1.github.io/)

---

## TL;DR
Low-rank adaptation serves as an implicit forgetting regularizer in continual learning.

## Abstract

The central tension in continual learning (CL) is the trade-off between *plasticity* (acquiring new knowledge) and *stability* (retaining prior knowledge). We study how a pre-trained backbone can be continually updated to absorb new knowledge while preserving existing capabilities, via **capacity control**: regulating the **effective rank** of each parameter update, a per-step quantity directly controllable inside a LoRA update.

A controlled probe of LoRA rank and placement across modules and tasks reveals a consistent trade-off, with a moderate-rank sweet spot that varies by placement and task, leaving no universally optimal fixed rank; a formal bound shows forgetting grows with rank.

Building on these findings, we propose **Co**ntinual **Dy**namic **R**ank-Selective LoR**A** (**CoDyRA**), which jointly trains each LoRA update with adaptive rank minimization via sparsity-promoting regularization on per-component importance weights. The supervised objective drives $\color{purple}{\text{plasticity}}$; rank minimization regularizes $\color{green}{\text{forgetting}}$.

We show that adaptive rank minimization serves as a forgetting regularizer in the CL regime, protecting general capability and prior-task knowledge simultaneously by controlling forgetting against the current model state. Across MTIL, X-TAIL, and TRACE (CLIP, LLaMA, Gemma), CoDyRA matches or exceeds prior CL methods on learning accuracy while achieving the lowest forgetting, balancing plasticity and stability.

## Key Takeaways from Analyses

- **Takeaway 1:** LoRA placement is itself a $\color{purple}{\text{plasticity}}$–$\color{green}{\text{stability}}$ lever; no single fixed choice dominates.
- **Takeaway 2:** The $\color{purple}{\text{plasticity}}$–$\color{green}{\text{stability}}$ balance is governed by LoRA rank: $\color{purple}{\text{high rank}}$ favors plasticity, $\color{green}{\text{low rank}}$ favors stability, with a sweet spot at moderate rank.
- **Takeaway 3:** The sweet-spot rank is not universal: its location varies systematically by module and by downstream task.
- (See more details in the [paper](https://arxiv.org/abs/2412.01004).)

## Overview of CoDyRA Methodology

![CoDyRA overview diagram](figs/overview.png)

CoDyRA introduces a dynamic rank-selection LoRA, enabling each pre-trained weight matrix to adaptively retain only the necessary ranks for downstream adaptation while preserving pre-trained capabilities. After each task, the rank-pruned LoRA updates merge into the backbone, adding no inference overhead.

**Key properties:**
- ✅ No past data, task IDs, or per-task modules — operates under a strict CL regime
- ✅ No inference overhead — updates merge into the backbone
- ✅ A single rank-based criterion protects $\color{green}{\text{general (pretrained) capability}}$ *and* $\color{green}{\text{prior-task knowledge}}$
- ✅ Fewest trainable parameters among baselines (4.4M vs 60M–130M)
- ✅ Lowest Backward Transfer (BWT 1.87%) across replay, modular, orthogonal-subspace, and fixed-rank LoRA baselines

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
- Please refer to the following guide for setting up datasets: [CoOp](https://github.com/KaiyangZhou/CoOp/blob/main/DATASETS.md)


## Running CoDyRA

```bash
bash runner_codyra.sh
```

## License
CoDyRA is released under the Apache License 2.0. See [`LICENSE`](LICENSE) for details.

## Citation

```bibtex
@article{lu2024adaptive,
  title   = {Take Only What You Need: Adaptive Rank Minimization as Forgetting Regularizer in Continual Learning},
  author  = {Lu, Haodong and Zhao, Chongyang and Xue, Jason and Yao, Lina and Moore, Kristen and Gong, Dong},
  journal = {arXiv preprint arXiv:2412.01004},
  year    = {2024}
}
```

## Acknowledgement

Our repo benefits from [MoE-Adapters](https://github.com/JiazuoYu/MoE-Adapters4CL) and [RAIL](https://github.com/linghan1997/Regression-based-Analytic-Incremental-Learning). We thank them for their wonderful works.
