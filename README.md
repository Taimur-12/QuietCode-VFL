# Product-Quantized Passive Representations for Communication-Efficient Reconstruction Privacy in Medical Vertical Federated Learning

**Group 38 | CS 437 — Deep Learning | Spring 2026**

GitHub Repository: [https://github.com/Taimur-12/CS-437-VFL](https://github.com/Taimur-12/CS-437-VFL)

---

## Table of Contents

- [Project Overview](#project-overview)
- [Repository Structure](#repository-structure)
- [Getting Started](#getting-started)
- [Experimental Design](#experimental-design)
- [Key Innovations](#key-innovations)
- [Results](#results)
- [Citations](#citations)

---

## Project Overview

In skin-lesion diagnosis, dermoscopic images may be held by an imaging provider while patient metadata (age, sex, anatomical site) and diagnostic labels sit with a clinical record system. Vertical federated learning (VFL) lets both parties collaborate on a shared classification task without exchanging raw data — but the passive image party still has to transmit intermediate neural representations to the active side, and these can be inverted to recover private images.

This project studies whether a **learned product-quantized bottleneck** at the passive client's communication boundary can simultaneously improve diagnostic utility, reduce image reconstructability, and cut communication cost — with no changes to the active client or server.

We evaluate 11 transmission methods across three task stages using ISIC-2019 (8-class skin lesion benchmark, 25,331 images). The passive party owns the image; the active party owns metadata and the label. Our best configuration, `H_vq_K64`, improves balanced accuracy from 0.752 to 0.786 over continuous VFL, reduces reconstruction SSIM from 0.622 to 0.503, and uses **853× fewer bits**.

---

## Repository Structure

```
dl-vfl-derma/
├── VFL_VQ_Experiments.ipynb       ← Main notebook: all 11 methods, training, InverNet attack, figures
├── Embedding_Analysis.ipynb       ← Silhouette score + t-SNE analysis of learned representations
├── Reconstruction_Attack.ipynb    ← Reconstruction grid visualization per method
├── make_figures.py                ← Pareto curve + bits vs. WACC/SSIM/LPIPS plots
│
├── figures/                       ← All generated figures (Pareto, reconstruction grids, bits plots)
├── results_final/                 ← Per-method JSON metrics (utility + reconstruction, 2 seeds each)
│                                    (model checkpoints available separately — see below)
├── embedding_analysis/            ← Silhouette scores, t-SNE PNGs, codebook utilization
│
├── DL Project Report Template/    ← Full LaTeX source
└── 38_27100028_27100183_Report.pdf ← Final compiled report
```

---

## Getting Started

### Prerequisites

```
Python 3.8+
torch >= 2.0
torchvision
timm                  # EfficientNet-B0 backbone
lpips                 # Perceptual loss for InverNet
scikit-learn          # Silhouette scoring
matplotlib
pandas
numpy
```

Install in one step:

```bash
pip install torch torchvision timm lpips scikit-learn matplotlib pandas numpy
```

---

### Model Checkpoints

Trained model weights for all 11 methods × 2 seeds (22 `.pt` files, ~374 MB total) are hosted on Google Drive and are not included in this repository:

**[Download Checkpoints](https://drive.google.com/drive/folders/1gS6W70RRdUbKQVLTXbEKrs1yRIs9PuSz?usp=share_link)**

Place the downloaded `checkpoints/` folder inside `results_final/` before running Stage B or the reconstruction notebooks.

### Dataset

[ISIC 2019 Training Data](https://www.kaggle.com/datasets/andrewmvd/isic-2019) from Kaggle. The notebooks expect the dataset at:

```
/kaggle/input/datasets/andrewmvd/isic-2019/ISIC_2019_Training_Input/
```

### Running the Notebooks

All notebooks are designed to run on **Kaggle with a GPU (T4 or P100)**. Open `VFL_VQ_Experiments.ipynb` and run sections sequentially:

- **Stage A** — trains all 11 VFL methods (2 seeds each)
- **Stage B** — trains InverNetV9 reconstruction attacker per method
- **Stage C** — aggregates results and produces figures

`Embedding_Analysis.ipynb` and `Reconstruction_Attack.ipynb` load saved results from `results_final/` and `figures/` — they do not require re-training.

---

## Experimental Design

The project follows the TA's required task progression:

### Baseline — Continuous VFL (`A_plain_vfl`)

The passive party encodes the image with a frozen EfficientNet-B0 (1280-d output) and transmits the full floating-point embedding. The active party fuses it with metadata via a small MLP and classifies. This establishes the utility ceiling and the worst-case privacy exposure.

### First Improvement — Projection + Sign Quantization

Two improvements over the baseline are evaluated:

- **`A_proj_vfl`** — adds a learned 128-d linear projection before transmission. Tests whether dimensionality reduction alone improves privacy or utility.
- **`S_sign_quant`** / **`S_rand_sign`** — transmit a 128-bit binary vector (sign of each dimension, or random-masked sign). A cheap non-learned discrete baseline at 128 bits.

### Second Improvement — Product-Quantized Bottleneck (`H_vq_*`)

The passive party projects the 1280-d EfficientNet embedding to 128 dimensions, splits it into M equal subspaces, and independently quantizes each subspace to the nearest entry in a learned codebook of K vectors. Only the M codebook indices are transmitted (M·log₂K bits). The active party reconstructs the 128-d vector via lookup before classification.

Seven VQ ablations study the effect of codebook size K, number of subspaces M, commitment loss weight β, and codebook initialization:

| Method | K | M | β | Bits |
|---|---|---|---|---|
| `H_vq_K64` | 64 | 8 | 0.25 | 48 |
| `H_vq_K256` | 256 | 8 | 0.25 | 64 |
| `H_vq_M4` | 256 | 4 | 0.25 | 32 |
| `H_vq_M16` | 256 | 16 | 0.25 | 128 |
| `H_vq_commit_low` | 256 | 8 | 0.10 | 64 |
| `H_vq_commit_high` | 256 | 8 | 0.50 | 64 |
| `H_vq_no_kmeans` | 256 | 8 | 0.25 | 64 |

---

## Key Innovations

### 1. Product-Quantized Communication Bottleneck

The VQ layer is applied to the **128-dimensional projected representation**, not the 1280-d EfficientNet embedding. Each transmission is M codebook indices:

```python
# Split 128-d projection into M subspaces, quantize each independently
z_q, indices, commit_loss = product_quantizer(z_proj)  # z_proj: (B, 128)
# Transmitted: indices of shape (B, M) — M·log2(K) bits total
```

Gradients flow through the discrete lookup via the straight-through estimator. Codebooks are initialized with mini-batch K-means at epoch 4 to avoid early codebook collapse.

### 2. InverNetV9 Reconstruction Attacker

Each method is evaluated against a passive inversion attacker trained on intercepted embeddings:

```
FC(in_dim → 256×4×4) → ResBlock × 2 → ConvTranspose × 4 (→ 64×64) → Refine → Tanh
Loss: MSE + 0.1 · LPIPS | Epochs: 50
```

The attacker's FC layer scales with the input dimension, giving higher-dimensional methods a proportionally larger attacker — conservative for VQ methods, fair for the baseline.

### 3. Representation Regularization

VQ discretization forces the passive encoder to map visually similar lesions to the same codebook entry, improving within-class compactness. Cosine silhouette scores confirm this:

| Method | Silhouette | Bits |
|---|---|---|
| `A_plain_vfl` | 0.015 | 40960 |
| `A_proj_vfl` | −0.015 | 4096 |
| `S_sign_quant` | 0.058 | 128 |
| `H_vq_K64` | 0.074 | 48 |
| `H_vq_M4` | **0.136** | 32 |
| `H_vq_M16` | 0.053 | 128 |

`A_proj_vfl` having a **negative** silhouette score rules out the projection layer as the source of regularization — it comes entirely from VQ discretization.

---

## Results

All 11 methods evaluated over 2 seeds. WACC = weighted balanced accuracy (primary metric, ISIC-2019 is class-imbalanced). SSIM: higher = easier reconstruction = weaker privacy. LPIPS: higher = harder reconstruction = stronger privacy.

| Method | Bits | WACC ± std | SSIM ↓ | LPIPS ↑ |
|---|---|---|---|---|
| `A_plain_vfl` (baseline) | 40960 | 0.752 ± 0.004 | 0.622 | 0.111 |
| `A_proj_vfl` | 4096 | 0.767 ± 0.001 | 0.585 | 0.139 |
| `S_rand_sign` | 128 | 0.723 ± 0.007 | 0.529 | 0.214 |
| `S_sign_quant` | 128 | 0.770 ± 0.006 | 0.538 | 0.196 |
| **`H_vq_K64`** | **48** | **0.786 ± 0.007** | **0.503** | **0.241** |
| `H_vq_K256` | 64 | 0.773 ± 0.003 | 0.514 | 0.227 |
| `H_vq_no_kmeans` | 64 | 0.779 ± 0.000 | 0.520 | 0.232 |
| `H_vq_M4` | 32 | 0.771 ± 0.005 | 0.512 | 0.239 |
| `H_vq_M16` | 128 | 0.775 ± 0.002 | 0.540 | 0.194 |
| `H_vq_commit_low` | 64 | 0.778 ± 0.007 | 0.512 | 0.226 |
| `H_vq_commit_high` | 64 | 0.780 ± 0.001 | 0.512 | 0.231 |

**`H_vq_K64` Pareto dominates all baselines:** +3.4 pp WACC over continuous VFL, SSIM reduced by 0.119, at 853× fewer bits.

---

## Citations

1. Yang et al., "UIFV: Data Reconstruction Attack in Vertical Federated Learning," arXiv 2406.12588, 2024.
2. van den Oord et al., "Neural Discrete Representation Learning (VQ-VAE)," NeurIPS 2017.
3. Tan & Le, "EfficientNet: Rethinking Model Scaling for CNNs," ICML 2019.
4. Oh et al., "FedVQCS: Vector Quantized Compressed Sensing for Federated Learning," arXiv 2204.07692, 2022.
5. Anoosha et al., "HybridVFL: Cross-Modal Transformer Fusion for Medical VFL," arXiv 2512.10701, UCC 2025.
6. Combalia et al., "BCN20000: Dermoscopic Lesions in the Wild," arXiv 1908.02288, 2019.
7. Tschandl et al., "The HAM10000 Dataset," Scientific Data, 2018.
8. Bengio et al., "Estimating or Propagating Gradients Through Stochastic Neurons," arXiv 1308.3432, 2013.
