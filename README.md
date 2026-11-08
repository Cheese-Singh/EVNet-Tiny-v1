# EVNet-Tiny

A compact hybrid **ConvNeXt + global-attention** image classifier, trained from scratch on **Tiny ImageNet-200** (64×64, 200 classes) with **6.70M parameters**.

| Metric (Tiny ImageNet val, 10,000 images) | Result |
|---|---|
| **Top-1 accuracy** | **67.40%** (epoch 137, EMA weights) |
| **Top-5 accuracy** | 85.78% at the best top-1 epoch |
| Parameters | 6.70M |
| Compute (approx.) | ~1.46 GMACs per 64×64 image |
| Hardware | Single Tesla T4 (Colab), ~4.4–5.2 min/epoch |

![Training curves](assets/training_curves.png)

---

## Architecture

EVNet-Tiny (`TinyHybridNetV2` in the notebook) is a four-stage pyramid. The first three stages are ConvNeXt-style convolutional blocks; the last stage, where the feature map is only 8×8, uses global self-attention.

```
Input 3 × 64 × 64
 │
 ├─ Stem: Conv3×3 (3→64) → LayerNorm2d → GELU              64 × 64 × 64
 ├─ Stage 1: 2 × ConvNeXtBlock (dim 64)                     64 × 64 × 64
 ├─ Downsample: LN → Conv2×2, stride 2 (64→128)
 ├─ Stage 2: 2 × ConvNeXtBlock (dim 128)                    128 × 32 × 32
 ├─ Downsample: LN → Conv2×2, stride 2 (128→256)
 ├─ Stage 3: 4 × ConvNeXtBlock (dim 256)                    256 × 16 × 16
 ├─ Downsample: LN → Conv2×2, stride 2 (256→384)
 ├─ Stage 4: 2 × HybridAttentionBlock (dim 384, 8 heads)    384 × 8 × 8
 └─ Head: LN → GlobalAvgPool → Dropout(0.2) → Linear(384→200)
```

**ConvNeXtBlock** (stages 1–3): 7×7 depthwise conv → LayerNorm → 1×1 conv (×4 expansion) → GELU → 1×1 conv, with a learnable per-channel residual scale (LayerScale, initialised at 1e-3) and stochastic depth.

**HybridAttentionBlock** (stage 4): pre-norm multi-head self-attention over all 64 spatial tokens (implemented with 1×1 convs for Q/K/V and PyTorch's `scaled_dot_product_attention`), followed by a 1×1-conv MLP. Both sub-layers use LayerScale and stochastic depth.

**Design choices**

- *Convolutions early, attention late.* Early stages work at high resolution where attention would be expensive (4,096 tokens at 64×64) and where local inductive bias helps most. At 8×8 there are only 64 tokens, so global attention is cheap and lets the network mix information across the whole image before classification.
- *LayerNorm everywhere* (no BatchNorm), so behaviour doesn't depend on batch statistics.
- *Stochastic depth* ramping linearly from 0 to 0.1 across the 10 blocks.
- *LayerScale at 1e-3* to stabilise residual blocks early in training.

---

## Training recipe

| Item | Setting |
|---|---|
| Data | Tiny ImageNet-200: 100,000 train / 10,000 val, 64×64 |
| Optimiser | AdamW, base LR 8e-4, weight decay 0.05 (none on norms/biases/1-D params) |
| Schedule | 5 epochs linear warm-up, then cosine decay to 1e-6, planned for 200 epochs |
| Batch size | 128, mixed precision (fp16 autocast), grad-clip 1.0 |
| Loss | Cross-entropy, label smoothing 0.05 |
| Augmentation | RandomResizedCrop (scale 0.65–1.0), horizontal flip, RandAugment (2 ops, mag 7), RandomErasing (p=0.1) |
| Batch mixing | MixUp (α=0.2) or CutMix (α=1.0), applied to 70% of batches |
| Regularisation | Stochastic depth 0.1, head dropout 0.2 |
| EMA | Weights EMA with decay 0.999; **all reported numbers use the EMA model** |
| Seed | 42 |

---

## Results

### Main run

| Epoch | Val top-1 | Val top-5 | Val loss |
|---:|---:|---:|---:|
| 1 | 4.17% | 13.49% | 5.015 |
| 20 | 45.69% | 73.14% | 2.555 |
| 50 | 61.99% | 84.41% | 1.887 |
| 100 | 66.49% | 86.39% | 1.776 |
| **137 (best)** | **67.40%** | 85.78% | 1.795 |
| 162 (last logged) | 66.81% | 84.76% | 1.856 |

Accuracy climbs quickly until about epoch 60 and then plateaus around 66–67%. After roughly epoch 100, training loss keeps falling while validation loss flattens and then rises (1.78 → 1.86 by epoch 162), a sign of overfitting on a 100k-image dataset.

### Follow-up experiments (starting from the epoch-137 checkpoint)

| Experiment | Logged epochs | Val top-1 | vs. 67.40% baseline |
|---|---:|---:|---|
| Low-LR fine-tune (1e-4, cosine, same augmentations) | 3 | 67.31% → 67.11% | no gain |
| Fine-tune with S²-FracMix (spectral-residual-saliency-guided patch mixing with fractal blending, plus MixUp/CutMix/ResizeMix modes; LR 4e-5) | 4 | 67.21% → 66.67% | no gain |

Neither experiment improved on the baseline in the epochs recorded in the notebook, and both logs stop well short of the planned 30 epochs, so these are early indications, not final verdicts. (The S²-FracMix run reports lower validation loss than the main run only because its validation function uses plain cross-entropy without label smoothing; the two loss values aren't comparable.)

---

## Comparison with ResNet

The notebook's `build_model()` can also build **ResNet-18** and **ResNet-50** with the same 64×64 adaptation (3×3 stride-1 stem, no max-pool), so a like-for-like comparison is a one-line change (`MODEL_NAME = "resnet18"`). Those runs are **not in this notebook yet**, so their accuracy columns are left open rather than filled with numbers from other papers, which use different recipes and aren't comparable.

| | EVNet-Tiny | ResNet-18 | ResNet-50 |
|---|---|---|---|
| Building block | ConvNeXt block + attention in last stage | BasicBlock (two 3×3 convs) | Bottleneck (1×1, 3×3, 1×1) |
| Normalisation | LayerNorm | BatchNorm | BatchNorm |
| Activation | GELU | ReLU | ReLU |
| Spatial mixing | 7×7 depthwise conv; global self-attention at 8×8 | 3×3 conv | 3×3 conv |
| Downsampling | 2×2 stride-2 conv (patchify-style) | stride-2 3×3 conv | stride-2 3×3 conv |
| Residual scaling | LayerScale (1e-3) | none | none |
| Stochastic depth | yes (0.1) | no | no |
| Parameters (200 classes) | **6.70M** | ~11.27M | ~23.91M |
| Approx. MACs at 64×64 | **~1.46G** | ~2.22G | ~5.19G |
| Top-1 on Tiny ImageNet, same recipe | **67.40%** | *to run* | *to run* |

Parameter counts for the ResNets are computed from the standard torchvision layer shapes with the 64×64 stem and a 200-way head; MACs count convolutions, linear layers and attention matrix products only (norms and activations ignored). EVNet-Tiny's own count was checked against the 6.70M printed by the notebook.

**What the architectural comparison suggests.** EVNet-Tiny uses roughly 40% fewer parameters than ResNet-18 and about 30% less compute, while borrowing the modernised design of ConvNeXt (large depthwise kernels, LayerNorm, GELU, LayerScale, patchify-style downsampling) and a transformer-style global mixing step at the end. Whether that translates into higher accuracy than the ResNets under an identical recipe is exactly what the missing runs would show, and is an open question here.

### Related work

- **ResNet** (He et al., 2016): residual learning with 3×3 convolution blocks; the standard baseline.
- **ConvNeXt** (Liu et al., 2022): a ResNet modernised with 7×7 depthwise convs, LayerNorm, GELU, inverted bottlenecks; EVNet-Tiny's stages 1–3 follow this block design.
- **Vision Transformer** (Dosovitskiy et al., 2021) and **hybrid conv-attention models** such as CoAtNet (Dai et al., 2021): motivate using convolutions early and self-attention at low resolution.
- **MixUp** (Zhang et al., 2018), **CutMix** (Yun et al., 2019), **RandAugment** (Cubuk et al., 2020), **stochastic depth** (Huang et al., 2016), **LayerScale** (Touvron et al., 2021): the training ingredients used here.
- **Spectral residual saliency** (Hou & Zhang, 2007) and **fractal-based augmentation**: the inspiration for the experimental S²-FracMix augmentation.

---

## Limitations and notes

- **The training log is incomplete.** The cosine schedule was set for 200 epochs, but the notebook's recorded output ends at epoch 162 (resumed across several Colab sessions), and epochs 139–150 are missing from the log. The best checkpoint (epoch 137) is the one reported; the learning rate had not fully decayed to its minimum.
- **Best-epoch selection uses the validation set.** Tiny ImageNet's public validation split doubles as the test set here, so 67.40% is a best-epoch-on-val figure and is slightly optimistic compared with a held-out test number.
- **Single seed (42).** There are no confidence intervals, so differences of a few tenths of a percent (such as those between the fine-tuning runs and the baseline) are within normal run-to-run noise.
- **No ResNet baselines yet** (see above).
- **Overfitting after about epoch 100.** Stronger regularisation, a shorter schedule, or higher stochastic depth would be natural next things to try.

---

## Reproducing

1. Open `EVNet_Tiny.ipynb` in Google Colab with a GPU runtime (T4 is enough).
2. Run the cells in order. The notebook downloads Tiny ImageNet, builds the data loaders, defines the model and trains with automatic checkpointing and resume (checkpoints are saved to Google Drive).
3. To run a baseline, set `MODEL_NAME = "resnet18"` (or `"resnet50"`) in the configuration cell and re-run training in a separate checkpoint folder.

```
.
├── EVNet_Tiny.ipynb        # data, model, training, fine-tuning, S²-FracMix experiments
├── assets/
│   └── training_curves.png # accuracy and loss curves from the main run
├── evnet-tiny.bib          # BibTeX for citing this work and the papers it builds on
└── README.md
```

## Citation

If you use this work, please cite the repository:

```bibtex
@misc{singh2026evnettiny,
  author       = {Singh, Ekamveer},
  title        = {{EVNet-Tiny}: A Compact Hybrid {ConvNeXt}--Attention Network for {Tiny ImageNet}},
  year         = {2026},
  howpublished = {\url{https://github.com/Cheese-Singh/EVNet-Tiny}},
  note         = {Source code}
}
```

This entry, together with the references for the architecture and training recipe, is also in [`evnet-tiny.bib`](evnet-tiny.bib).
