<div align="center">

# 🔮 Prism Attention: Proposal-Refined Index Sharing Mechanism for Efficient LLMs Inference

[![NeurIPS 2026](https://img.shields.io/badge/NeurIPS%202026-Accepted-6f42c1.svg)](#-news)
[![arXiv](https://img.shields.io/badge/arXiv-2610.22985-b31b1b.svg?logo=arxiv&logoColor=white)](https://arxiv.org/abs/2610.22985)
[![Paper PDF](https://img.shields.io/badge/Paper-PDF-informational.svg)](https://arxiv.org/pdf/2610.22985)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

</div>

---

## 🎉 News

- **[2026-09]** Prism Attention has been **accepted to NeurIPS 2026**. 🎉
- **[2026-09]** Code release: Qwen3 decoding model (`prism` / `indexcache` / `dense` modes), INT2 key-cache Triton kernels, offline IndexCache calibration, and the reasoning / LongBench evaluation harnesses.

## 📖 Quick Scan

Token-level sparse attention makes long-context decoding cheaper, but finding the top-$\kappa$ tokens still requires scoring the **entire** KV cache at **every** layer. Cross-layer index sharing avoids that by computing indices only at a few anchor layers and reusing them downstream — at the cost of accuracy, because attention patterns drift with depth and reused indices progressively miss the layer-wise optimum.

**Prism Attention** replaces rigid reuse with a coarse-to-fine scheme. A *Proposal Layer* emits an **expanded** candidate pool of $n = \rho\kappa$ tokens scored against an **INT2-compressed** key cache; each downstream *Refined Layer* re-scores only those $n$ candidates with its own query and picks its own top-$\kappa$. Selection cost per refined layer drops from $\mathcal{O}(L_t)$ to $\mathcal{O}(\rho\kappa)$ — constant in context length — while layer-wise adaptability is preserved.

![Prism Attention overview](assets/method.png)

The design is motivated by a measurable property of attention dynamics: **inter-layer salient token shift is highly localized**. The exact top-$\kappa$ sets differ across layers, but the core high-attention tokens are stable and most changes land near the boundary of the previously selected region. The index overlap between an anchor layer and its downstream layers decays sharply with distance, and faster at longer contexts.

Prism Attention features:
- **Dynamic Index Refinement** — refined layers perform a bounded local top-$\kappa$ search inside the inherited proposal pool instead of inheriting a frozen index set.
- **Quantized Proposal Generation** — proposal scoring runs on a group-wise asymmetric INT2 key cache with a fused dequant + QK^T + GQA max-reduce Triton kernel, cutting the memory traffic of full-context scanning.
- **Training-free** — only an offline layer-partition calibration is needed; model weights are untouched.
- **Drop-in HF model** — `PrismDecodingQwen3ForCausalLM` loads any Qwen3 checkpoint via `from_pretrained`, with `prism`, `indexcache`, and `dense` modes in one class for apples-to-apples comparison.

## 📂 Repository Structure

```
PrismAttention/
├── prism_attn/                                  # installable package (pip install -e .)
│   ├── decode_sparse/
│   │   ├── qwen3/
│   │   │   ├── configuration_qwen3_prismattn.py # PrismQwen3Config
│   │   │   └── modeling_qwen3_prismattn_inference.py
│   │   │                                        #   PrismDecodingQwen3ForCausalLM,
│   │   │                                        #   Qwen3PrismAttention, reuse-map resolution
│   │   ├── attn_gate_inf.py                     # C2AttnGate: proposal / top-k selector
│   │   ├── cache_quant_utils.py                 # KC2CompressionCacheBuffered (INT2 K cache)
│   │   ├── attention_forward_dense.py           # FlashAttention prefill + left-pad flash decode
│   │   └── attention_forward_sparse.py          # token-sparse decode path
│   ├── kernels/
│   │   ├── quant/
│   │   │   ├── int2_quant.py                    # int2_quantize: 2-bit group-wise K quantization
│   │   │   └── int2_qk_maxreduce.py             # int2_attention_score{,_indexed}: fused
│   │   │                                        #   dequant + QK^T + GQA max-reduce (+ sink/recent)
│   │   └── varlen/
│   │       ├── flash_decode_varlen_left_pad_max_v2.py  # flash_decode_leftpad
│   │       └── token_sparse_decode_optimized.py        # token_sparse_flash_decode_optimized
│   ├── modules/                                 # Triton fused RMSNorm + rotary helpers
│   └── utils.py                                 # HF outputs extended with k_compressed_cache
├── calibrate/
│   ├── data/openr1_1000.jsonl                   # example calibration corpus
│   └── indexcache/                              # offline Proposal/Refined layer calibration
│       ├── offline_calibration.py
│       ├── run_calibration.sh
│       └── output-g4/indexcache_config.json     # shipped Qwen3-4B g4 partition
├── eval/
│   ├── reasoning_tasks/                         # AIME24/25, MATH500, GPQA, OlympiadBench, LiveCodeBench
│   │   ├── scripts/eval_prism_attention.sh      # entry point
│   │   ├── parallel_run_hf.py                   # multi-GPU launcher, per-task bs / sample count
│   │   ├── eval_hf.py                           # single-GPU generation worker
│   │   └── summary_results.py                   # Pass@1 aggregation
│   └── longbench/                               # LongBench / LongBench-E
│       ├── eval_longbench_prism.sh              # entry point
│       ├── pred_threshold_decode.py             # prediction
│       ├── eval.py  metrics.py  average_score.py
│       └── config/                              # dataset2prompt / dataset2maxlen
└── assets/                                      # figures
```

## 💻 System Environment

The environment used to produce the paper numbers:

| Component | Specification | Notes |
| :--- | :--- | :--- |
| **OS** | Ubuntu 22.04 LTS | |
| **Python** | 3.11 | Recommended |
| **PyTorch** | 2.4.0 | `torch>=2.4.0` |
| **Triton** | 3.0.0 | **Required** — all custom kernels are Triton |
| **transformers** | 4.51.0 | Pinned; the model subclasses HF Qwen3 |
| **CUDA** | 12.1+ | **Required** |
| **GPU** | NVIDIA H800 / A100 / H100 (80GB) | Paper latency measured on H800 |

> The INT2 scoring kernel uses an inline PTX `cvt.rn.f16x2.e4m3x2` trick, so an SM80+ (Ampere or newer) GPU is required. Decode kernels assume a GQA group size of at most 16 query heads per KV head.

## ⚙️ Installation

```bash
# 1. Environment
conda create -n prism python=3.11 -y
conda activate prism

# 2. PyTorch
pip install torch==2.4.0

# 3. Dependencies + the package itself
git clone https://github.com/tianzhenxu/PrismAttention.git && cd PrismAttention
pip install -r requirements.txt
pip install flash-attn==2.5.8 --no-build-isolation   # prefill + dense decode baseline
pip install -e .
```

## ⚡ Quick Start

Prefill always runs dense (FlashAttention varlen); sparsity applies to the decoding steps.

```python
import torch
from transformers import AutoTokenizer
from prism_attn import PrismDecodingQwen3ForCausalLM

model_name = "Qwen/Qwen3-4B"
tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")

model = PrismDecodingQwen3ForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    attention_method="prism",       # "prism" | "indexcache" | "dense"
    token_budget=2048,              # final sparse budget kappa
    candidate_factor=2.0,           # proposal expansion rho -> pool of rho * kappa
    proposal_layers=[0, 7, 8, 13, 14, 18, 21, 28, 35],   # from offline calibration
    sink_size=64,
    recent_size=192,
    use_flash_rope=True,            # fused rotary kernel
    fused_norm=True,                # fused Triton RMSNorm
).cuda().eval()

prompt = "Solve: find all integer solutions of x^2 - 3y^2 = 1 with |x| < 50."
inputs = tokenizer([prompt], return_tensors="pt").to("cuda")
out = model.generate(**inputs, max_new_tokens=512, do_sample=False)
print(tokenizer.decode(out[0], skip_special_tokens=True))
```

The remaining options (`reuse_map`, `start_layer`, `int2_group_size`, `max_cache_length`, ...) are defined in `prism_attn/decode_sparse/qwen3/configuration_qwen3_prismattn.py` and can all be passed straight to `from_pretrained`.

### Environment overrides

The shell scripts configure Prism through the environment rather than through `from_pretrained`; these take precedence over the config values.

```bash
export PRISM_CANDIDATE_FACTOR=2.0                       # overrides candidate_factor
export PRISM_PROPOSAL_LAYERS=0,7,8,13,14,18,21,28,35    # overrides reuse_map / proposal_layers

# IndexCache baseline needs rigid reuse:
# export PRISM_CANDIDATE_FACTOR=1.0
```

Reuse-map resolution order: `PRISM_PROPOSAL_LAYERS` → `config.reuse_map` → `config.proposal_layers` → no sharing.

## 🧭 Offline Calibration (Proposal / Refined partition)

Which layers act as proposal layers is decided once, offline, by a greedy/beam search that minimizes token-level cross-entropy on a held-out calibration set — the same training-free procedure IndexCache uses, adapted to Qwen3. Edit the model, dataset and search settings at the top of the script before running:

```bash
cd calibrate/indexcache
bash run_calibration.sh
```

The run writes `indexcache_config.json` with `f_layers` (proposal layers), `s_layers` and `s_to_f_mapping`. A calibrated Qwen3-4B g4 partition is shipped at `calibrate/indexcache/output-g4/indexcache_config.json`. Feed `f_layers` into inference either as `proposal_layers=[...]` or via `PRISM_PROPOSAL_LAYERS`.

Calibration is memory-intensive at long sequence lengths; `offline_calibration.py` supports `torch.distributed` and falls back to single-GPU when `RANK` is unset.

## ⚖️ Evaluation

### Reasoning benchmarks


```bash
cd eval/reasoning_tasks
bash scripts/eval_prism_attention.sh
```


### LongBench

```bash
cd eval/longbench
bash eval_longbench_prism.sh
```



## 📝 Citation

If you find Prism Attention useful in your research, please consider citing:

```bibtex
@article{tian2026prism,
  title={Prism Attention: Proposal-Refined Index Sharing Mechanism for Efficient LLMs Inference},
  author={Zhenxu Tian and Kebin Liu and Zhengwu Yang and Yi Su and Qingqing Dang and Yanlin Sha and Kaipeng Deng and Dianhai Yu and Yanjun Ma and Juntao Li and Min Zhang},
  journal={arXiv preprint arXiv:2610.22985},
  year={2026}
}
```

