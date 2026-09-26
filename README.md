# Understanding the Dynamics of Demonstration Conflict in In-Context Learning

## Overview

We investigate how language models process conflicting demonstrations during in-context learning (ICL).

**Key Findings:**
- Single corrupted demonstration causes performance degradation heavily and consistently.
- We reveal a **two-phase computational structure**: models encode both correct and corrupted rules in intermediate layers (conflict creation), then fail to properly resolve them in late layers (conflict resolution).
- Two types of attention heads identified:
  - **Vulnerability Heads** (early-middle layers): Create positional bias and high corruption sensitivity
  - **Susceptible Heads** (late layers): Reduce support for correct predictions under corruption
- Ablating identified heads mitigates performance degradation, confirming their causal roles

## Quick Start

### Clean Baseline Evaluation
Establishes model ICL capabilities without corruption.

```bash
./run/0-run_clean_baseline_test.sh
```
It helps validate that models exhibit near-chance 0-shot performance but strong few-shot performance, confirming genuine demonstration reliance.

---

### Corruption Evaluation
Tests model robustness to single-position corruption across 4/6/8-shot scenarios.

```bash
./run/1-run_corruption_test.sh
```

It measures performance degradation when one demonstration is corrupted while maintaining majority rule. Evaluates position-specific corruption effects.

### Mech Interp Pipeline
Identifies vulnerability and susceptible attention heads through mechanistic analysis.

```bash
./run/2-run_interp_pipeline.sh
```

**Pipeline Steps:**
1. **`extract_acs_metrics.py`** - Extracts attention allocation and corruption sensitivity scores for each head
2. **`find_susc_heads.py`** - Computes susceptibility scores via logit attribution
3. **`comprehensive_head_ablation.py`** - Validates causal roles through targeted ablation

---

## Additional Mechanistic Analysis Tools

### Linear Probes (`interp/probes/`)
Train linear classifiers to detect rule encoding across layers.

```python
# Train probes
python interp/probes/linear_probe_trainer.py \
    --model_name Qwen/Qwen3-4B \
    --dataset operator_induction_text

# Evaluate
python interp/probes/linear_probe_evaluator.py \
    --model_name Qwen/Qwen3-4B \
    --dataset operator_induction_text
```
---

### Logit Lens (`interp/logit_lens/`)
Project intermediate representations to observe prediction formation layer-by-layer.

```python
python interp/logit_lens/run_logit_lens.py \
    --model_name Qwen/Qwen3-4B \
    --dataset operator_induction_text \
    --n_shot 4
```

---

