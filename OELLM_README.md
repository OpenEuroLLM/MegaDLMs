# OpenEuroLLM - MegaDLM

This document tracks changes made to the upstream MegaDLMs codebase for the OpenEurLLM project on the JUPITER cluster.

## Training

Submit training
```bash
sbatch pretrain_difflm.job
```


## Installation

Build container:
```
export APPTAINER_TMPDIR=/dev/shm/$USER && mkdir -p /dev/shm/$USER/
export APPTAINER_CACHEDIR=/e/scratch/e-sta-openeurollm/$USER/apptainer

OUTPUT_SIF=MegaDLM-24-11.sif
apptainer build $OUTPUT_SIF container.def
```

---

## Bug fixes

### Variable-length sequence not aligned to tensor-parallel size (`gpt_model.py`)

**File:** `megatron/core/models/difflm/gpt_model.py`  
**Method:** `GPTModel.use_varilen_data`

**Problem:**
`use_varilen_data` randomly truncates the input sequence with probability `difflm_varilen_prob` (default 1%). It drew `random_length` uniformly from `[2, seq_length]`. In `difflm-noshift` mode the model trims the first token (`input_ids = input_ids[:, 1:]`), so the sequence that reaches the embedding layer has length `random_length - 1`. With sequence-parallel enabled, `reduce_scatter_to_sequence_parallel_region` requires this first dimension to be divisible by the tensor-parallel (TP) size. An arbitrary `random_length - 1` is almost never divisible by TP > 1, causing an `AssertionError` in `_reduce_scatter_along_first_dim`.

**Fix:**  
After drawing `random_length`, round it so that `random_length - 1` is the largest multiple of `tp_size` that does not exceed the drawn value (minimum `tp_size`):

```python
tp_size = self.args.tensor_model_parallel_size
effective_len = max(tp_size, (random_length - 1) // tp_size * tp_size)
random_length = effective_len + 1
```

The seeded `np.random.randint` call itself is unchanged, preserving consistent RNG state across data-parallel ranks.

---

### `apply_rope_fusion` fails with NGC containers >= 26.03 (TE >= 2.x)

**File:** `megatron/core/extensions/transformer_engine.py`

**Problem:**
Upgrading from NGC `24.11` to `26.03` (TransformerEngine 2.13.0) breaks `apply_rope_fusion`. The code imports `FusedRoPEFunc` from `transformer_engine.pytorch.attention`, but that path no longer exists in TE 2.x, causing both `fused_apply_rotary_pos_emb` and `fused_apply_rotary_pos_emb_thd` to silently fall back to `None`. The validation check in `TransformerConfig.__post_init__` then raises:
```
ValueError: apply_rope_fusion is not available. Please install TE >= 1.4 or Apex.
```


**Workaround:**
Add `--no-rope-fusion` to the job script's `GPT_MODEL_ARGS`.

**Proper fix:**
Add `export UB_SKIPMC=1`  # GH200+TE: skip CUDA Multicast, use IPC-based comm+GEMM overlap
Update the import in `transformer_engine.py` to fall back to the new TE 2.x location:
```python
try:
    from transformer_engine.pytorch.attention import FusedRoPEFunc
except ImportError:
    from transformer_engine.pytorch.rope import FusedRoPEFunc
```
(The exact new module path needs verification inside the 26.03 container.)


### `--tp-comm-overlap` incompatible with difflm-noshift

**Problem:**
```bash
[rank20]: RuntimeError: /workspace/TransformerEngine/transformer_engine/pytorch/csrc/extensions/comm_gemm_overlap.cpp:263 in function copy_into_buffer: Assertion failed: _ubufs[_tp_id].numel() == input_size. Tried to copy an invalid tensor into a local chunk of a Userbuffers buffer (input_size=16777216, local_ubuf_size=16781312)
```
`local_ubuf_size = 16781312 = 4097 × 4096` TE allocates the userbuffer based on `--seq-length` 4097
`input_size = 16777216 = 4096 × 4096`, the actual tensor is 4096 because difflm-noshift trims the first token
`--tp-comm-overlap` is incompatible with difflm-noshift's seq trim as long as `--seq-length` is 4097 but the actual compute seq is 4096. You'd need to align these: either pass `--seq-length` 4096 and load 4097-token samples at the data level (so after trimming you get 4096), or live without the comm overlap for now.
