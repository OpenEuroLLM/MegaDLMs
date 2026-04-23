"""
Scaling efficiency analysis from Megatron-LM SLURM log files.

Supports two log formats:
  - Newer Megatron-LM: metrics in "wandb: Run summary:" with TFLOPS and
    "Tokens per second per GPU" fields.
  - Older / DiffLM: metrics in "Full wandb run summary:" block with a
    "throughput" (TFLOPs/GPU) field; tokens/s is derived from batch size,
    sequence length, and iteration time.

Fill in GPUS_PER_NODE, PEAK_GPU_TFLOPS, and EXPERIMENTS below, then run:
    python plot_scaling.py
"""

import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import FuncFormatter
from pathlib import Path


GPUS_PER_NODE = 4

# BF16 peak TFLOPs/GPU for MFU calculation (H100 SXM5 = 989).
# Set to None to skip MFU reporting.
PEAK_GPU_TFLOPS = 989.0

# Map number of GPUs to the corresponding SLURM log file path.
EXPERIMENTS = {
    4: "output/dllm_n1_tp4_pp1_gbs16_mbs1_fsdp0/slurm-389149.log",
    8: "output/dllm_n2_tp4_pp1_gbs32_mbs1_fsdp0/slurm-389150.log",
    16: "output/dllm_n4_tp4_pp1_gbs64_mbs1_fsdp0/slurm-389151.log",
    32: "output/dllm_n8_tp4_pp1_gbs128_mbs1_fsdp0/slurm-389152.log",
    64: "output/dllm_n16_tp4_pp1_gbs256_mbs1_fsdp0/slurm-389183.log",
    128: "output/dllm_n32_tp4_pp1_gbs512_mbs1_fsdp0/slurm-389153.log",
    256: "output/dllm_n64_tp4_pp1_gbs1024_mbs1_fsdp0/slurm-389154.log",
    512: "output/dllm_n128_tp4_pp1_gbs2048_mbs1_fsdp0/slurm-389155.log",
    1024: "output/dllm_n256_tp4_pp1_gbs4096_mbs1_fsdp0/slurm-389174.log"
}

# Output filename for the combined figure (None = show interactively).
OUTPUT_FILE = "dllm_jupiter_scaling.png"

# Title for the top (token throughput) bar chart.
PLOT_TITLE = "Token Throughput DiffLM 8B Jupiter (TP 4, PP 1, GAS 16, MBS 1)"

_WORLD_SIZE_RE = re.compile(r"using world size:\s*(\d+)")

# Newer format: "wandb: Run summary:" block
_WANDB_TFLOPS_RE    = re.compile(r"wandb:\s+TFLOPS\s+([\d.]+)")
_WANDB_TOK_GPU_RE   = re.compile(r"wandb:\s+Tokens per second per GPU\s+([\d.]+)")
_WANDB_BS_RE        = re.compile(r"wandb:\s+batch-size\s+(\d+)")
_WANDB_ITER_TIME_RE = re.compile(r"wandb:\s+iteration-time\s+([\d.]+)")

# Older format: "Full wandb run summary:" block (indented "  key: value" lines)
_FULL_SUMMARY_RE = re.compile(r"^Full wandb run summary:\n((?:  .+\n?)*)", re.MULTILINE)
_KV_RE = re.compile(r"^\s+([\w /-]+):\s+([\d.Ee+\-]+)\s*$", re.MULTILINE)

# Per-iteration lines (fallback for seq_len / TFLOPS verification)
_ITER_LINE_RE = re.compile(
    r"iteration\s+\d+/\s*\d+.*?"
    r"elapsed time per iteration \(ms\):\s*([\d.]+).*?"
    r"throughput per GPU \(TFLOP/s/GPU\):\s*([\d.]+).*?"
    r"global batch size:\s*(\d+)",
    re.DOTALL,
)
_ITER_SEQ_LEN_RE = re.compile(r"real input length:\s+([\d.E+]+)")


def _parse_full_summary(text: str) -> dict:
    """Parse 'Full wandb run summary:' block into a float-valued dict."""
    m = _FULL_SUMMARY_RE.search(text)
    if not m:
        return {}
    block = m.group(1)
    result = {}
    for kv in _KV_RE.finditer(block):
        key = kv.group(1).strip()
        try:
            result[key] = float(kv.group(2))
        except ValueError:
            pass
    return result


def parse_log(path: str) -> dict:
    """Return final metrics from a Megatron-LM SLURM log (both formats)."""
    text = Path(path).read_text(errors="replace")

    world_size_m = _WORLD_SIZE_RE.search(text)
    world_size = int(world_size_m.group(1)) if world_size_m else None

    full = _parse_full_summary(text)

    # --- TFLOPs/GPU ---
    m = _WANDB_TFLOPS_RE.search(text)
    if m:
        tflops_per_gpu = float(m.group(1))
    elif "throughput" in full:
        tflops_per_gpu = full["throughput"]
    else:
        # last per-iteration line as fallback
        iter_lines = _ITER_LINE_RE.findall(text)
        if iter_lines:
            tflops_per_gpu = float(iter_lines[-1][1])
        else:
            raise ValueError(f"Could not find TFLOPs/GPU in {path}")

    # --- Iteration time (seconds) ---
    m = _WANDB_ITER_TIME_RE.search(text)
    if m:
        s_per_step = float(m.group(1))
    elif "iteration-time" in full:
        s_per_step = full["iteration-time"]
    else:
        iter_lines = _ITER_LINE_RE.findall(text)
        if iter_lines:
            s_per_step = float(iter_lines[-1][0]) / 1000.0
        else:
            raise ValueError(f"Could not find iteration-time in {path}")

    # --- Global batch size ---
    m = _WANDB_BS_RE.search(text)
    if m:
        global_bs = int(m.group(1))
    elif "batch-size" in full:
        global_bs = int(full["batch-size"])
    else:
        iter_lines = _ITER_LINE_RE.findall(text)
        if iter_lines:
            global_bs = int(iter_lines[-1][2])
        else:
            raise ValueError(f"Could not find batch-size in {path}")

    # --- Sequence length (needed when tok/s/GPU is absent) ---
    seq_len = None
    if "real input length" in full:
        seq_len = int(full["real input length"])
    else:
        m = _ITER_SEQ_LEN_RE.search(text)
        if m:
            seq_len = int(float(m.group(1)))

    # --- Tokens/s/GPU ---
    m = _WANDB_TOK_GPU_RE.search(text)
    if m:
        tok_per_s_per_gpu = float(m.group(1))
    elif seq_len is not None and world_size:
        tok_per_s_per_gpu = (global_bs * seq_len) / s_per_step / world_size
    else:
        raise ValueError(f"Could not determine tokens/s/GPU in {path}")

    tokens_per_step = global_bs * seq_len if seq_len else tok_per_s_per_gpu * s_per_step * world_size
    tok_per_s = tok_per_s_per_gpu * world_size if world_size else None

    return {
        "world_size":        world_size,
        "global_bs":         global_bs,
        "seq_len":           seq_len,
        "tokens_per_step":   tokens_per_step,
        "s_per_step":        s_per_step,
        "tflops_per_gpu":    tflops_per_gpu,
        "tok_per_s_per_gpu": tok_per_s_per_gpu,
        "tok_per_s":         tok_per_s,
    }

def main():
    gpu_counts = sorted(EXPERIMENTS.keys())
    records = {}
    for n_gpus in gpu_counts:
        log_path = EXPERIMENTS[n_gpus]
        print(f"Parsing {log_path} ({n_gpus} GPUs) …")
        rec = parse_log(log_path)
        records[n_gpus] = rec


    # Baseline for efficiency: smallest GPU count
    baseline_gpus  = gpu_counts[0]
    baseline_tok_s = records[baseline_gpus]["tok_per_s"]

    show_mfu = PEAK_GPU_TFLOPS is not None
    mfu_header = f"{'MFU':>7}" if show_mfu else ""
    header = (
        f"{'Nodes':>6}  {'GPUs':>5}  {'GBS':>5}  {'SeqLen':>7}  "
        f"{'Tok/step':>10}  {'s/step':>7}  {'Tok/s/GPU':>10}  "
        f"{'TFLOPs/GPU':>11}  {'Tokens/s':>12}  {'Efficiency':>11}"
        + (f"  {mfu_header}" if show_mfu else "")
    )
    sep = "-" * len(header)
    print()
    print(sep)
    print(header)
    print(sep)

    table_rows = []
    for n_gpus in gpu_counts:
        r     = records[n_gpus]
        nodes = n_gpus // GPUS_PER_NODE
        optimal_tok_s = baseline_tok_s * (n_gpus / baseline_gpus)
        efficiency    = (r["tok_per_s"] / optimal_tok_s * 100.0) if optimal_tok_s else float("nan")
        mfu = (r["tflops_per_gpu"] / PEAK_GPU_TFLOPS * 100.0) if show_mfu else float("nan")

        table_rows.append({
            "n_gpus": n_gpus,
            "nodes": nodes,
            "optimal_tok_s": optimal_tok_s,
            "efficiency": efficiency,
            "mfu": mfu,
            **r,
        })

        tok_step_str = f"{r['tokens_per_step']:>10.0f}" if r["tokens_per_step"] else f"{'N/A':>10}"
        seq_str = f"{r['seq_len']:>7}" if r["seq_len"] else f"{'N/A':>7}"
        mfu_str = f"  {mfu:>6.1f}%" if show_mfu else ""
        print(
            f"{nodes:>6}  {n_gpus:>5}  {r['global_bs']:>5}  {seq_str}  "
            f"{tok_step_str}  {r['s_per_step']:>7.3f}  {r['tok_per_s_per_gpu']:>10.0f}  "
            f"{r['tflops_per_gpu']:>11.1f}  {r['tok_per_s']:>12.0f}  "
            f"{efficiency:>10.1f}%{mfu_str}"
        )
    print(sep)

    n_gpus_arr      = np.array([tr["n_gpus"]          for tr in table_rows])
    tflops_arr      = np.array([tr["tflops_per_gpu"]  for tr in table_rows])
    tok_s_arr       = np.array([tr["tok_per_s"]       for tr in table_rows])
    tok_s_gpu_arr   = np.array([tr["tok_per_s_per_gpu"] for tr in table_rows])
    optimal_arr     = np.array([tr["optimal_tok_s"]   for tr in table_rows])
    efficiency_arr  = np.array([tr["efficiency"]      for tr in table_rows])

    x_labels = n_gpus_arr.astype(str)
    x_pos    = np.arange(len(n_gpus_arr))

    mil_fmt = FuncFormatter(lambda x, _: f"{x/1e6:.1f}M")

    fig = plt.figure(figsize=(14, 10))
    gs  = gridspec.GridSpec(2, 1, figure=fig, hspace=0.5)

    ax1 = fig.add_subplot(gs[0])
    bars1 = ax1.bar(x_pos, tok_s_arr, color="#AED6F1", edgecolor="white", width=0.6,
                    label="_nolegend_")
    ax1.bar_label(bars1, labels=[f"{v/1e6:.2f}M" for v in tok_s_arr],
                  padding=3, fontsize=9, color="black")
    ax1.plot(x_pos, tok_s_arr,   marker="o", color="#2874A6", linewidth=2,
             linestyle="-", label="Measured Tok/s")
    ax1.plot(x_pos, optimal_arr, marker="o", color="#E67E22", linewidth=2,
             linestyle=":", label="Optimal scaling")
    ax1.set_ylabel("Tokens / second", fontsize=11, color="black")
    ax1.set_title(PLOT_TITLE, fontsize=11, fontweight="bold", color="black")
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(x_labels)
    ax1.set_ylim(0, max(optimal_arr) * 1.25)
    ax1.yaxis.set_major_formatter(mil_fmt)
    ax1.grid(axis="y", linestyle="--", alpha=0.5)
    ax1.spines[["top", "right"]].set_visible(False)
    ax1.tick_params(colors="black")

    ax1r = ax1.twinx()
    ax1r.plot(x_pos, efficiency_arr, marker="s", color="#C0392B", linewidth=2,
              linestyle=":", label="Efficiency (%)")
    ax1r.set_ylabel("Efficiency (%)", fontsize=11, color="black")
    ax1r.tick_params(axis="y", labelcolor="black")
    ax1r.set_ylim(0, 115)
    ax1r.axhline(100, color="#C0392B", linewidth=0.8, linestyle="--", alpha=0.4)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines1r, labels1r = ax1r.get_legend_handles_labels()
    ax1.legend(lines1 + lines1r, labels1 + labels1r, loc="upper left", fontsize=9,
               bbox_to_anchor=(0.0, 0.88))

    # Bar plot: TFLOPs/GPU bars + Tokens/s/GPU line
    ax2 = fig.add_subplot(gs[1])
    bars2 = ax2.bar(x_pos, tflops_arr, color="#228B22", edgecolor="white", width=0.6,
                    label="TFLOPs/s / GPU")
    ax2.bar_label(bars2, fmt="%.1f", padding=3, fontsize=9, color="black")
    ax2.set_xlabel("Number of GPUs", fontsize=11, color="black")
    ax2.set_ylabel("TFLOPs/s / GPU", fontsize=11, color="black")
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(x_labels)
    ax2.set_ylim(0, max(tflops_arr) * 1.2)
    ax2.grid(axis="y", linestyle="--", alpha=0.5)
    ax2.spines[["top", "right"]].set_visible(False)
    ax2.tick_params(colors="black")

    ax2r = ax2.twinx()
    tok_gpu_line, = ax2r.plot(x_pos, tok_s_gpu_arr, marker="D", color="#8E44AD",
                              linewidth=2, linestyle=":", label="Tokens/s / GPU")
    for xi, val in zip(x_pos, tok_s_gpu_arr):
        ax2r.annotate(f"{val:,.0f}", (xi, val),
                      textcoords="offset points", xytext=(0, 7),
                      ha="center", fontsize=11, color="#8E44AD",
                      fontweight="bold")
    ax2r.set_ylabel("Tokens/s / GPU", fontsize=11, color="black")
    ax2r.tick_params(axis="y", labelcolor="black")
    ax2r.set_ylim(0, max(tok_s_gpu_arr) * 1.5)
    ax2r.spines[["top"]].set_visible(False)

    tflops_patch = plt.Rectangle((0, 0), 1, 1, fc="#228B22")
    ax2.legend(
        [tflops_patch, tok_gpu_line],
        ["TFLOPs/s / GPU", "Tokens/s / GPU"],
        loc="lower right", fontsize=9,
        labelcolor="black",
    )

    if OUTPUT_FILE:
        fig.savefig(OUTPUT_FILE, dpi=150, bbox_inches="tight")
        print(f"\nFigure saved to {OUTPUT_FILE}")
    else:
        plt.show()


if __name__ == "__main__":
    main()