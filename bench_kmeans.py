import argparse
import csv
from typing import Callable, Tuple

import torch
import triton


try:
    import torch.cuda.nvtx as nvtx
    HAS_NVTX = True
except ImportError:
    HAS_NVTX = False

def nvtx_range(name: str):
    class _CM:
        def __enter__(self):
            if HAS_NVTX: nvtx.range_push(name)
        def __exit__(self, *_):
            if HAS_NVTX: nvtx.range_pop()
    return _CM()


# Deepseek configs
CFG = dict(
    num_attention_heads = 128,
    index_n_heads       = 64,
    index_head_dim      = 128,
    kv_lora_rank        = 512,
    qk_rope_head_dim    = 64,
    v_head_dim          = 128,
    index_topk          = 2048,
)

HEAD_DIM   = CFG["index_head_dim"]
N_HEADS    = CFG["index_n_heads"]
INDEX_TOPK = CFG["index_topk"]

# Expected kernels names, may change 
from kmeans_pytorch     import pytorch_kmeans_bf16
from kmeans_triton_bf16 import triton_kmeans_bf16
from kmeans_triton_fp8  import triton_kmeans_fp8

# Mock data
def make_inputs(
    batch: int,
    seq_len: int,
    n_heads: int,
    head_dim: int,
    n_clusters: int,
    device: str = "cuda",
    seed: int = 42,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns:
        keys_bf16  [B, H, S, D]  bfloat16
        keys_fp8   [B, H, S, D]  float8_e4m3fn
        centroids  [K, D]        bfloat16

    Centroids are the first K tokens of head 0 — deterministic and identical
    across all three kernels so assignment comparison is valid.
    """
    torch.manual_seed(seed)
    keys_bf16 = torch.randn(
        batch, n_heads, seq_len, head_dim,
        dtype=torch.bfloat16, device=device,
    )
    keys_fp8  = keys_bf16.to(torch.float8_e4m3fn)
    centroids = keys_bf16[0, 0, :n_clusters, :].clone()
    return keys_bf16, keys_fp8, centroids

def run_correctness(
    keys_bf16: torch.Tensor,
    keys_fp8:  torch.Tensor,
    centroids: torch.Tensor,
) -> dict:
    """
    All three kernels receive the same fixed initial centroids (.clone() per
    call prevents in-place updates from cross-contaminating).

    BF16: assignment match should be 1.0000 — any deviation is a kernel bug.
    FP8:  we would see some difference due to precision loss, measure how much that is
    """
    fixed = centroids.clone()

    ref_asgn, ref_cents = pytorch_kmeans_bf16(keys_bf16, fixed.clone())
    tri_asgn, tri_cents = triton_kmeans_bf16 (keys_bf16, fixed.clone())
    fp8_asgn, fp8_cents = triton_kmeans_fp8  (keys_fp8,  fixed.clone())

    def assignment_match(a, b) -> float:
        return (a == b).float().mean().item()

    def centroid_stats(ref, other) -> dict:
        err = (ref.float() - other.float()).abs()
        return dict(
            max_abs       = err.max().item(),
            mean_abs      = err.mean().item(),
            allclose_1pct = torch.allclose(ref.float(), other.float(), atol=1e-2, rtol=1e-2),
        )

    return dict(
        bf16_assignment_match = assignment_match(ref_asgn, tri_asgn),
        bf16_centroid         = centroid_stats(ref_cents, tri_cents),
        fp8_assignment_match  = assignment_match(ref_asgn, fp8_asgn),
        fp8_centroid          = centroid_stats(ref_cents, fp8_cents),
    )

def bench_fn(fn: Callable, *args, warmup: int = 25, rep: int = 100) -> float:
    """Returns median latency in milliseconds."""
    return triton.testing.do_bench(
        lambda: fn(*args),
        warmup=warmup,
        rep=rep,
        return_mode="median",
    )

def run_latency(keys_bf16: torch.Tensor, keys_fp8:  torch.Tensor, centroids: torch.Tensor) -> dict:
    ms_pytorch  = bench_fn(pytorch_kmeans_bf16, keys_bf16, centroids.clone())
    ms_tri_bf16 = bench_fn(triton_kmeans_bf16,  keys_bf16, centroids.clone())
    ms_tri_fp8  = bench_fn(triton_kmeans_fp8,   keys_fp8,  centroids.clone())
    return dict(
        pytorch_bf16_ms   = ms_pytorch,
        triton_bf16_ms    = ms_tri_bf16,
        triton_fp8_ms     = ms_tri_fp8,
        speedup_bf16      = ms_pytorch / ms_tri_bf16,
        speedup_fp8       = ms_pytorch / ms_tri_fp8,
        bf16_vs_fp8_ratio = ms_tri_bf16 / ms_tri_fp8,
    )


# Measure memory bandwidth using NCU nvtx
def run_nvtx_profile(
    keys_bf16: torch.Tensor,
    keys_fp8:  torch.Tensor,
    centroids: torch.Tensor,
) -> None:
    torch.cuda.synchronize()
    with nvtx_range("kmeans_pytorch_bf16"):
        pytorch_kmeans_bf16(keys_bf16, centroids.clone())
    torch.cuda.synchronize()
    with nvtx_range("kmeans_triton_bf16"):
        triton_kmeans_bf16(keys_bf16, centroids.clone())
    torch.cuda.synchronize()
    with nvtx_range("kmeans_triton_fp8"):
        triton_kmeans_fp8(keys_fp8, centroids.clone())
    torch.cuda.synchronize()
    print("NVTX profile pass complete — inspect ncu for dram__bytes counters.")


SEQ_LENS = [512, 1024, INDEX_TOPK, 4096, 8192]
K_VALS   = [8, 16, 32]
BATCH    = 1


def run_sweep(device: str = "cuda") -> list[dict]:
    rows = []
    for seq_len in SEQ_LENS:
        for K in K_VALS:
            keys_bf16, keys_fp8, centroids = make_inputs(
                batch=BATCH, seq_len=seq_len,
                n_heads=N_HEADS, head_dim=HEAD_DIM,
                n_clusters=K, device=device,
            )
            lat  = run_latency(keys_bf16, keys_fp8, centroids)
            corr = run_correctness(keys_bf16, keys_fp8, centroids)

            rows.append(dict(
                seq_len = seq_len,
                K       = K,
                **{f"lat_{k}":  v for k, v in lat.items()},
                **{f"corr_{k}": v for k, v in corr.items() if isinstance(v, (int, float))},
            ))
            print(
                f"  seq={seq_len:5d}  K={K:2d} | "
                f"pytorch={lat['pytorch_bf16_ms']:.3f}ms  "
                f"tri_bf16={lat['triton_bf16_ms']:.3f}ms  "
                f"tri_fp8={lat['triton_fp8_ms']:.3f}ms | "
                f"speedup_fp8={lat['speedup_fp8']:.2f}x | "
                f"asgn_bf16={corr['bf16_assignment_match']:.4f}  "
                f"asgn_fp8={corr['fp8_assignment_match']:.4f}"
            )
    return rows

def print_summary(rows: list[dict]) -> None:
    header = (
        f"{'seq':>6} {'K':>3} | "
        f"{'pt_bf16':>9} {'tri_bf16':>9} {'tri_fp8':>9} | "
        f"{'spdup_bf16':>10} {'spdup_fp8':>9} | "
        f"{'asgn_bf16':>10} {'asgn_fp8':>10}"
    )
    sep = "-" * len(header)
    print("\n" + sep)
    print("K-Means Benchmark  |  memory BW: run --profile then ncu")
    print(sep)
    print(header)
    print(sep)
    for r in rows:
        print(
            f"{r['seq_len']:>6} {r['K']:>3} | "
            f"{r['lat_pytorch_bf16_ms']:>8.3f}ms "
            f"{r['lat_triton_bf16_ms']:>8.3f}ms "
            f"{r['lat_triton_fp8_ms']:>8.3f}ms | "
            f"{r['lat_speedup_bf16']:>9.2f}x "
            f"{r['lat_speedup_fp8']:>8.2f}x | "
            f"{r['corr_bf16_assignment_match']:>10.4f} "
            f"{r['corr_fp8_assignment_match']:>10.4f}"
        )
    print(sep + "\n")

def parse_args():
    p = argparse.ArgumentParser(description="K-Means clustering benchmark")
    p.add_argument("--profile", action="store_true",
                   help="Emit NVTX regions for ncu memory BW profiling (skips sweep)")
    p.add_argument("--seq",     type=int, nargs="+", default=SEQ_LENS,
                   help="Sequence lengths to sweep")
    p.add_argument("--k",       type=int, nargs="+", default=K_VALS,
                   help="Number of clusters to sweep")
    p.add_argument("--batch",   type=int, default=BATCH)
    return p.parse_args()

def main():
    args = parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device required.")

    device = "cuda"
    print(f"\nDevice     : {torch.cuda.get_device_name(0)}")
    print(f"Dims       : heads={N_HEADS}, head_dim={HEAD_DIM}, index_topk={INDEX_TOPK}, batch={args.batch}\n")

    if args.profile:
        keys_bf16, keys_fp8, centroids = make_inputs(
            batch=args.batch, seq_len=INDEX_TOPK,
            n_heads=N_HEADS, head_dim=HEAD_DIM,
            n_clusters=16, device=device,
        )
        run_nvtx_profile(keys_bf16, keys_fp8, centroids)
        return

    global SEQ_LENS, K_VALS, BATCH
    SEQ_LENS = args.seq
    K_VALS   = args.k
    BATCH    = args.batch

    print("Running sweep...")
    rows = run_sweep(device=device)
    print_summary(rows)

if __name__ == "__main__":
    main()
