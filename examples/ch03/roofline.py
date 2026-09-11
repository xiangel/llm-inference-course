"""Roofline helpers for Part 3.

Teaching arithmetic, not a profiler. Peak numbers are labeled vendor peaks
so they can be swapped when a datasheet changes.

Last verified: 2026-09-10.
H100 SXM dense BF16 ≈ 989 TFLOPS (NVIDIA H100; the ~1979 number is *with sparsity*).
H100 SXM HBM3 bandwidth = 3.35 TB/s (same sheet as chapter 3.1).
"""

from __future__ import annotations

# NVIDIA H100 Tensor Core GPU: dense BF16 without sparsity.
H100_DENSE_BF16_TFLOPS = 989.0
H100_HBM_TBS = 3.35
TB = 1e12


def flop_per_byte_weight_gemm(tokens: int, dtype_bytes: int = 2) -> float:
    """Arithmetic intensity of a weight GEMM reused across ``tokens`` rows.

    Each parameter is one multiply-add (2 FLOPs) and is read once
    (``dtype_bytes``). Decode uses tokens=1; Prefill uses the prompt length.
    """
    return 2 * tokens / dtype_bytes


def ridge_point(peak_flops: float, peak_bandwidth_bytes: float) -> float:
    """Intensity where the compute roof meets the bandwidth roof."""
    return peak_flops / peak_bandwidth_bytes


def roofline_flops(
    intensity: float,
    peak_flops: float,
    peak_bandwidth_bytes: float,
) -> float:
    return min(peak_flops, intensity * peak_bandwidth_bytes)


def h100_ridge_flop_per_byte() -> float:
    return ridge_point(H100_DENSE_BF16_TFLOPS * 1e12, H100_HBM_TBS * TB)


def print_worked_examples() -> None:
    ridge = h100_ridge_flop_per_byte()
    dec = flop_per_byte_weight_gemm(1)
    pre = flop_per_byte_weight_gemm(1024)
    peak = H100_DENSE_BF16_TFLOPS * 1e12
    bw = H100_HBM_TBS * TB
    print("H100 ridge FLOP/byte", ridge)
    print("decode AI", dec, "bound TFLOPS", roofline_flops(dec, peak, bw) / 1e12)
    print("prefill S=1024 AI", pre, "bound TFLOPS", roofline_flops(pre, peak, bw) / 1e12)
    print("decode is memory bound", dec < ridge)
    print("prefill 1024 is compute bound", pre > ridge)


if __name__ == "__main__":
    print_worked_examples()
