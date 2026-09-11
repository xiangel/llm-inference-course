"""Data-center GPU spec sheet used by chapter 3.1.

This is a teaching table copied from vendor publications. It is not
`nvidia-smi`, not a driver query, and not a promise that your SKU matches.

Capacity in vendor tables is usually decimal GB (10^9 bytes). The book
compares that to GiB (1024^3) because weight math is done in GiB.

Last verified: 2026-09-10.
"""

from __future__ import annotations

from dataclasses import dataclass


GIB = 1024**3
GB = 10**9


@dataclass(frozen=True)
class GpuSpec:
    name: str
    architecture: str
    memory_vendor_gb: float
    bandwidth_tbs: float
    notes: str
    source: str

    @property
    def memory_gib(self) -> float:
        return self.memory_vendor_gb * GB / GIB


# NVIDIA A100 80GB SXM: https://www.nvidia.com/en-us/data-center/a100/
A100_80GB_SXM = GpuSpec(
    name="NVIDIA A100 80GB SXM",
    architecture="Ampere",
    memory_vendor_gb=80,
    bandwidth_tbs=2.039,
    notes="HBM2e；NVLink 600 GB/s；TDP 400W",
    source="https://www.nvidia.com/en-us/data-center/a100/",
)

# NVIDIA H100 SXM: Tensor Core GPU datasheet (SXM 80GB HBM3, ~3.35 TB/s).
# Marketing tables sometimes round bandwidth to 3 TB/s.
H100_SXM = GpuSpec(
    name="NVIDIA H100 SXM",
    architecture="Hopper",
    memory_vendor_gb=80,
    bandwidth_tbs=3.35,
    notes="HBM3；NVLink 900 GB/s；TDP 最高 700W",
    source="https://www.nvidia.com/en-us/data-center/h100/",
)

# NVIDIA H200 SXM: https://www.nvidia.com/en-us/data-center/h200/
H200_SXM = GpuSpec(
    name="NVIDIA H200 SXM",
    architecture="Hopper",
    memory_vendor_gb=141,
    bandwidth_tbs=4.8,
    notes="HBM3e；与 H100 同代计算，主要加显存和带宽；TDP 最高 700W",
    source="https://www.nvidia.com/en-us/data-center/h200/",
)

# NVIDIA DGX B200: 8 Blackwell GPUs, 1,440 GB and 64 TB/s HBM3e total.
# Per-GPU = 180 GB, 8 TB/s. https://www.nvidia.com/en-us/data-center/dgx-b200/
B200 = GpuSpec(
    name="NVIDIA B200 (DGX B200 单卡份额)",
    architecture="Blackwell",
    memory_vendor_gb=1440 / 8,
    bandwidth_tbs=64 / 8,
    notes="由整机 1,440 GB / 64 TB/s 除以 8 得到；含 FP4",
    source="https://www.nvidia.com/en-us/data-center/dgx-b200/",
)

# NVIDIA L40S: https://www.nvidia.com/en-us/data-center/l40s/
L40S = GpuSpec(
    name="NVIDIA L40S",
    architecture="Ada Lovelace",
    memory_vendor_gb=48,
    bandwidth_tbs=0.864,
    notes="GDDR6，不是 HBM；无 NVLink；PCIe 推理/多模态常见",
    source="https://www.nvidia.com/en-us/data-center/l40s/",
)

# AMD Instinct MI300X: https://www.amd.com/en/products/accelerators/instinct/mi300/mi300x.html
MI300X = GpuSpec(
    name="AMD Instinct MI300X",
    architecture="CDNA 3",
    memory_vendor_gb=192,
    bandwidth_tbs=5.3,
    notes="HBM3；TBP 750W；软件栈是 ROCm，不是 CUDA",
    source="https://www.amd.com/en/products/accelerators/instinct/mi300/mi300x.html",
)

DATACENTER_GPUS = (
    A100_80GB_SXM,
    H100_SXM,
    H200_SXM,
    B200,
    L40S,
    MI300X,
)


def vendor_gb_to_gib(vendor_gb: float) -> float:
    return vendor_gb * GB / GIB


def fits_bf16_70b_weights(gpu: GpuSpec, weight_gib: float = 130.385160446167) -> bool:
    """Whether vendor memory exceeds the book's 70B BF16 weight estimate."""
    return gpu.memory_gib > weight_gib


def print_spec_sheet() -> None:
    print("=== Data-center GPU spec sheet (teaching copy) ===")
    print(f"{'name':<36} {'GB':>7} {'GiB':>8} {'TB/s':>7}  arch")
    for gpu in DATACENTER_GPUS:
        print(
            f"{gpu.name:<36} {gpu.memory_vendor_gb:7.1f} "
            f"{gpu.memory_gib:8.2f} {gpu.bandwidth_tbs:7.3f}  {gpu.architecture}"
        )
    print()
    print("70B BF16 weights ≈ 130.39 GiB")
    for gpu in DATACENTER_GPUS:
        flag = "fits" if fits_bf16_70b_weights(gpu) else "too small"
        print(f"  {gpu.name}: {flag}")


if __name__ == "__main__":
    print_spec_sheet()
