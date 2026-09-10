from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "examples" / "ch00"))
sys.path.insert(0, str(ROOT / "examples" / "ch03"))

from kv_cache_memory import GIB, weight_bytes_approx  # noqa: E402
from gpu_spec_sheet import (  # noqa: E402
    A100_80GB_SXM,
    B200,
    DATACENTER_GPUS,
    H200_SXM,
    L40S,
    MI300X,
    fits_bf16_70b_weights,
    vendor_gb_to_gib,
)


class GpuSpecSheetTests(unittest.TestCase):
    def test_vendor_gb_to_gib_80(self):
        self.assertAlmostEqual(vendor_gb_to_gib(80), 80e9 / GIB, places=9)

    def test_70b_weights_do_not_fit_a100_80gb(self):
        weight_gib = weight_bytes_approx(70e9) / GIB
        self.assertAlmostEqual(weight_gib, 130.385160446167, places=6)
        self.assertFalse(fits_bf16_70b_weights(A100_80GB_SXM, weight_gib))

    def test_h200_141gb_is_just_above_70b_weights(self):
        weight_gib = weight_bytes_approx(70e9) / GIB
        self.assertAlmostEqual(H200_SXM.memory_gib, 141e9 / GIB, places=9)
        self.assertTrue(fits_bf16_70b_weights(H200_SXM, weight_gib))
        self.assertLess(H200_SXM.memory_gib - weight_gib, 2.0)

    def test_b200_share_of_dgx(self):
        self.assertEqual(B200.memory_vendor_gb, 180)
        self.assertEqual(B200.bandwidth_tbs, 8)

    def test_l40s_is_gddr_not_hbm_class_bandwidth(self):
        self.assertEqual(L40S.memory_vendor_gb, 48)
        self.assertLess(L40S.bandwidth_tbs, 1.0)
        self.assertGreater(A100_80GB_SXM.bandwidth_tbs, 2.0)

    def test_mi300x_has_largest_memory_in_the_teaching_table(self):
        self.assertEqual(max(g.memory_vendor_gb for g in DATACENTER_GPUS), 192)
        self.assertEqual(MI300X.memory_vendor_gb, 192)


if __name__ == "__main__":
    unittest.main()
