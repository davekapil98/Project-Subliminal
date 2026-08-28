from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("stage1_host", ROOT / "scripts/stage1_host.py")
assert SPEC is not None and SPEC.loader is not None
HOST = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(HOST)


def test_gb100_product_name_variants() -> None:
    assert HOST._is_gb100_product("NVIDIA GB100, 81559 MiB, 590.12")
    assert HOST._is_gb100_product("NVIDIA B100, 81559 MiB, 590.12")
    assert HOST._is_gb100_product("NVIDIA B200, 183380 MiB, 590.12")
    assert not HOST._is_gb100_product("NVIDIA RTX 3050 Laptop GPU, 4096 MiB, 580.95")
    assert not HOST._is_gb100_product("NVIDIA A100-SXM4-80GB, 81920 MiB, 580.95")
