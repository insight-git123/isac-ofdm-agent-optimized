"""评测条目1-3：numerology提取、CP时长、参数一致性（不依赖外部 PDF）"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tasks.task1_run import extract_numerology, enrich, SOURCES
from src.validation.validators import (
    validate_numerology,
    validate_cp_duration,
    compute_normal_cp_us,
    EXPECTED_NUMEROLOGY,
)


# 内嵌样例（模拟真实 PDF 提取后的文本格式）
SAMPLE_TEXT = """
4.2 Numerologies
MultipleOFDMnumerologiesaresupportedasgivenbyTable4.2-1.

Table4.2-1:Supportedtransmissionnumerologies.
0 15 Normal
1 30 Normal
2 60 Normal,Extended
3 120 Normal
4 240 Normal
5 480 Normal
6 960 Normal

4.3 Frame structure
"""


def test_entry1_numerology_table():
    """条目1：SCS-CP 映射正确，扩展CP仅 μ=2。"""
    rows = enrich(extract_numerology(SAMPLE_TEXT))
    assert len(rows) == 7
    by_mu = {r["mu"]: r for r in rows}
    assert by_mu[0]["scs_khz"] == 15 and by_mu[0]["cp_types"] == ["normal"]
    assert by_mu[1]["scs_khz"] == 30 and by_mu[1]["cp_types"] == ["normal"]
    assert by_mu[2]["scs_khz"] == 60 and set(by_mu[2]["cp_types"]) == {"normal", "extended"}
    assert by_mu[3]["scs_khz"] == 120 and by_mu[3]["cp_types"] == ["normal"]
    v = validate_numerology(rows)
    assert v["passed"], v["errors"]


def test_entry2_cp_duration():
    """条目2：正常CP时长与3GPP计算一致（相对误差≤1%）。"""
    rows = enrich(extract_numerology(SAMPLE_TEXT))
    v = validate_cp_duration(rows, tol=0.01)
    assert v["passed"], v["errors"]
    by_mu = {r["mu"]: r for r in rows}
    assert abs(by_mu[2]["cp_duration_us"] - 1.17) < 0.01


def test_entry3_release_lock_and_source():
    """条目3前置：Rel-18 已锁定，L1来源可溯源。"""
    assert SOURCES["release_lock"] == "Rel-18"
    ts = next(s for s in SOURCES["sources"] if s["id"] == "ts38211")
    assert ts["level"] == "L1"
    assert ts["release"] == "Rel-18"
    assert ts["version"].startswith("v18.")
    assert ts["url"].startswith("https://")