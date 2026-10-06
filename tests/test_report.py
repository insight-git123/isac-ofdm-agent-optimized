"""Task5 报告生成单元测试。"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.report.report_generator import generate_report

ROOT = Path(__file__).resolve().parents[1]


def test_report_generates():
    """报告应能正常生成且不为空。"""
    md = generate_report(ROOT)
    assert len(md) > 500
    assert "# ISAC-OFDM Agent 实验报告" in md


def test_report_has_dynamic_sections():
    """报告应包含从 JSON 动态生成的章节。"""
    md = generate_report(ROOT)
    # 如果 result.json 存在，应出现对应表头
    task3_json = ROOT / "outputs" / "task3"
    if list(task3_json.glob("result_mu*.json")):
        assert "4.3 检测结果" in md
        assert "4.4 统计汇总" in md
    if (ROOT / "outputs" / "task4" / "isac_sweep_report.json").exists():
        assert "5.1 动态结论" in md


def test_report_no_hardcoded_conclusions():
    """报告不应包含硬编码的结论文字。"""
    md = generate_report(ROOT)
    # 这些是旧版硬编码的，新版应该没有
    forbidden = [
        "检测结果与理论预测完全一致",
        "(150.1m, 30.1m/s) | 真实 |",
    ]
    for s in forbidden:
        assert s not in md, f"发现硬编码文字: {s}"