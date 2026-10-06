"""Task1端到端：3GPP文档 → 波形参数提取 → params.yaml + 验证报告"""
import argparse
import json
import re
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.ingestion.pdf_parser import parse_and_cache
from src.retrieval.retriever import Retriever
from src.retrieval.citation import format_citation, save_citations
from src.validation.validators import (
    validate_numerology,
    validate_cp_duration,
    compute_normal_cp_us,
)

PROCESSED = ROOT / "data" / "processed" / "ts38211_parsed.txt"
OUT = ROOT / "outputs" / "task1"
SOURCES = yaml.safe_load((ROOT / "config" / "sources.yaml").read_text(encoding="utf-8"))


def resolve_raw_path(cli_raw: str = None) -> Path:
    """按优先级解析输入源：
       1. 命令行 --raw 参数
       2. data/raw/ts38211.pdf（本地真实 PDF，可选）
       3. data/raw/sample_ts38211.txt（仓库样例，兜底）
    """
    if cli_raw:
        p = Path(cli_raw)
        if not p.is_absolute():
            p = ROOT / p
        if not p.exists():
            raise FileNotFoundError(f"指定的输入文件不存在: {p}")
        return p

    pdf_path = ROOT / "data" / "raw" / "ts38211.pdf"
    if pdf_path.exists():
        return pdf_path

    txt_path = ROOT / "data" / "raw" / "sample_ts38211.txt"
    if txt_path.exists():
        print(f"[ingestion] 未找到 ts38211.pdf，回退到样例文本: {txt_path.name}")
        return txt_path

    raise FileNotFoundError(
        "找不到任何输入文件。请指定 --raw 参数，"
        "或把 ts38211.pdf 放到 data/raw/"
    )


def get_source(sid: str) -> dict:
    for s in SOURCES["sources"]:
        if s["id"] == sid:
            return s
    raise KeyError(sid)


def extract_numerology(text: str):
    """从 Table 4.2-1 区域抽取 mu / SCS / CP 类型（适配真实 3GPP PDF 排版）。"""
    retriever = Retriever(text)
    table_lines = retriever.extract_section(
        "4.2 Numerologies", "4.3 Frame structure"
    )

    rows = []
    for line in table_lines:
        m = re.search(
            r"(\d+)\s+(\d+)\s+(Normal(?:\s*,\s*Extended)?)",
            line, flags=re.IGNORECASE,
        )
        if not m:
            continue
        mu = int(m.group(1))
        scs = int(m.group(2))
        cp_raw = m.group(3).lower()

        cp_types = []
        if "normal" in cp_raw:
            cp_types.append("normal")
        if "extended" in cp_raw:
            cp_types.append("extended")
        rows.append({"mu": mu, "scs_khz": scs, "cp_types": cp_types})
    return rows


def enrich(rows):
    for r in rows:
        if "normal" in r["cp_types"]:
            r["cp_duration_us"] = round(compute_normal_cp_us(r["mu"]), 4)
        else:
            r["cp_duration_us"] = None
    return rows


def main():
    parser = argparse.ArgumentParser(description="Task1: 3GPP numerology extraction")
    parser.add_argument("--raw", type=str, default=None,
                        help="输入文件路径（PDF 或 TXT），默认自动查找")
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    raw_path = resolve_raw_path(args.raw)
    print(f"[ingestion] 输入源: {raw_path}")

    # 1. 解析 + 缓存
    text = parse_and_cache(raw_path, PROCESSED)
    print(f"[ingestion] 解析 {raw_path.name} -> {len(text.splitlines())} 行")

    # 2. 抽取
    rows = extract_numerology(text)
    print(f"[extraction] 识别 {len(rows)} 条 numerology")
    rows = enrich(rows)

    # 3. 验证
    v1 = validate_numerology(rows)
    v2 = validate_cp_duration(rows)
    validation = {"numerology": v1, "cp_duration": v2}
    (OUT / "validation_report.json").write_text(
        json.dumps(validation, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # 4. 引用
    src = get_source("ts38211")
    citations = [
        {
            "section": "Sec 4.2 Table 4.2-1",
            "citation": format_citation(src, "Sec 4.2 Table 4.2-1"),
            "fields": ["mu", "scs_khz", "cp_types"],
        },
        {
            "section": "Sec 4.3.1",
            "citation": format_citation(src, "Sec 4.3.1"),
            "fields": ["cp_duration_us"],
        },
    ]
    save_citations(citations, OUT / "citations.json")

    # 5. 生成 params.yaml
    params = {
        "meta": {
            "project": "isac-ofdm-agent",
            "task": "task1_extraction",
            "release": SOURCES["release_lock"],
            "source": f"3GPP {src['number']} {src['version']}",
            "input_file": raw_path.name,
            "validation_passed": v1["passed"] and v2["passed"],
        },
        "numerology": rows,
    }
    (OUT / "params.yaml").write_text(
        yaml.safe_dump(params, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    (ROOT / "config" / "params.yaml").write_text(
        yaml.safe_dump(params, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )

    # 6. 汇总
    ok = v1["passed"] and v2["passed"]
    print(f"[validation] numerology: {'PASS' if v1['passed'] else 'FAIL'}")
    for e in v1["errors"]:
        print("   -", e)
    print(f"[validation] cp_duration: {'PASS' if v2['passed'] else 'FAIL'}")
    for e in v2["errors"]:
        print("   -", e)
    print(f"[output] {OUT / 'params.yaml'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())