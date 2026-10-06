"""引用格式化与溯源记录。"""
import json
from pathlib import Path
from typing import Dict, List


def format_citation(source: Dict, section: str) -> str:
    return (
        f"[{source['level']}] 3GPP {source['number']} "
        f"{source['version']}, {source['release']}, {section}"
    )


def save_citations(citations: List[Dict], path: Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(
        json.dumps(citations, indent=2, ensure_ascii=False), encoding="utf-8"
    )