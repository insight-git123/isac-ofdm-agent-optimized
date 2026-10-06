"""轻量关键词索引（无嵌入依赖，MVP）。"""
import re
from typing import Dict, List


def tokenize(text: str) -> List[str]:
    return re.findall(r"[A-Za-z0-9_]+", text.lower())


def build_index(text: str) -> Dict[str, List[int]]:
    index: Dict[str, List[int]] = {}
    for i, line in enumerate(text.splitlines()):
        for tok in set(tokenize(line)):
            index.setdefault(tok, []).append(i)
    return index