"""基于关键词的检索器，返回行号+内容+得分。"""
from typing import Dict, List

from src.ingestion.indexer import build_index, tokenize


class Retriever:
    def __init__(self, text: str):
        self.lines = text.splitlines()
        self.index = build_index(text)

    def search(self, query: str, top_k: int = 8) -> List[Dict]:
        tokens = tokenize(query)
        scores: Dict[int, int] = {}
        for tok in tokens:
            for ln in self.index.get(tok, []):
                scores[ln] = scores.get(ln, 0) + 1
        ranked = sorted(scores.items(), key=lambda x: (-x[1], x[0]))[:top_k]
        return [
            {"line_no": ln, "score": sc, "text": self.lines[ln].strip()}
            for ln, sc in ranked
        ]

    def extract_section(self, start_kw: str, end_kw: str) -> List[str]:
        """从 start_kw 行到 end_kw 行之间的文本（含）。"""
        out, capturing = [], False
        for line in self.lines:
            if start_kw in line:
                capturing = True
            if capturing:
                out.append(line.rstrip())
            if capturing and end_kw in line and start_kw not in line:
                break
        return out