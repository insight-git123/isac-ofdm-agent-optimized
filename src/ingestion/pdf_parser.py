"""PDF/文本解析：优先 pdfplumber，缺失时回退纯文本。"""
from pathlib import Path


def parse_pdf(path: Path) -> str:
    try:
        import pdfplumber
        with pdfplumber.open(path) as pdf:
            return "\n".join(page.extract_text() or "" for page in pdf.pages)
    except ImportError:
        raise RuntimeError(
            "pdfplumber 未安装。请 pip install pdfplumber，或提供 .txt 输入。"
        )


def parse(path: Path) -> str:
    path = Path(path)
    if path.suffix.lower() == ".pdf":
        return parse_pdf(path)
    return path.read_text(encoding="utf-8")


def parse_and_cache(src: Path, dst: Path) -> str:
    text = parse(src)
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    Path(dst).write_text(text, encoding="utf-8")
    return text