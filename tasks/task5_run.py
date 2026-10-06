"""Task5 端到端：生成 Markdown 实验报告"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.report.report_generator import generate_report

def main():
    print("[Task5] 开始生成实验报告...")
    
    out_dir = ROOT / "outputs" / "task5"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    report_content = generate_report(ROOT)
    
    report_path = out_dir / "experiment_report.md"
    report_path.write_text(report_content, encoding="utf-8")
    
    print(f"[output] Markdown 报告已保存至: {report_path}")
    print("[Task5] 完成！请在 VS Code 中打开该 Markdown 文件并预览。")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())