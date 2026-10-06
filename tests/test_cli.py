"""统一 CLI 单元测试。"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from isac_ofdm.__main__ import build_parser


def test_cli_parser_builds():
    """argparse 构造应成功。"""
    parser = build_parser()
    assert parser is not None


def test_cli_extract_command():
    """extract 子命令应可解析。"""
    parser = build_parser()
    args = parser.parse_args(["extract"])
    assert args.command == "extract"


def test_cli_simulate_command():
    parser = build_parser()
    args = parser.parse_args(["simulate", "--mu", "2", "--mimo"])
    assert args.command == "simulate"
    assert args.mu == 2
    assert args.mimo is True


def test_cli_sweep_command():
    parser = build_parser()
    args = parser.parse_args(["sweep", "--mc-trials", "30"])
    assert args.command == "sweep"
    assert args.mc_trials == 30


def test_cli_all_command():
    parser = build_parser()
    args = parser.parse_args(["all", "--mc-trials", "10"])
    assert args.command == "all"


def test_cli_requires_subcommand():
    """不给子命令应报错。"""
    parser = build_parser()
    try:
        parser.parse_args([])
        assert False, "应抛出 SystemExit"
    except SystemExit:
        pass