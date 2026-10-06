#!/usr/bin/env bash
set -euo pipefail

echo "========================================="
echo "   ISAC-OFDM Agent 全流程一键复现"
echo "========================================="

if [ -d ".venv/Scripts" ]; then
    source .venv/Scripts/activate
elif [ -d ".venv/bin" ]; then
    source .venv/bin/activate
fi

echo -e "\n[1/12] Task1: 3GPP 参数提取..."
python tasks/task1_run.py --raw data/raw/sample_ts38211.txt

echo -e "\n[2/12] Task2: OFDM 波形生成..."
python tasks/task2_run.py --mu 3

echo -e "\n[3/12] Task2-NRGrid: NR 资源网格..."
python tasks/task2_nr_grid.py --mu 3

echo -e "\n[4/12] Task2-SSB: SS/PBCH block..."
python tasks/task2_ssb.py --n-id 42

echo -e "\n[5/12] Task2-Coding: 卷积编码 BER..."
python tasks/task2_coding.py

echo -e "\n[6/12] Task3: 严格时域脉冲压缩..."
python tasks/task3_time_domain.py --mu 3

echo -e "\n[7/12] Task3: 标准 TDL-A 多径信道..."
python tasks/task3_tdl.py --mu 3 --model TDL-A

echo -e "\n[8/12] Task3: RDM + CFAR + 鬼影抑制..."
python tasks/task3_run.py --mu 3 --swerling --multipath --suppress

echo -e "\n[9/12] Task3-MIMO: 3D Range-Doppler-Angle..."
python tasks/task3_mimo.py --mu 3

echo -e "\n[10/12] Task3-CPOFDM: 全链路验证..."
python tasks/task3_cp_ofdm.py --mu 3

echo -e "\n[11/12] Task4: 蒙特卡洛性能扫描 (20 次加速)..."
python tasks/task4_run.py --mc-trials 20

echo -e "\n[12/12] Task4-Analysis + Task5..."
python tasks/task4_analysis.py
python tasks/task5_run.py

echo -e "\n========================================="
echo "   ✅ 全部完成！"
echo "   报告: outputs/task5/experiment_report.md"
echo "========================================="