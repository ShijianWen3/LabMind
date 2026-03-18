"""
plot_csv.py — 从 CSV 文件解析 (x, y) 数据并绘制折线图
用法:
    python plot_csv.py data.csv
    python plot_csv.py data.csv --title "我的折线图" --out chart.png
"""

import csv
import sys
import argparse
import os
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib import font_manager

# ── 自动检测并启用中文字体 ────────────────────────────────────────────────────
_CN_FONTS = [
    "PingFang SC", "Hiragino Sans GB",          # macOS
    "Microsoft YaHei", "SimHei", "SimSun",       # Windows
    "WenQuanYi Micro Hei", "Noto Sans CJK SC",  # Linux
    "Source Han Sans CN",
]
_available = {f.name for f in font_manager.fontManager.ttflist}
for _fn in _CN_FONTS:
    if _fn in _available:
        matplotlib.rcParams["font.family"] = _fn
        break
matplotlib.rcParams["axes.unicode_minus"] = False   # 负号正常显示


# ── 1. 解析命令行参数 ────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="从 CSV 文件绘制折线图")
parser.add_argument("csv_file", help="输入的 CSV 文件路径")
parser.add_argument("--title", default="", help="图表标题（默认用文件名）")
parser.add_argument("--xlabel", default="X", help="X 轴标签")
parser.add_argument("--ylabel", default="Y", help="Y 轴标签")
parser.add_argument("--out", default="", help="输出图片路径（不填则弹窗显示）")
parser.add_argument("--delimiter", default=",", help="CSV 分隔符（默认英文逗号）")
args = parser.parse_args()


# ── 2. 读取并解析 CSV ────────────────────────────────────────────────────────
def load_csv(path: str, delimiter: str) -> tuple[list, list]:
    xs, ys = [], []
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.reader(f, delimiter=delimiter)
        for lineno, row in enumerate(reader, start=1):
            # 跳过空行和注释行
            if not row or row[0].strip().startswith("#"):
                continue
            if len(row) < 2:
                print(f"  警告：第 {lineno} 行列数不足，已跳过：{row}", file=sys.stderr)
                continue
            try:
                xs.append(float(row[0].strip()))
                ys.append(float(row[1].strip()))
            except ValueError:
                # 跳过表头或无法转换的行
                if lineno == 1:
                    print(f"  提示：跳过疑似表头行：{row}", file=sys.stderr)
                else:
                    print(f"  警告：第 {lineno} 行无法解析，已跳过：{row}", file=sys.stderr)
    return xs, ys


if not os.path.isfile(args.csv_file):
    sys.exit(f"错误：找不到文件 '{args.csv_file}'")

x_data, y_data = load_csv(args.csv_file, args.delimiter)

if len(x_data) == 0:
    sys.exit("错误：未能从文件中解析出任何有效数据点")

print(f"成功读取 {len(x_data)} 个数据点")


# ── 3. 动态调整画布尺寸 ──────────────────────────────────────────────────────
n = len(x_data)

# 宽度随点数增长，最小 8 英寸，最大 24 英寸
base_width  = 8
extra_width = min(n * 0.15, 16)          # 每点贡献 0.15 英寸，上限 16
fig_w = base_width + extra_width

# 高度固定比例，适当随宽度缩放
fig_h = max(5, fig_w * 0.45)

# 点数很多时启用紧凑模式（标记变小、线变细）
compact = n > 80
marker_size = 3 if compact else 6
line_width  = 1.2 if compact else 1.8


# ── 4. 绘图 ─────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(fig_w, fig_h))

ax.plot(
    x_data, y_data,
    marker="o",
    markersize=marker_size,
    linewidth=line_width,
    color="#2563EB",          # 蓝色主线
    markerfacecolor="white",
    markeredgecolor="#1D4ED8",
    markeredgewidth=1.2,
    zorder=3,
)

# 背景网格
ax.grid(True, linestyle="--", linewidth=0.5, color="#CBD5E1", alpha=0.8)
ax.set_axisbelow(True)

# 轴标签 & 标题
title = args.title if args.title else os.path.splitext(os.path.basename(args.csv_file))[0]
ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
ax.set_xlabel(args.xlabel, fontsize=11)
ax.set_ylabel(args.ylabel, fontsize=11)

# X 轴刻度：点数多时自动稀疏
if n > 30:
    ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=12))

# 数据标注（点数少时才显示）
if n <= 20:
    for xi, yi in zip(x_data, y_data):
        ax.annotate(
            f"{yi:.2f}",
            xy=(xi, yi),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center", va="bottom",
            fontsize=8, color="#374151",
        )

fig.tight_layout()


# ── 5. 保存或展示 ────────────────────────────────────────────────────────────
if args.out:
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"图表已保存至：{args.out}")
else:
    print("正在显示图表窗口…（关闭窗口后程序退出）")
    plt.show()