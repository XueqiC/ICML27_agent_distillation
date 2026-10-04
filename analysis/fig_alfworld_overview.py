"""One-page overview figure of ALFWorld and this project's harness (2026-10-04).

The example episode is the real smoke run reports/runs/smoke-teacher-1 (gpt-6-luna, train task); counts come from
data/manifest/alfworld_split_freeze.json. Run: python3 analysis/fig_alfworld_overview.py -> reports/figs/alfworld_overview.png
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

for path in ("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",):
    if Path(path).exists():
        font_manager.fontManager.addfont(path)
plt.rcParams["font.family"] = ["Noto Sans CJK JP", "DejaVu Sans"]

INK, INK2, MUTED = "#0b0b0b", "#52514e", "#8a8984"
BLUE, ORANGE, AQUA, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#b8b7b1"
SURFACE = "#fcfcfb"


def box(ax, x, y, w, h, edge, title, lines, *, fill=None, title_size=12.5, size=9.6, mono=False, dashed=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.006,rounding_size=0.012",
                                linewidth=1.6, edgecolor=edge, facecolor=fill or SURFACE,
                                linestyle=(0, (4, 3)) if dashed else "solid"))
    ax.text(x + 0.012, y + h - 0.016, title, fontsize=title_size, color=INK, weight="bold", va="top")
    yy = y + h - 0.056
    for line in lines:
        ax.text(x + 0.014, yy, line, fontsize=size, color=INK2, va="top",
                family="monospace" if mono and not any("一" <= ch <= "鿿" for ch in line) else None)
        yy -= size * 0.00205


def arrow(ax, a, b, color=INK2, rad=0.0):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=14, linewidth=1.6, color=color,
                                 connectionstyle=f"arc3,rad={rad}"))


fig = plt.figure(figsize=(17, 10.5), dpi=150)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
fig.patch.set_facecolor(SURFACE)

ax.text(0.02, 0.975, "ALFWorld 是什么:文本版的家务任务环境 + 我们论文一的 harness", fontsize=18, weight="bold", color=INK, va="top")
ax.text(0.02, 0.938, "ALFWorld 把 ALFRED 的家务任务做成纯文本游戏(TextWorld 引擎)。智能体每一步读一段文字观察,"
        "从\"可选命令\"里挑一条执行,直到完成目标或步数用完。", fontsize=10.5, color=INK2, va="top")

# ① environment
box(ax, 0.02, 0.43, 0.30, 0.47, BLUE, "① 环境:一局游戏 = 一个房间 + 物体 + 一个目标", [
    "6 类任务模板:",
    "  • pick_and_place_simple   把 X 放到 Y",
    "  • look_at_obj_in_light     在台灯下看 X",
    "  • pick_clean_then_place    洗干净再放",
    "  • pick_heat_then_place     加热再放",
    "  • pick_cool_then_place     冷却再放",
    "  • pick_two_obj_and_place   放两个 X",
    "",
    "数据划分(本项目冻结的 manifest):",
    "  • train   3,553 局可玩 → 种子与扩展只能从这里来",
    "  • dev     valid_seen 140 局 → 调参 / 选 checkpoint",
    "  • test    valid_unseen 134 局 → 封存,只做最终评估",
    "",
    "奖励:只有终局 0/1(完成目标才算成功);",
    "本 harness 每局最多 40 步。",
])
box(ax, 0.035, 0.445, 0.27, 0.085, GRAY, "隐藏信息(只进审计字段,绝不进提示)", [
    "PDDL 世界状态事实、expert plan、policy_commands、",
    "won 标记 —— 用来判定成功与做机制分析,模型看不到。",
], fill="#f1f0ec", title_size=10.5, size=9.2, dashed=True)

# ② actions = tools
box(ax, 0.345, 0.43, 0.30, 0.47, ORANGE, "② \"工具\" = 文本命令(每步给出可选列表)", [
    "命令模板(必须从当前 admissible commands 里选一条):",
    "  go to {容器}            take {物体} from {容器}",
    "  move {物体} to {容器}    open / close {容器}",
    "  heat {物体} with {微波炉}  cool {物体} with {冰箱}",
    "  clean {物体} with {水槽}   use {台灯}",
    "  examine {x}   look   inventory   help",
    "",
    "环境返回的是一句文字反馈,例如:",
    "  \"You arrive at desk 1. On the desk 1, you see a",
    "   book 2, a bowl 1, a cd 3, a pencil 2 ...\"",
    "  \"You pick up the alarmclock 1 from the sidetable 1.\"",
    "  不合法或无效的动作 → \"Nothing happens.\"",
    "",
    "关键难点:物体藏在柜子 / 抽屉里要自己找;",
    "动作有前提(先 open 才能看里面、先拿起才能放);",
    "失败后要识别并换路线 —— 正是论文一要研究的能力。",
])

# ③ harness loop
box(ax, 0.67, 0.43, 0.31, 0.47, AQUA, "③ 我们的 harness:一个闭环回合", [], title_size=12.5)
steps = [
    (0.80, 0.815, "环境 reset / step\n(textworld + AlfredDemangler,不装 expert)"),
    (0.80, 0.725, "可见输入 = 目标 + 历史 + 当前观察\n+ admissible commands(白名单字段)"),
    (0.80, 0.635, "智能体:教师 gpt-6-luna(黑盒 API)\n或 学生 gemma-4-12B + LoRA(vLLM)"),
    (0.80, 0.545, "输出(ReAct,第二阶段实现):\nThought: …   Action: <一条命令>"),
]
for x, y, text in steps:
    ax.add_patch(FancyBboxPatch((x - 0.115, y - 0.035), 0.23, 0.07, boxstyle="round,pad=0.004,rounding_size=0.01",
                                linewidth=1.2, edgecolor=AQUA, facecolor="#ffffff"))
    ax.text(x, y, text, ha="center", va="center", fontsize=9.3, color=INK)
for (x1, y1, _), (x2, y2, _) in zip(steps, steps[1:]):
    arrow(ax, (x1, y1 - 0.036), (x2, y2 + 0.036))
arrow(ax, (0.925, 0.545), (0.925, 0.815), color=AQUA, rad=0.0)
ax.text(0.932, 0.68, "解析命令\n→ step\n成功或\n40 步停", fontsize=8.6, color=INK2, va="center")
ax.text(0.685, 0.442, "每一步写一行事件日志(JSONL + 内容寻址 blob);\n"
        "分支 = reset + 动作前缀重放,逐步校验状态一致", fontsize=8.8, color=INK2, va="bottom")

# example episode (real smoke run)
box(ax, 0.02, 0.05, 0.96, 0.35, INK2, "真实例子:gpt-6-luna 跑的一个 train 任务(smoke-teacher-1,4 步成功,重放一致)", [], title_size=12)
rows = [
    ("初始观察", "You are in the middle of a room. Looking quickly around you, you see a bed 1, a desk 1, a drawer 3, a drawer 2,\n"
                 "a drawer 1, a garbagecan 1, a laundryhamper 1, a shelf 2, a shelf 1, and a sidetable 1.  Your task is to: put a alarmclock in desk.",
     "可选命令 13 条:go to bed 1 … go to sidetable 1, help, inventory, look"),
    ("第 1 步  go to sidetable 1", "You arrive at sidetable 1. On the sidetable 1, you see a alarmclock 3, a alarmclock 2, a alarmclock 1, a creditcard 1, ...",
     "可选命令变成 22 条(多了 take … from sidetable 1、examine …)"),
    ("第 2 步  take alarmclock 1 from sidetable 1", "You pick up the alarmclock 1 from the sidetable 1.", "16 条"),
    ("第 3 步  go to desk 1", "You arrive at desk 1. On the desk 1, you see a book 2, a bowl 1, a cd 3, a pencil 2, a pencil 1, and a remotecontrol 1.", "15 条"),
    ("第 4 步  move alarmclock 1 to desk 1", "You move the alarmclock 1 to the desk 1.", "→ 目标完成:won = True(审计字段),回合结束,成功"),
]
y = 0.335
for label, obs, note in rows:
    ax.text(0.035, y, label, fontsize=9.8, color=ORANGE if label.startswith("第") else BLUE, weight="bold", va="top")
    ax.text(0.26, y, obs, fontsize=8.9, color=INK, va="top", family="monospace")
    ax.text(0.26, y - (0.036 if "\n" in obs else 0.021), note, fontsize=8.6, color=MUTED, va="top")
    y -= 0.062 if "\n" in obs else 0.052

out = Path(__file__).resolve().parents[1] / "reports" / "figs" / "alfworld_overview.png"
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, facecolor=SURFACE)
print(out)
