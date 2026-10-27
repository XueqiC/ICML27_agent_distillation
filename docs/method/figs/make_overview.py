"""Method overview flowchart (replaces the mermaid block: GitHub's mermaid truncated CJK labels)."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

from matplotlib import font_manager
FONT = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
font_manager.fontManager.addfont(FONT)
plt.rcParams['font.family'] = font_manager.FontProperties(fname=FONT).get_name()
nodes = [('A', '当前学生独立执行', 'A'), ('B', 'Harness 记录公开历史与结果', 'A'),
         ('C', 'Teacher 判断当前学习需求', 'B'), ('D', '必要时给出具体指导，学生继续行动', 'B'),
         ('E', '环境产生新的观察与结果', 'B'), ('F', '形成有效监督与条件练习', 'C'),
         ('G', '更新学生', 'C'), ('H', '新任务上无指导评测', 'eval')]
colour = {'A': '#DCEBF7', 'B': '#FCE9D2', 'C': '#DDF2E8', 'eval': '#EEEEEE'}
edge = {'A': '#0072B2', 'B': '#E69F00', 'C': '#009E73', 'eval': '#666666'}
W, H, GAP, X = 6.6, 0.62, 0.42, 0.0
fig, ax = plt.subplots(figsize=(9.6, 9.2))
ys = {}
for i, (k, text, part) in enumerate(nodes):
    y = -i * (H + GAP); ys[k] = y
    ax.add_patch(FancyBboxPatch((X - W / 2, y - H / 2), W, H, boxstyle='round,pad=0.02,rounding_size=0.12',
                                fc=colour[part], ec=edge[part], lw=1.8))
    ax.text(X, y, text, ha='center', va='center', fontsize=15)


def arrow(a, b, **kw):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle='-|>', mutation_scale=18, lw=1.6, color='#333333', **kw))


for (k1, *_), (k2, *_) in zip(nodes, nodes[1:]):
    arrow((X, ys[k1] - H / 2), (X, ys[k2] + H / 2))
# loop E -> C on the right, G -> A on the left
arrow((X + W / 2, ys['E']), (X + W / 2, ys['C']), connectionstyle='arc,angleA=0,angleB=0,armA=34,armB=34,rad=8')
arrow((X - W / 2, ys['G']), (X - W / 2, ys['A']), connectionstyle='arc,angleA=180,angleB=180,armA=34,armB=34,rad=8')
ax.text(X + W / 2 + 1.05, (ys['C'] + ys['E']) / 2, '逐观察\n重新判断', ha='center', va='center', fontsize=12, color='#444444')
ax.text(X - W / 2 - 1.15, (ys['A'] + ys['G']) / 2, '每轮更新后\n重新采集', ha='center', va='center', fontsize=12, color='#444444')
# part labels
for part, ks, lab in [('A', 'AB', 'A. 发现学习需求'), ('B', 'CDE', 'B. 联合指导'), ('C', 'FG', 'C. 学习与迁移')]:
    y = sum(ys[k] for k in ks) / len(ks)
    ax.text(X + W / 2 + 2.0, y, lab, ha='left', va='center', fontsize=13, color=edge[part], fontweight='bold')
ax.set_xlim(-5.6, 8.4); ax.set_ylim(ys['H'] - 0.8, 0.8); ax.axis('off')
fig.tight_layout()
for ext in ('png', 'pdf'):
    fig.savefig(f'docs/method/figs/method_overview.{ext}', bbox_inches='tight', dpi=160)
print('ok')
