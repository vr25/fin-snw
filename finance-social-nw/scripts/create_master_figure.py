"""Create master figure summarizing entire paper story."""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 11))
fig.suptitle('Master Summary: ChatGPT Disruption Restructures AI Narratives & Markets', 
             fontsize=14, fontweight='bold', y=0.98)

# Panel 1: Timeline & Event
ax1.set_xlim(0, 10)
ax1.set_ylim(0, 10)
ax1.axis('off')
ax1.text(5, 9, 'Natural Experiment: ChatGPT Launch', fontsize=12, fontweight='bold', ha='center',
         bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
ax1.text(2, 7.5, 'Nov 1, 2022\nChatGPT Launch', fontsize=10, ha='center',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
ax1.arrow(3.5, 7.5, 1.5, 0, head_width=0.3, head_length=0.3, fc='black', ec='black')
ax1.text(5, 7.5, 'Narrative\nRestructuring', fontsize=10, ha='center',
         bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
ax1.arrow(6.5, 7.5, 1.5, 0, head_width=0.3, head_length=0.3, fc='black', ec='black')
ax1.text(8, 7.5, 'Price\nRepricing', fontsize=10, ha='center',
         bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))

ax1.text(5, 5.5, 'Key Mechanism:', fontsize=11, fontweight='bold', ha='center')
ax1.text(5, 4.8, 'GPUs enable LLMs -> LLMs enable Cloud AI -> Cloud AI enables Enterprise', 
         fontsize=9, ha='center', style='italic',
         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.6))

ax1.text(5, 3, 'Data: 2,193 ticker-pair relations from 1,496 GDELT articles (Nov 2022 - Sep 2026)', 
         fontsize=8, ha='center', style='italic')
ax1.text(5, 2.2, 'Sectors: GPU (NVDA, AMD, AVGO, SMCI) | Cloud (MSFT, GOOGL, META, PLTR) | Speculative (BBAI, IONQ, RGTI, SOUN) | Control (KO, PG, XOM)', 
         fontsize=7, ha='center')

# Panel 2: Finding 1 - GPU Density Growth
ax2.set_xlim(0, 4)
ax2.set_ylim(0, 5)
ax2.axis('off')
ax2.text(2, 4.5, 'Finding 1: GPU Narrative Shock', fontsize=11, fontweight='bold', ha='center',
         bbox=dict(boxstyle='round', facecolor='#FF6B6B', alpha=0.7))

# Bar chart for density
periods = ['Early\n(Nov-Apr)', 'Late\n(May-Sep)']
gpu_density = [0.167, 1.0]
cloud_density = [0.667, 1.0]

x_pos = np.array([0.5, 1.5])
width = 0.3

bars1 = ax2.bar(x_pos - width/2, gpu_density, width, label='GPU', color='#FF6B6B', alpha=0.7)
bars2 = ax2.bar(x_pos + width/2, cloud_density, width, label='Cloud', color='#4ECDC4', alpha=0.7)

ax2.set_ylabel('Network Density', fontsize=9, fontweight='bold')
ax2.set_xticks(x_pos)
ax2.set_xticklabels(periods, fontsize=9)
ax2.set_ylim(0, 1.2)
ax2.legend(fontsize=9)

for bar in bars1:
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height + 0.03,
            f'{height:.2f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
for bar in bars2:
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height + 0.03,
            f'{height:.2f}', ha='center', va='bottom', fontsize=8, fontweight='bold')

ax2.text(2, 0.2, 'GPU: 6.0x growth | Cloud: 1.5x growth', fontsize=9, ha='center', fontweight='bold',
         bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))

# Panel 3: Finding 2 - Homophily & Value Chain
ax3.set_xlim(0, 4)
ax3.set_ylim(0, 5)
ax3.axis('off')
ax3.text(2, 4.5, 'Finding 2: GPU-Cloud Homophily', fontsize=11, fontweight='bold', ha='center',
         bbox=dict(boxstyle='round', facecolor='#4ECDC4', alpha=0.7))

# Pie chart for relation types
sizes = [102, 45]
labels = ['Causal-driver\n(102)', 'Substitute\n(45)']
colors = ['#2ecc71', '#e74c3c']
explode = (0.05, 0)

wedges, texts, autotexts = ax3.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
                                     explode=explode, startangle=90, textprops={'fontsize': 9})
for autotext in autotexts:
    autotext.set_color('white')
    autotext.set_fontweight('bold')

ax3.text(2, -1.2, 'Homophily Index: 2.22', fontsize=10, ha='center', fontweight='bold',
         bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
ax3.text(2, -1.6, '(Complementary framing)', fontsize=8, ha='center', style='italic')

# Panel 4: Finding 3 - Price Correlations
ax4.set_xlim(0, 5)
ax4.set_ylim(0, 5)
ax4.axis('off')
ax4.text(2.5, 4.5, 'Finding 3: Price Correlations', fontsize=11, fontweight='bold', ha='center',
         bbox=dict(boxstyle='round', facecolor='#FFE66D', alpha=0.7))

correlations = {
    'GPU intra': 0.502,
    'Cloud intra': 0.397,
    'Speculative': 0.379,
    'Control': 0.269,
    'GPU-Cloud': 0.342,
    'GPU-Spec': 0.078,
}

y_labels = list(correlations.keys())
corr_values = list(correlations.values())

y_pos = np.arange(len(y_labels))
colors_bar = ['#FF6B6B' if 'GPU' in y else '#4ECDC4' if 'Cloud' in y else '#FFE66D' if 'Speculative' in y else '#95E1D3' for y in y_labels]

bars = ax4.barh(y_pos, corr_values, color=colors_bar, alpha=0.7, edgecolor='black')
ax4.set_yticks(y_pos)
ax4.set_yticklabels(y_labels, fontsize=9)
ax4.set_xlabel('Correlation (ρ)', fontsize=9, fontweight='bold')
ax4.set_xlim(0, 0.6)

for i, (bar, val) in enumerate(zip(bars, corr_values)):
    ax4.text(val + 0.02, bar.get_y() + bar.get_height()/2, f'{val:.3f}', 
            va='center', fontsize=8, fontweight='bold')

plt.tight_layout()
plt.savefig('analysis/figures/master_summary.png', dpi=300, bbox_inches='tight')
print("✅ Master summary figure saved: analysis/figures/master_summary.png")
plt.close()
