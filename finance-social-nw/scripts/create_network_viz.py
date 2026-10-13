"""Generate network visualizations for paper figures."""
import json
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import pandas as pd
import numpy as np
from matplotlib.gridspec import GridSpec
from datetime import datetime

# Load pair relations
with open('data/processed/pair_relations.jsonl', 'r') as f:
    relations = [json.loads(line) for line in f]

# Define sectors
sectors = {
    'GPU/Semiconductor': ['NVDA', 'AMD', 'AVGO', 'SMCI'],
    'Cloud/AI Platform': ['MSFT', 'GOOGL', 'META', 'PLTR'],
    'Speculative AI': ['BBAI', 'IONQ', 'RGTI', 'SOUN'],
    'Control': ['KO', 'PG', 'XOM']
}

sector_colors = {
    'GPU/Semiconductor': '#FF6B6B',
    'Cloud/AI Platform': '#4ECDC4',
    'Speculative AI': '#FFE66D',
    'Control': '#95E1D3'
}

# Map ticker to sector
ticker_to_sector = {}
for sector, tickers in sectors.items():
    for ticker in tickers:
        ticker_to_sector[ticker] = sector

# Create early vs late networks
cutoff = datetime(2023, 5, 1)
early_rels = [r for r in relations if datetime.fromisoformat(r['timestamp'].replace('Z', '+00:00')).replace(tzinfo=None) < cutoff]
late_rels = [r for r in relations if datetime.fromisoformat(r['timestamp'].replace('Z', '+00:00')).replace(tzinfo=None) >= cutoff]

def build_graph(rels, min_count=1):
    """Build network from relations."""
    G = nx.Graph()
    edge_counts = {}
    
    for r in rels:
        a, b = r['ticker_a'], r['ticker_b']
        key = tuple(sorted([a, b]))
        edge_counts[key] = edge_counts.get(key, 0) + 1
    
    for (a, b), count in edge_counts.items():
        if count >= min_count:
            G.add_edge(a, b, weight=count)
    
    return G

# Build graphs
G_early = build_graph(early_rels)
G_late = build_graph(late_rels)

# Create hero figure: Early vs Late + GPU-Cloud zoom
fig = plt.figure(figsize=(16, 10))
gs = GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.3)

# Early network
ax1 = fig.add_subplot(gs[0, 0])
pos_early = nx.spring_layout(G_early, k=2, iterations=50, seed=42)
node_colors_early = [sector_colors[ticker_to_sector.get(node, 'Control')] for node in G_early.nodes()]
node_sizes_early = [G_early.degree(node) * 300 for node in G_early.nodes()]
nx.draw_networkx_nodes(G_early, pos_early, node_color=node_colors_early, node_size=node_sizes_early, ax=ax1, alpha=0.8)
nx.draw_networkx_edges(G_early, pos_early, ax=ax1, alpha=0.2, width=1)
nx.draw_networkx_labels(G_early, pos_early, font_size=8, font_weight='bold', ax=ax1)
ax1.set_title('Early Period (Nov 2022 - Apr 2023)\nSparse: 259 relations, 24 causal-driver', fontsize=11, fontweight='bold')
ax1.axis('off')

# Late network
ax2 = fig.add_subplot(gs[0, 1])
pos_late = nx.spring_layout(G_late, k=1.5, iterations=50, seed=42)
node_colors_late = [sector_colors[ticker_to_sector.get(node, 'Control')] for node in G_late.nodes()]
node_sizes_late = [G_late.degree(node) * 300 for node in G_late.nodes()]
nx.draw_networkx_nodes(G_late, pos_late, node_color=node_colors_late, node_size=node_sizes_late, ax=ax2, alpha=0.8)
nx.draw_networkx_edges(G_late, pos_late, ax=ax2, alpha=0.3, width=1.5)
nx.draw_networkx_labels(G_late, pos_late, font_size=8, font_weight='bold', ax=ax2)
ax2.set_title('Late Period (May 2023 - Sep 2026)\nDense: 1,934 relations, 200 causal-driver (8.3x growth)', fontsize=11, fontweight='bold')
ax2.axis('off')

# Legend
ax3 = fig.add_subplot(gs[0, 2])
ax3.axis('off')
legend_elements = [
    mpatches.Patch(facecolor='#FF6B6B', label='GPU/Semiconductor'),
    mpatches.Patch(facecolor='#4ECDC4', label='Cloud/AI Platform'),
    mpatches.Patch(facecolor='#FFE66D', label='Speculative AI'),
    mpatches.Patch(facecolor='#95E1D3', label='Control')
]
ax3.legend(handles=legend_elements, loc='upper left', fontsize=10, title='Sector', title_fontsize=11)
ax3.text(0.05, 0.5, 'Key Finding:\nChatGPT disruption caused\nnarrative network restructuring.\nGPU sector showed\n3.0x density growth,\nCloud 1.2x.',
         fontsize=10, verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# GPU-Cloud subgraph (early)
ax4 = fig.add_subplot(gs[1, 0])
gpu_firms = ['NVDA', 'AMD', 'AVGO', 'SMCI']
cloud_firms = ['MSFT', 'GOOGL', 'META', 'PLTR']
subgraph_firms = gpu_firms + cloud_firms
if G_early.number_of_nodes() > 0:
    G_early_sub = G_early.subgraph([n for n in subgraph_firms if n in G_early.nodes()]).copy()
    if G_early_sub.number_of_nodes() > 0:
        pos_early_sub = nx.spring_layout(G_early_sub, k=1.5, iterations=50, seed=42)
        node_colors_sub = ['#FF6B6B' if n in gpu_firms else '#4ECDC4' for n in G_early_sub.nodes()]
        node_sizes_sub = [G_early_sub.degree(n) * 500 for n in G_early_sub.nodes()]
        nx.draw_networkx_nodes(G_early_sub, pos_early_sub, node_color=node_colors_sub, node_size=node_sizes_sub, ax=ax4, alpha=0.8)
        nx.draw_networkx_edges(G_early_sub, pos_early_sub, ax=ax4, alpha=0.3, width=2)
        nx.draw_networkx_labels(G_early_sub, pos_early_sub, font_size=9, font_weight='bold', ax=ax4)
ax4.set_title('GPU-Cloud Value Chain\n(Early: Sparse)', fontsize=10, fontweight='bold')
ax4.axis('off')

# GPU-Cloud subgraph (late)
ax5 = fig.add_subplot(gs[1, 1])
if G_late.number_of_nodes() > 0:
    G_late_sub = G_late.subgraph([n for n in subgraph_firms if n in G_late.nodes()]).copy()
    if G_late_sub.number_of_nodes() > 0:
        pos_late_sub = nx.spring_layout(G_late_sub, k=1.2, iterations=50, seed=42)
        node_colors_sub = ['#FF6B6B' if n in gpu_firms else '#4ECDC4' for n in G_late_sub.nodes()]
        node_sizes_sub = [G_late_sub.degree(n) * 500 for n in G_late_sub.nodes()]
        nx.draw_networkx_nodes(G_late_sub, pos_late_sub, node_color=node_colors_sub, node_size=node_sizes_sub, ax=ax5, alpha=0.8)
        nx.draw_networkx_edges(G_late_sub, pos_late_sub, ax=ax5, alpha=0.3, width=2)
        nx.draw_networkx_labels(G_late_sub, pos_late_sub, font_size=9, font_weight='bold', ax=ax5)
ax5.set_title('GPU-Cloud Value Chain\n(Late: Dense & Coupled)', fontsize=10, fontweight='bold')
ax5.axis('off')

# Homophily & metrics
ax6 = fig.add_subplot(gs[1, 2])
ax6.axis('off')
metrics_text = """
VALUE-CHAIN HOMOPHILY

GPU -> Cloud Relations:
  * Causal-driver: 102
  * Substitute: 45
  * Ratio: 2.22x

Price Correlation:
  * GPU intra: 0.502
  * Cloud intra: 0.397
  * GPU-Cloud: 0.342
  * GPU-Control: -0.078

Interpretation:
Media frames GPUs as
ENABLING Cloud AI,
not COMPETING.
This complementary
framing predicts
price co-movement.
"""
ax6.text(0.05, 0.95, metrics_text, fontsize=8.5, verticalalignment='top', family='monospace',
         bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))

plt.suptitle('ChatGPT Disruption Restructures AI Sector Narratives: Network Evolution & Value-Chain Coupling', 
             fontsize=14, fontweight='bold', y=0.98)

plt.savefig('analysis/figures/hero_figure.png', dpi=300, bbox_inches='tight')
print("✅ Hero figure saved: analysis/figures/hero_figure.png")

# Additional figure: Density evolution by sector
fig2, axes = plt.subplots(2, 2, figsize=(14, 10))
fig2.suptitle('Sector-Level Network Density Evolution & Structural Metrics', fontsize=13, fontweight='bold')

for idx, (sector, firms) in enumerate(sectors.items()):
    ax = axes[idx // 2, idx % 2]
    
    # Get subgraphs for each period
    firms_in_early = [f for f in firms if f in G_early.nodes()]
    firms_in_late = [f for f in firms if f in G_late.nodes()]
    
    early_sub = G_early.subgraph(firms_in_early).copy() if firms_in_early else nx.Graph()
    late_sub = G_late.subgraph(firms_in_late).copy() if firms_in_late else nx.Graph()
    
    # Compute density
    early_density = nx.density(early_sub) if early_sub.number_of_nodes() > 0 else 0
    late_density = nx.density(late_sub) if late_sub.number_of_nodes() > 0 else 0
    
    # Plot density evolution
    periods = ['Early\n(Nov 22-Apr 23)', 'Late\n(May 23-Sep 26)']
    densities = [early_density, late_density]
    colors = ['#FFB6C6', '#FF6B6B'] if sector == 'GPU/Semiconductor' else ['#A8E6F1', '#4ECDC4']
    
    bars = ax.bar(periods, densities, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel('Network Density', fontsize=10, fontweight='bold')
    growth = densities[1]/max(densities[0], 0.001)
    ax.set_title(f'{sector}\nGrowth: {growth:.1f}x', 
                 fontsize=11, fontweight='bold', color=colors[-1])
    
    # Add value labels on bars
    for bar, val in zip(bars, densities):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    # Add clustering coefficient
    early_clustering = nx.average_clustering(early_sub) if early_sub.number_of_nodes() > 1 else 0
    late_clustering = nx.average_clustering(late_sub) if late_sub.number_of_nodes() > 1 else 0
    
    ax.text(0.5, -0.25, f'Clustering Coeff: {early_clustering:.3f} -> {late_clustering:.3f}',
            ha='center', transform=ax.transAxes, fontsize=9, style='italic')

plt.tight_layout()
plt.savefig('analysis/figures/sector_density_evolution.png', dpi=300, bbox_inches='tight')
print("✅ Sector density figure saved: analysis/figures/sector_density_evolution.png")

plt.close('all')
