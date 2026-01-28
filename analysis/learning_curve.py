'''
This code will produce the learning curve for different agents
that are specified in the json files
Status : Complete (not completed the key based best parameter selection part)
Updates : This files combines all json of the same agent as well
'''
import os, sys, time, copy
from datetime import datetime
sys.path.append(os.getcwd())
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter

from src.utils.json_handling import get_sorted_dict
from analysis.utils import find_best, smoothen_runs
from src.utils.formatting import create_folder
from analysis.colors import agent_colors, line_node

# read the arguments etc
if len(sys.argv) < 4:
    print("usage : python analysis/learning_curve.py legend(y/n) <metric> <list of json files>")
    exit()

SMALL_SIZE = 8
MEDIUM_SIZE = 10
BIGGER_SIZE = 12
BIGGEST_SIZE = 14

rcParams = {
    'font.size': BIGGER_SIZE,
    'axes.titlesize': BIGGER_SIZE,
    'axes.labelsize': BIGGER_SIZE,
    'xtick.labelsize': BIGGER_SIZE,
    'ytick.labelsize': BIGGER_SIZE,
    'legend.fontsize': SMALL_SIZE,
    'figure.titlesize': BIGGER_SIZE
}
plt.rcParams.update(rcParams)

assert sys.argv[1].lower() in ['y', 'n'], "[ERROR], Choose between y/n"

show_legend = sys.argv[1].lower() == 'y'
metric = sys.argv[2].lower()
json_files = sys.argv[3:]  # all the json files



json_handles = [get_sorted_dict(j) for j in json_files]

def confidence_interval(mean, stderr):
    return (mean - stderr, mean + stderr)

def format_episode_ticks(x, pos):
    """Format x-axis ticks to show episodes in thousands."""
    return f'{int(x/1000)}k' if x > 0 else '0'

def plot_curve(ax, data, label=None, color=None):
    """Plot learning curve with transparent confidence interval and solid mean line."""
    mean = data['mean'].reshape(-1)
    stderr = data['stderr'].reshape(-1)
    
    # Smooth the mean line very strongly for a clean center line
    mean_smooth = smoothen_runs(mean, factor=0.99995)
    
    # Calculate confidence interval with strong smoothing
    (low_ci, high_ci) = confidence_interval(mean, stderr)
    low_ci_smooth = smoothen_runs(low_ci, factor=0.995)
    high_ci_smooth = smoothen_runs(high_ci, factor=0.995)
    
    if color is not None:
        # Plot confidence interval with low opacity for better distinction
        ax.fill_between(range(mean.shape[0]), low_ci_smooth, high_ci_smooth, 
                         color=color, alpha=0.15, zorder=2, edgecolor='none')
        # Plot solid, very highly smoothed mean line
        base, = ax.plot(mean_smooth, label=label, linewidth=3.0, color=color, zorder=3, alpha=1.0)
    else:
        ax.fill_between(range(mean.shape[0]), low_ci_smooth, high_ci_smooth, 
                         alpha=0.15, zorder=2, edgecolor='none')
        base, = ax.plot(mean_smooth, label=label, linewidth=3.0, zorder=3, alpha=1.0)
    
    return mean_smooth, base.get_color()

def annotate_curve(ax, y, label, color, position_idx):
    """Annotate curve with algorithm name using arrow pointing to curve."""
    if len(y) == 0:
        return
    
    # Strategically distribute annotations to avoid overlap and crossing arrows
    # Each config: (x_position_fraction, (horizontal_offset, vertical_offset))
    annotation_configs = [
        (0.60, (75, 55)),      # DQN - right, high up
        (0.18, (-70, 75)),     # DQN Optimistic - left, top
        (0.22, (-70, 25)),     # Double DQN - left, middle height
        (0.50, (75, -15)),     # DQN LR Decay - center, slightly down
        (0.85, (35, -75)),     # DQN N-Step - very far right, bottom
    ]
    
    config = annotation_configs[position_idx % len(annotation_configs)]
    x_frac, xy_text = config
    
    x_pos = int(len(y) * x_frac)
    y_pos = y[min(x_pos, len(y)-1)]
    
    # Determine arrow curvature based on direction to avoid crossing
    h_offset, v_offset = xy_text
    if h_offset < 0:  # Arrow going left
        rad = -0.3
    elif v_offset > 40:  # Arrow going up
        rad = 0.2
    elif v_offset < -40:  # Arrow going down
        rad = -0.2
    else:
        rad = 0.15
    
    ax.annotate(label, xy=(x_pos, y_pos), xytext=xy_text,
                textcoords='offset points', fontsize=8.5, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.95, 
                          edgecolor=color, linewidth=1.8),
                color=color, zorder=5,
                arrowprops=dict(arrowstyle='->', color=color, lw=1.8, 
                                connectionstyle=f'arc3,rad={rad}'))

fig, (ax_return, ax_length) = plt.subplots(1, 2, figsize=(15, 5.5), dpi=150)

# Collect algorithm names for filename
algorithm_names = []

for en, js in enumerate(json_handles):
    run, param, data = find_best(js, data='returns', metric=metric, minmax='max')
    if param is None:
        print(f"Warning: No valid data found for json {en}")
        continue
    print(param)
    
    agent = param['algo']
    algorithm_names.append(agent)
    color = agent_colors.get(agent, 'black')
    
    # Plot returns on left axis
    x_return, _ = plot_curve(ax_return, data['returns'], label=agent, color=color)
    
    # Plot episode length on right axis
    x_length, _ = plot_curve(ax_length, data['num_steps'], label=agent, color=color)

# Add annotations after all curves are plotted (so we have proper axis limits)
for en, js in enumerate(json_handles):
    run, param, data = find_best(js, data='returns', metric=metric, minmax='max')
    if param is None:
        continue
    
    agent = param['algo']
    color = agent_colors.get(agent, 'black')
    x_return, _ = plot_curve(ax_return, data['returns'], label=agent, color=color)
    x_length, _ = plot_curve(ax_length, data['num_steps'], label=agent, color=color)
    
    annotate_curve(ax_return, x_return, agent.upper().replace('_', ' '), color, en)
    annotate_curve(ax_length, x_length, agent.upper().replace('_', ' '), color, en)

# Configure left plot (Average Return)
ax_return.set_xlabel('Number of Episodes', fontsize=12, fontweight='bold')
ax_return.set_ylabel('Average Return', fontsize=12, fontweight='bold')
ax_return.set_title('Average Return', fontsize=13, fontweight='bold', pad=12)
ax_return.xaxis.set_major_formatter(FuncFormatter(format_episode_ticks))
ax_return.spines['top'].set_visible(False)
ax_return.spines['right'].set_visible(False)
ax_return.spines['left'].set_linewidth(1.2)
ax_return.spines['bottom'].set_linewidth(1.2)
ax_return.grid(True, alpha=0.25, linestyle='--', linewidth=0.7)
ax_return.set_axisbelow(True)

# Configure right plot (Average Episode Length)
ax_length.set_xlabel('Number of Episodes', fontsize=12, fontweight='bold')
ax_length.set_ylabel('Average Episode Length', fontsize=12, fontweight='bold')
ax_length.set_title('Average Episode Length', fontsize=13, fontweight='bold', pad=12)
ax_length.xaxis.set_major_formatter(FuncFormatter(format_episode_ticks))
ax_length.spines['top'].set_visible(False)
ax_length.spines['right'].set_visible(False)
ax_length.spines['left'].set_linewidth(1.2)
ax_length.spines['bottom'].set_linewidth(1.2)
ax_length.grid(True, alpha=0.25, linestyle='--', linewidth=0.7)
ax_length.set_axisbelow(True)

foldername = './plots'
create_folder(foldername)

# Create filename with algorithm names and timestamp
algorithms_str = '_'.join(algorithm_names)
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
filename = f'learning_curve_{algorithms_str}_{timestamp}.png'

fig.tight_layout()
plt.savefig(f'{foldername}/{filename}', dpi=150, bbox_inches='tight')
print(f"\nPlot saved to: {foldername}/{filename}")

