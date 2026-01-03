'''
This script calculates convergence metrics for RL algorithms (Table 3 style metrics):
- Episodes to convergence
- Steps to convergence  
- Average return over last 1000 steps

Usage: python analysis/convergence_metrics.py <list of json files>
Example: python analysis/convergence_metrics.py experiments/PaperPlots/dqn.json experiments/PaperPlots/ppo.json
'''
import os
import sys
import numpy as np

sys.path.append(os.getcwd())

from src.utils.json_handling import get_sorted_dict, get_param_iterable
from src.utils.formatting import create_file_name
from analysis.utils import pkl_loader, smoothen_runs

# Check arguments
if len(sys.argv) < 2:
    print("Usage: python analysis/convergence_metrics.py <list of json files>")
    exit()

json_files = sys.argv[1:]


def calculate_convergence_episode(returns, threshold=0.85, window=100):
    """
    Find the episode where the algorithm converges.
    Convergence is defined as when the smoothed return exceeds threshold
    and stays stable (within 0.02) for at least 'window' episodes.
    
    Args:
        returns: Array of returns per episode
        threshold: Minimum return value to consider converged
        window: Number of episodes to check for stability
    
    Returns:
        Episode number where convergence occurs, or -1 if not converged
    """
    if len(returns) < window:
        return -1
    
    # Smooth the returns
    smoothed = smoothen_runs(returns, factor=0.95)
    
    # Find first point where smoothed return exceeds threshold
    for i in range(len(smoothed) - window):
        if smoothed[i] >= threshold:
            # Check if it stays stable for 'window' episodes
            window_data = smoothed[i:i+window]
            if np.std(window_data) < 0.02 and np.min(window_data) >= threshold * 0.98:
                return i
    
    return -1


def calculate_metrics_for_experiment(json_handle):
    """
    Calculate convergence metrics for all runs of an experiment configuration.
    
    Returns:
        dict with algorithm name and metrics
    """
    # Load all runs for this experiment
    returns_all = []
    num_steps_all = []
    
    iterable = get_param_iterable(json_handle)
    algo_name = None
    
    for i in iterable:
        if algo_name is None:
            algo_name = i.get('algo', 'unknown')
        
        # Remove tracking flags
        i.pop('use_tensorboard', None)
        i.pop('track', None)
        
        folder, file = create_file_name(i)
        filename = folder + file + '.dw'
        
        if not os.path.exists(filename):
            print(f"Warning: File not found: {filename}")
            continue
        
        try:
            data = pkl_loader(filename)
            returns_all.append(data['returns'])
            num_steps_all.append(data['num_steps'])
        except Exception as e:
            print(f"Error loading {filename}: {e}")
            continue
    
    if len(returns_all) == 0:
        print(f"No data found for {algo_name}")
        return None
    
    # Convert to numpy arrays
    returns_all = np.array(returns_all)  # shape: (num_seeds, num_episodes)
    num_steps_all = np.array(num_steps_all)
    
    # Calculate metrics across all seeds
    convergence_episodes = []
    convergence_steps = []
    avg_returns_last_1000 = []
    
    for seed_idx in range(len(returns_all)):
        returns = returns_all[seed_idx]
        steps = num_steps_all[seed_idx]
        
        # Find convergence episode
        conv_episode = calculate_convergence_episode(returns)
        
        if conv_episode > 0:
            convergence_episodes.append(conv_episode)
            # Sum steps up to convergence
            convergence_steps.append(np.sum(steps[:conv_episode]))
        
        # Calculate average return over last 1000 steps
        if len(returns) >= 1000:
            avg_returns_last_1000.append(np.mean(returns[-1000:]))
        else:
            avg_returns_last_1000.append(np.mean(returns))
    
    # Calculate statistics
    results = {
        'algorithm': algo_name.upper(),
        'num_seeds': len(returns_all),
        'convergence_episodes_mean': np.mean(convergence_episodes) if convergence_episodes else -1,
        'convergence_episodes_std': np.std(convergence_episodes) if convergence_episodes else 0,
        'convergence_steps_mean': np.mean(convergence_steps) if convergence_steps else -1,
        'convergence_steps_std': np.std(convergence_steps) if convergence_steps else 0,
        'avg_return_mean': np.mean(avg_returns_last_1000),
        'avg_return_std': np.std(avg_returns_last_1000),
        'converged_seeds': len(convergence_episodes)
    }
    
    return results


# Process all JSON files
all_results = []

for json_file in json_files:
    print(f"\nProcessing {json_file}...")
    json_handle = get_sorted_dict(json_file)
    results = calculate_metrics_for_experiment(json_handle)
    
    if results:
        all_results.append(results)


# Print results in table format
print("\n" + "="*100)
print("CONVERGENCE METRICS (Table 3 style)")
print("="*100)
print(f"{'Algorithm':<15} {'Episodes (K)':<15} {'Steps (M)':<15} {'Average Return':<20} {'Converged Seeds':<20}")
print("-"*100)

for res in all_results:
    algo = res['algorithm']
    
    if res['convergence_episodes_mean'] > 0:
        episodes = f"{res['convergence_episodes_mean']/1000:.1f} ± {res['convergence_episodes_std']/1000:.1f}"
        steps = f"{res['convergence_steps_mean']/1e6:.2f} ± {res['convergence_steps_std']/1e6:.2f}"
    else:
        episodes = "N/A"
        steps = "N/A"
    
    avg_return = f"{res['avg_return_mean']:.2f} ± {res['avg_return_std']:.2f}"
    converged = f"{res['converged_seeds']}/{res['num_seeds']}"
    
    print(f"{algo:<15} {episodes:<15} {steps:<15} {avg_return:<20} {converged:<20}")

print("="*100)

# Print additional details
print("\nNotes:")
print("- Convergence is defined as when smoothed return exceeds 0.85 and stays stable")
print("- Average Return is calculated over the last 1000 time steps")
print("- Values show mean ± std across all seeds")
print("- 'Converged Seeds' shows how many seeds reached convergence\n")
