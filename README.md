# ICU-Sepsis DQN Variants - Course Project

## Project Overview

This repository contains work for a university course on **Deep Reinforcement Learning** at TUM. The project involves taking an existing research publication that applies deep reinforcement learning and implementing algorithmic modifications to improve performance.

### Base Research

We build upon the [ICU-Sepsis: A Benchmark MDP Built from Real Medical Data](https://arxiv.org/abs/2406.05646) paper, which found that **DQN achieved the best results** among tested algorithms on the ICU-Sepsis environment.

### Our Modifications

We implemented **four DQN variants** to investigate different improvement strategies:

1. **Optimistic DQN** - Positive Q-initialization for enhanced exploration through Xavier weight initialization and positive bias
2. **Double DQN** - Reduced Q-value overestimation by decoupling action selection from evaluation
3. **DQN + Learning Rate Decay** - Stabilized learning over time through linear LR decay
4. **N-Step DQN** - Faster reward propagation using 3-step returns for better credit assignment

All variants are compared against the **Standard DQN baseline** using 39 random seeds per algorithm (195 total experiments) to ensure statistical robustness.

---

# The ICU-Sepsis Environment (Baseline Algorithms Implementation)

The **ICU-Sepsis** environment is a reinforcement learning environment that
simulates the treatment of sepsis in an intensive care unit (ICU). The
environment is introduced in the paper
[ICU-Sepsis: A Benchmark MDP Built from Real Medical Data](https://arxiv.org/abs/2406.05646),
accepted at the Reinforcement Learning Conference, 2024. ICU-Sepsis is built
using the [MIMIC-III Dataset](https://physionet.org/content/mimiciii/1.4/),
based on the work of
[Komorowski et al. (2018)](https://www.nature.com/articles/s41591-018-0213-5). The environment can be found at the following [Repository](https://github.com/icu-sepsis/icu-sepsis/tree/main).


Citation:
```bibtex
@inproceedings{
  choudhary2024icusepsis,
  title={{ICU-Sepsis}: A Benchmark {MDP} Built from Real Medical Data},
  author={Kartik Choudhary and Dhawal Gupta and Philip S. Thomas},
  booktitle={Reinforcement Learning Conference},
  year={2024},
  url={https://arxiv.org/abs/2406.05646}
}
```

## Prerequisites

### Python Environment Setup

This project requires **Python 3.11** (Python 3.12 is not compatible due to package dependencies).

Create and activate a conda environment:

```bash
conda create -n deep_reinforcement_learning python=3.11
conda activate deep_reinforcement_learning
```

### Install Dependencies

Install required Python packages:

```bash
pip install -r requirements.txt
```

**Note**: The requirements include:
- `numpy==1.25.2` (requires Python ≤3.11)
- `torch==2.0.1`
- `gymnasium==0.28.1`
- `icu-sepsis==2.0.1`
- Other RL and logging packages

### Install GNU Parallel

GNU parallel is required to run multiple experiment configurations in parallel.

**On macOS:**

```bash
# Accept Xcode license (if needed)
sudo xcodebuild -license accept

# Install via Homebrew
brew install parallel
```

**On Linux:**

```bash
# Ubuntu/Debian
sudo apt-get install parallel

# CentOS/RHEL
sudo yum install parallel
```

To silence the citation notice, run once:

```bash
parallel --citation
```

## Code Organization

The repository is structured as follows:

- **`experiments/`** - JSON configuration files specifying hyperparameters and experiment settings
- **`src/`** - Source code for DQN variants and baseline algorithms
- **`run/`** - Scripts for executing experiments (local and distributed)
- **`analysis/`** - Data processing, visualization, and convergence analysis tools
- **`results/`** - Raw experimental results (one file per seed/configuration)
- **`processed/`** - Aggregated data across seeds
- **`plots/`** - Generated visualizations and learning curves

## Quick Start

### Running Experiments with JSON Configuration

Experiments are configured using JSON files that specify algorithm hyperparameters. For example, `experiments/debug.json` contains settings like:
- `algo`: Algorithm name (e.g., "dqn", "dqn_optimistic", "ppo", "sac")
- `seed`: List of random seeds for multiple runs (e.g., `[0,1,2,3,4]`)
- `learning_rate`: Learning rate values to test (can be single value or list)
- `buffer_size`: Replay buffer size for off-policy algorithms
- `batch_size`: Mini-batch size for training
- `max_episodes`: Number of training episodes

Parameters specified as lists will generate multiple experiment configurations (Cartesian product). Each combination of parameters with each seed creates a unique experiment.

**Switching between algorithms:** To test a modified algorithm (e.g., DQN with optimistic initialization), simply change the `"algo"` field in `debug.json`:
```json
"algo": "dqn_optimistic"
```

To run a specific experiment configuration:

```bash
python src/mainjson.py experiments/debug.json 0
```
The above command will run the first configuration in the `debug.json` file (configuration index 0).


### Executing Parameter Sweeps

To run multiple configurations in parallel using GNU Parallel:

```bash
python run/local.py -p src/mainjson.py -j experiments/debug.json
``` 

This executes all parameter combinations in parallel. Limit concurrent threads using the `-c` flag based on your system resources.

Results are stored in the `results/` directory, with one file per hyperparameter configuration and seed.

### Analysis Workflow

#### 1. Aggregate Results Across Seeds

Aggregate raw results across different random seeds:

```bash
python analysis/process_data.py experiments/debug.json
```

This produces aggregated statistics in the `processed/` directory.

#### 2. Generate Learning Curves

Visualize learning curves with confidence intervals:

```bash
python analysis/learning_curve.py y returns auc experiments/debug.json
```

This plots returns using area under the curve (AUC) for hyperparameter selection. Plots are saved to `plots/`.

#### 3. Calculate Convergence Metrics
To calculate performance metrics similar to Table 3 in the paper (episodes to convergence, steps to convergence, and average return).

**Development/Debugging:**

```bash
python analysis/convergence_metrics.py experiments/debug.json
```

**Production Analysis:**

```bash
python analysis/convergence_metrics.py experiments/PaperPlots/dqn.json experiments/PaperPlots/ppo.json experiments/PaperPlots/sac.json
```

**Output Metrics:**
- **Episodes (K)** - Episodes to convergence (thousands)
- **Steps (M)** - Total environment steps to convergence (millions)
- **Average Return** - Mean return over final 1000 steps
- **Converged Seeds** - Fraction reaching convergence threshold (0.85)

## Large-Scale DQN Variants Comparison Study

### Study Description

This comprehensive study compares **5 DQN variants** to evaluate the effectiveness of different algorithmic improvements on the ICU-Sepsis environment:

1. **Standard DQN** (Baseline) - Classic Deep Q-Network implementation
2. **DQN with Optimistic Q-Initialization** - Uses Xavier uniform weight initialization and positive bias (bias_const=1.0) to encourage early exploration
3. **Double DQN** - Decouples action selection from action evaluation to reduce overestimation bias
4. **DQN with Learning Rate Decay** - Implements linear learning rate decay from initial LR to near-zero over training
5. **N-Step DQN** - Uses 3-step returns for better credit assignment

All algorithms use **identical hyperparameters** for fair comparison:
- **39 random seeds** per algorithm (195 total runs)
- **500,000 episodes** per run
- **Replay buffer**: 10,000 transitions
- **Batch size**: 64
- **Learning rate**: 0.001 (initial)
- **Exploration**: ε-greedy from 1.0 → 0.001 over 25% of episodes
- **Target network update**: Every 512 steps

**Estimated runtime**: 13 hours on a 16-core system (15 cores utilized, 1 reserved for system)

### Running the Comparison Study

**Option 1: Sequential Execution (Recommended)**

Execute algorithms sequentially using all available cores per algorithm:

```bash
./run/run_comparison_sequential.ps1
```

**Features:**
- Maximum throughput per algorithm (15 cores)
- Estimated runtime: ~13 hours
- Progress updates after each algorithm completes

**Option 2: Parallel Execution**

Run all algorithms simultaneously with distributed cores:

```bash
./run/run_comparison_study.ps1 -CoresPerExperiment 3
```

**Features:**
- 5 parallel processes (3 cores each)
- Estimated runtime: ~13 hours
- Individual monitoring per algorithm
- Higher system resource utilization

### Monitoring Progress

**Total completed runs:**
```bash
ls results/*/*.dw | wc -l
```

Expected: **195 files** (39 seeds × 5 algorithms)

**Progress by algorithm:**
```bash
for dir in dqn dqn_optimistic double_dqn dqn_lrdecay dqn_nstep; do
  echo "$dir: $(ls results/$dir/*.dw 2>/dev/null | wc -l)/39"
done
```

### Analyzing Results

After all experiments complete:

**1. Aggregate results across seeds:**
```bash
python analysis/process_data.py experiments/comparison_dqn.json
python analysis/process_data.py experiments/comparison_dqn_optimistic.json
python analysis/process_data.py experiments/comparison_double_dqn.json
python analysis/process_data.py experiments/comparison_dqn_lrdecay.json
python analysis/process_data.py experiments/comparison_dqn_nstep.json
```

**2. Generate comparative learning curves:**
```bash
python analysis/learning_curve.py y returns auc \
  experiments/comparison_dqn.json \
  experiments/comparison_dqn_optimistic.json \
  experiments/comparison_double_dqn.json \
  experiments/comparison_dqn_lrdecay.json \
  experiments/comparison_dqn_nstep.json
```

**3. Calculate convergence metrics:**
```bash
python analysis/convergence_metrics.py \
  experiments/comparison_dqn.json \
  experiments/comparison_dqn_optimistic.json \
  experiments/comparison_double_dqn.json \
  experiments/comparison_dqn_lrdecay.json \
  experiments/comparison_dqn_nstep.json
```

### Expected Outputs

**Generated Artifacts:**
- Learning curves with confidence intervals (5-algorithm comparison)
- Convergence metrics table (episodes, steps, average returns)
- Statistical analysis across 39 seeds per algorithm

**Output Locations:**
- **`plots/`** - Learning curve visualizations (PNG/PDF)
- **`processed/`** - Aggregated statistics (.pcsd format)