# make this into a file which uses argparse to run experiments
import os, sys, time
sys.path.append(os.getcwd())
from src.algos.ppo import run_ppo
from src.algos.dqn import run_dqn
from src.algos.dqn_optimistic import run_dqn_optimistic
from src.algos.sac import run_sac
from src.algos.qlearning import run_qlearning
from src.algos.sarsa import run_sarsa




def get_algo(algo):
    if algo == 'ppo': # done -- converted to episodes
        return run_ppo
    elif algo == 'dqn': # done -- converted to episodes
        return run_dqn
    elif algo == 'dqn_optimistic': # DQN with optimistic Q-initialization
        return run_dqn_optimistic
    elif algo == 'sac': # done -- converted to episodes
        return run_sac
    elif algo == 'qlearning':
        return run_qlearning
    elif algo == 'sarsa':
        return run_sarsa
    else:
        raise NotImplementedError("Unknown algo {}".format(algo))


            