import argparse
import os, sys, time
sys.path.append(os.getcwd())
import random
import time
from distutils.util import strtobool

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from src.utils.utils import set_seeds, make_env, get_mask, encode_state, ReplayBuffer
from src.utils.models import QNetwork, linear_schedule




# args = parse_args()
def run_double_dqn(args, use_tensorboard=False, use_wandb=False):
    """
    Runs one training session of a Double DQN agent (with replay buffer + target network).

    Key ideas in this training loop:
    - Epsilon-greedy exploration with a linear decay schedule over episodes
    - Experience replay to break correlation between consecutive transitions
    - Target network to stabilize TD targets
    - Double DQN target computation (online net selects action, target net evaluates it)
    """

    # This implementation currently assumes a single environment instance (no true vectorization).
    assert args.num_envs == 1, "vectorized envs are not supported at the moment"

    # Logging arrays (one entry per episode)
    returns_ = -1 * np.ones([args.max_episodes])
    discounted_returns_ = -1 *  np.ones([args.max_episodes])
    num_steps = np.zeros([args.max_episodes])

    run_name = f"double_dqn_{args.seed}_{int(time.time())}"

    # Optional experiment tracking with Weights & Biases
    if use_wandb:
        import wandb
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )

    # Optional TensorBoard logging
    if use_tensorboard:
        writer = SummaryWriter(log_dir=f"runs/{run_name}")
        writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
        )

    # Reproducibility: seeds RNGs (python, numpy, torch, env)
    set_seeds(args.seed)

    # Device selection: kept CPU-only here (simple + predictable for teaching projects)
    device = torch.device("cpu")


    
    # -------------------------
    # Environment setup
    # -------------------------
    if hasattr(args, 'env_type'):
        env_type = int(args.env_type)
    else:
        env_type = None
    
    # SyncVectorEnv is used with num_envs=1 to keep the API consistent
    envs = gym.vector.SyncVectorEnv(
            [make_env( args.seed + i, env_type) for i in range(args.num_envs)]
        )

    # This agent assumes discrete actions (needed for argmax action selection)
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    # -------------------------
    # Networks + optimizer
    # -------------------------
    # Online Q-network: used for acting and learning
    q_network = QNetwork(envs).to(device)
    optimizer = optim.Adam(q_network.parameters(), lr=args.learning_rate)

    # Target Q-network: used only to compute stable TD targets
    target_network = QNetwork(envs).to(device)
    target_network.load_state_dict(q_network.state_dict()) # start identical

    # Useful for one-hot encoding state in a tabular/discrete-state environment
    n_states = envs.single_observation_space.n
    n_actions = envs.single_action_space.n
    print(f"Number of States: {n_states}, Number of actions: {n_actions}")

    # -------------------------
    # Replay buffer
    # -------------------------
    # Stores transitions (s, a, mask, r, s', mask', done) for off-policy learning.
    rb = ReplayBuffer(
        args.buffer_size
    )

    start_time = time.time()

    # -------------------------
    # Initial reset
    # -------------------------
    states, infos = envs.reset(seed=args.seed)
    obs = encode_state(states, n_states) # model-ready observation encoding
    action_masks = get_mask(infos, args.num_envs, n_actions) # mask invalid actions (env-specific)

    episode_number = 0
    global_step = 0

    # Episode stats (updated when an episode ends)
    total_reward = 0
    episode_length = 0

    # -------------------------
    # Main loop over episodes (driven by termination)
    # -------------------------
    while episode_number < args.max_episodes:
        # Epsilon schedule is defined over episodes (not timesteps) in this implementation
        epsilon = linear_schedule(args.start_e, 
                                  args.end_e, 
                                  args.exploration_fraction * args.max_episodes, 
                                  episode_number
        )

        # -----------------------------------
        # Action selection: epsilon-greedy
        # -----------------------------------
        # Important: some environments expose admissible actions via infos.
        allowed_actions = infos['admissible_actions'][0]
        if random.random() < epsilon:
            # Exploration: sample uniformly from admissible actions
            actions = np.array([random.choice(allowed_actions) for _ in range(envs.num_envs)])
        else:
            # Exploitation: choose argmax_a Q(s,a) subject to mask
            # Mask ensures invalid actions do not get selected.
            q_values = q_network(obs.to(device), action_masks.to(device))
            actions = torch.argmax(q_values, dim=1).cpu().numpy()

        # -----------------------------------
        # Step environment
        # -----------------------------------
        next_states, rewards, terminated, truncated, infos = envs.step(actions)
        global_step += 1
        next_obs = encode_state(next_states, n_states)
        next_action_masks = get_mask(infos, args.num_envs, n_actions)

        # -----------------------------------
        # Episode logging (end-of-episode)
        # -----------------------------------
        # In your environment setup, the reward is only given at the end.
        # Gymnasium provides episode statistics under infos['episode'] when using vector env wrappers.
        if terminated:
            total_reward = float(infos['episode']['r'][0])
            episode_length = int(infos['episode']['l'][0])

             # Discounted return: since reward arrives at end, the discount depends on episode length.
            total_return = total_reward * (args.gamma**(episode_length-1))

            returns_[episode_number] = total_reward
            discounted_returns_[episode_number] = total_return
            num_steps[episode_number] = episode_length
            episode_number += 1

            if use_tensorboard:
                writer.add_scalar("charts/episodic_return", total_reward, episode_number)
                writer.add_scalar("charts/episodic_length", episode_length, episode_number)
                writer.add_scalar("charts/episodic_discounted_return", total_return, episode_number)
                writer.add_scalar("charts/episodic_number", episode_number, episode_number)
                writer.add_scalar("charts/num_steps", global_step, episode_number)
                writer.add_scalar("charts/epsilon", epsilon, episode_number)

        # -----------------------------------
        # Store transition in replay buffer
        # -----------------------------------
        # We store the *encoded* observations and action masks to ensure training uses
        # the same representation as action selection.
        rb.push(obs, actions, action_masks, rewards, next_obs, next_action_masks, terminated)

        # Move to next timestep
        obs = next_obs
        action_masks = next_action_masks

        # -----------------------------------
        # Learning step (after warm-up)
        # -----------------------------------
        if global_step > args.learning_starts:
            # Train periodically to reduce compute and decorrelate updates
            if global_step % args.train_frequency == 0:
                data = rb.sample(args.batch_size)

                # ------------- Double DQN target computation -------------
                # We compute:
                #   y = r + gamma * Q_target(s', argmax_a Q_online(s', a)) * (1 - done)
                #
                # 1) Online net selects best next action (argmax)
                # 2) Target net evaluates that action's Q-value

                with torch.no_grad():
                    # (1) Action selection using online network
                    online_q_values = q_network(data.next_state, data.next_action_mask)
                    best_actions = online_q_values.argmax(dim=1, keepdim=True)

                    # (2) Action evaluation using target network
                    target_q_values = target_network(data.next_state, data.next_action_mask)
                    target_max = target_q_values.gather(1, best_actions).squeeze()

                    # (3) TD target (mask terminal transitions)
                    td_target = data.reward.flatten() + args.gamma * target_max * (1 - data.done.flatten())

                # Predicted Q(s,a) for the actions actually taken
                # note: This call should match your QNetwork signature.
                # If QNetwork requires a mask, pass it here as well (data.action_mask).            
                old_val = q_network(data.state).gather(1, data.action).squeeze()

                # Mean squared TD error (standard DQN loss)
                loss = F.mse_loss(td_target, old_val)

                # Periodic logging for performance diagnostics
                if global_step % 100 == 0:
                    if use_tensorboard:
                        writer.add_scalar("losses/td_loss", loss, global_step)
                        writer.add_scalar("losses/q_values", old_val.mean().item(), global_step)
                        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
                    print(f"SPS: {int(global_step / (time.time() - start_time))}, Episode: {int(episode_number)}, Step: {global_step}, Return: {total_reward}, Episode length: {episode_length}")

                # Gradient descent step on online network parameters
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # -----------------------------------
            # Target network update
            # -----------------------------------
            # This uses a "soft update":
            #   θ_target ← τ θ_online + (1-τ) θ_target
            # With τ=1.0 this becomes a hard copy update.
            if global_step % args.target_network_frequency == 0:
                for target_network_param, q_network_param in zip(target_network.parameters(), q_network.parameters()):
                    target_network_param.data.copy_(
                        args.tau * q_network_param.data + (1.0 - args.tau) * target_network_param.data
                    )


    # Sanity checks: since environment rewards are non-negative in your setup
    assert (returns_ >= 0.0).all(), "returns should be non-negative"
    assert (discounted_returns_ >= 0.0).all(), "discounted returns should be non-negative"

    envs.close()
    if use_tensorboard: writer.close()

    # Summary for quick reporting
    print("\n" + "="*60)
    print("DOUBLE DQN RUN SUMMARY")
    print("="*60)
    print(f"Total episodes run: {args.max_episodes}")
    print(f"Mean return: {returns_.mean():.2f}")
    print(f"Std return: {returns_.std():.2f}")
    print(f"Mean discounted return: {discounted_returns_.mean():.2f}")
    print(f"Mean episode length: {num_steps.mean():.2f}")
    print("="*60 + "\n")

    return returns_, discounted_returns_, num_steps


if __name__ == '__main__':
    # -------------------------
    # CLI arguments (experiment configuration)
    # -------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0,
        help="seed of the experiment")
    
    # Core training hyperparameters
    parser.add_argument("--max-episodes", type=int, default=1000)
    parser.add_argument("--learning-rate", type=float, default=2.5e-4,
        help="the learning rate of the optimizer")
    parser.add_argument("--num-envs", type=int, default=1,
        help="the number of parallel game environments")
    
    # Replay buffer + discounting
    parser.add_argument("--buffer-size", type=int, default=10000,
        help="the replay memory buffer size")
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma")
    
    # Target network updates (tau=1.0 -> hard update, tau<1.0 -> soft update)
    parser.add_argument("--tau", type=float, default=1.,
        help="the target network update rate")
    parser.add_argument("--target-network-frequency", type=int, default=500,
        help="the timesteps it takes to update the target network")
    
    # Training schedule
    parser.add_argument("--batch-size", type=int, default=128,
        help="the batch size of sample from the reply memory")
    parser.add_argument("--learning-starts", type=int, default=10000,
        help="timestep to start learning")
    parser.add_argument("--train-frequency", type=int, default=10,
        help="the frequency of training")

    # Exploration schedule
    parser.add_argument("--start-e", type=float, default=1,
        help="the starting epsilon for exploration")
    parser.add_argument("--end-e", type=float, default=0.05,
        help="the ending epsilon for exploration")
    parser.add_argument("--exploration-fraction", type=float, default=0.5,
        help="the fraction of `total-timesteps` it takes from start-e to go end-e")

    # Optional experiment tracking
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="if toggled, this experiment will be tracked with Weights and Biases")
    args = parser.parse_args()
    # fmt: on
    assert args.num_envs == 1, "vectorized envs are not supported at the moment"


    run_double_dqn(args, use_tensorboard=True, use_wandb=args.track)