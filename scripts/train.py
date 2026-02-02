#!/usr/bin/env python3
"""Modern imitation learning training script.

This script demonstrates how to use the imitation learning package
with behavioral cloning, GAIL, and AIRL algorithms.
"""

import argparse
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import torch
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
from omegaconf import OmegaConf

from imitation_learning import (
    BehavioralCloning, GAIL, AIRL,
    PolicyNetwork, DiscriminatorNetwork, ValueNetwork,
    ExpertDataCollector, ImitationEvaluator,
    set_seed, get_device, create_expert_policy
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train imitation learning algorithms")
    
    parser.add_argument("--env", type=str, default="CartPole-v1", 
                       help="Environment name")
    parser.add_argument("--algorithm", type=str, default="bc", 
                       choices=["bc", "gail", "airl"],
                       help="Algorithm to use")
    parser.add_argument("--num_episodes", type=int, default=100,
                       help="Number of expert episodes to collect")
    parser.add_argument("--num_epochs", type=int, default=100,
                       help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=64,
                       help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=1e-3,
                       help="Learning rate")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    parser.add_argument("--device", type=str, default="auto",
                       help="Device to use (auto, cpu, cuda, mps)")
    parser.add_argument("--save_dir", type=str, default="checkpoints",
                       help="Directory to save checkpoints")
    parser.add_argument("--config", type=str, default=None,
                       help="Path to config file")
    parser.add_argument("--eval_episodes", type=int, default=50,
                       help="Number of evaluation episodes")
    parser.add_argument("--render", action="store_true",
                       help="Render evaluation episodes")
    
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """Load configuration from file."""
    if config_path and os.path.exists(config_path):
        return OmegaConf.load(config_path)
    return {}


def create_environment(env_name: str, seed: int):
    """Create and configure environment."""
    env = gym.make(env_name)
    env.reset(seed=seed)
    return env


def collect_expert_data(env, expert_policy, num_episodes: int, seed: int):
    """Collect expert demonstration data."""
    collector = ExpertDataCollector(env, expert_policy)
    
    print(f"Collecting {num_episodes} expert episodes...")
    demonstrations = collector.collect_demonstrations(
        num_episodes=num_episodes,
        deterministic=True
    )
    
    print(f"Collected {len(demonstrations['states'])} expert transitions")
    return demonstrations


def train_behavioral_cloning(env, demonstrations, config):
    """Train behavioral cloning algorithm."""
    print("Training Behavioral Cloning...")
    
    # Create policy network
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    policy = PolicyNetwork(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dims=(64, 64),
        dropout_rate=0.1
    )
    
    # Create algorithm
    bc = BehavioralCloning(
        policy=policy,
        learning_rate=config.get("learning_rate", 1e-3),
        batch_size=config.get("batch_size", 64),
        num_epochs=config.get("num_epochs", 100),
        device=config.get("device", get_device())
    )
    
    # Prepare data
    expert_states = torch.tensor(demonstrations["states"], dtype=torch.float32)
    expert_actions = torch.tensor(demonstrations["actions"], dtype=torch.long)
    
    # Train
    metrics = bc.train(expert_states, expert_actions)
    
    print(f"Training completed. Final loss: {metrics['final_loss']:.4f}")
    
    return bc, metrics


def train_gail(env, demonstrations, config):
    """Train GAIL algorithm."""
    print("Training GAIL...")
    
    # Create networks
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    policy = PolicyNetwork(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dims=(64, 64)
    )
    
    discriminator = DiscriminatorNetwork(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dims=(64, 64)
    )
    
    # Create algorithm
    gail = GAIL(
        policy=policy,
        discriminator=discriminator,
        learning_rate_policy=config.get("learning_rate_policy", 3e-4),
        learning_rate_discriminator=config.get("learning_rate_discriminator", 1e-3),
        batch_size=config.get("batch_size", 64),
        num_epochs=config.get("num_epochs", 100),
        device=config.get("device", get_device())
    )
    
    # Prepare expert data
    expert_states = torch.tensor(demonstrations["states"], dtype=torch.float32)
    expert_actions = torch.tensor(demonstrations["actions"], dtype=torch.long)
    
    # Generate policy data (simplified - in practice, you'd collect this during training)
    policy_states = expert_states.clone()
    policy_actions = expert_actions.clone()
    
    # Train
    metrics = gail.train(expert_states, expert_actions, policy_states, policy_actions)
    
    print(f"Training completed. Final policy loss: {metrics['final_policy_loss']:.4f}")
    
    return gail, metrics


def train_airl(env, demonstrations, config):
    """Train AIRL algorithm."""
    print("Training AIRL...")
    
    # Create networks
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    policy = PolicyNetwork(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dims=(64, 64)
    )
    
    discriminator = DiscriminatorNetwork(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dims=(64, 64)
    )
    
    value_network = ValueNetwork(
        state_dim=state_dim,
        hidden_dims=(64, 64)
    )
    
    # Create algorithm
    airl = AIRL(
        policy=policy,
        discriminator=discriminator,
        value_network=value_network,
        learning_rate_policy=config.get("learning_rate_policy", 3e-4),
        learning_rate_discriminator=config.get("learning_rate_discriminator", 1e-3),
        learning_rate_value=config.get("learning_rate_value", 1e-3),
        batch_size=config.get("batch_size", 64),
        num_epochs=config.get("num_epochs", 100),
        device=config.get("device", get_device())
    )
    
    # Prepare data
    expert_states = torch.tensor(demonstrations["states"], dtype=torch.float32)
    expert_actions = torch.tensor(demonstrations["actions"], dtype=torch.long)
    expert_next_states = torch.tensor(demonstrations["next_states"], dtype=torch.float32)
    
    # Generate policy data (simplified)
    policy_states = expert_states.clone()
    policy_actions = expert_actions.clone()
    policy_next_states = expert_next_states.clone()
    
    # Train
    metrics = airl.train(
        expert_states, expert_actions, expert_next_states,
        policy_states, policy_actions, policy_next_states
    )
    
    print(f"Training completed. Final policy loss: {metrics['final_policy_loss']:.4f}")
    
    return airl, metrics


def evaluate_policy(env, policy, num_episodes: int, render: bool = False):
    """Evaluate trained policy."""
    evaluator = ImitationEvaluator(env)
    
    print(f"Evaluating policy over {num_episodes} episodes...")
    metrics = evaluator.evaluate_policy(
        policy=policy,
        num_episodes=num_episodes,
        deterministic=True,
        render=render
    )
    
    print(f"Evaluation Results:")
    print(f"  Mean Reward: {metrics['mean_reward']:.2f} ± {metrics['std_reward']:.2f}")
    print(f"  Success Rate: {metrics['success_rate']:.2%}")
    print(f"  Mean Length: {metrics['mean_length']:.1f}")
    print(f"  95% CI: [{metrics['ci_lower']:.2f}, {metrics['ci_upper']:.2f}]")
    
    return metrics


def save_results(algorithm, metrics, save_dir: str):
    """Save training results."""
    os.makedirs(save_dir, exist_ok=True)
    
    # Save model
    model_path = os.path.join(save_dir, f"{algorithm.__class__.__name__.lower()}_model.pt")
    torch.save(algorithm.policy.state_dict(), model_path)
    
    # Save metrics
    metrics_path = os.path.join(save_dir, f"{algorithm.__class__.__name__.lower()}_metrics.json")
    import json
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"Results saved to {save_dir}")


def main():
    """Main training function."""
    args = parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Determine device
    if args.device == "auto":
        device = get_device()
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    print(f"Environment: {args.env}")
    print(f"Algorithm: {args.algorithm}")
    print(f"Seed: {args.seed}")
    
    # Load configuration
    config = load_config(args.config) if args.config else {}
    config.update({
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "num_epochs": args.num_epochs,
        "device": device
    })
    
    # Create environment
    env = create_environment(args.env, args.seed)
    
    # Create expert policy
    expert_policy = create_expert_policy(args.env, args.seed)
    
    # Collect expert data
    demonstrations = collect_expert_data(env, expert_policy, args.num_episodes, args.seed)
    
    # Train algorithm
    if args.algorithm == "bc":
        algorithm, metrics = train_behavioral_cloning(env, demonstrations, config)
    elif args.algorithm == "gail":
        algorithm, metrics = train_gail(env, demonstrations, config)
    elif args.algorithm == "airl":
        algorithm, metrics = train_airl(env, demonstrations, config)
    else:
        raise ValueError(f"Unknown algorithm: {args.algorithm}")
    
    # Evaluate policy
    eval_metrics = evaluate_policy(algorithm, args.eval_episodes, args.render)
    
    # Save results
    save_results(algorithm, {**metrics, **eval_metrics}, args.save_dir)
    
    env.close()
    print("Training completed successfully!")


if __name__ == "__main__":
    main()
