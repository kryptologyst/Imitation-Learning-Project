#!/usr/bin/env python3
"""Simple example script demonstrating imitation learning.

This script shows how to use the modernized imitation learning package
to train a behavioral cloning model on CartPole.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import torch
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt

from imitation_learning import (
    BehavioralCloning, PolicyNetwork,
    ExpertDataCollector, ImitationEvaluator,
    set_seed, get_device, create_expert_policy
)


def main():
    """Main example function."""
    print("🤖 Imitation Learning Example")
    print("=" * 50)
    
    # Set random seed for reproducibility
    set_seed(42)
    print(f"Random seed set to 42")
    
    # Create environment
    env_name = "CartPole-v1"
    env = gym.make(env_name)
    env.reset(seed=42)
    print(f"Created environment: {env_name}")
    
    # Create expert policy
    expert_policy = create_expert_policy(env_name, seed=42)
    print("Created expert policy")
    
    # Collect expert demonstrations
    print("Collecting expert demonstrations...")
    collector = ExpertDataCollector(env, expert_policy)
    demonstrations = collector.collect_demonstrations(
        num_episodes=50,
        max_steps_per_episode=500,
        deterministic=True
    )
    
    print(f"Collected {len(demonstrations['states'])} expert transitions")
    
    # Create policy network
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    policy = PolicyNetwork(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dims=(64, 64),
        dropout_rate=0.1
    )
    print(f"Created policy network: {state_dim} -> {action_dim}")
    
    # Train behavioral cloning
    print("Training Behavioral Cloning model...")
    bc = BehavioralCloning(
        policy=policy,
        learning_rate=1e-3,
        batch_size=32,
        num_epochs=50,
        device=get_device()
    )
    
    expert_states = torch.tensor(demonstrations["states"], dtype=torch.float32)
    expert_actions = torch.tensor(demonstrations["actions"], dtype=torch.long)
    
    training_metrics = bc.train(expert_states, expert_actions)
    print(f"Training completed!")
    print(f"  Final loss: {training_metrics['final_loss']:.4f}")
    print(f"  Min loss: {training_metrics['min_loss']:.4f}")
    print(f"  Avg loss: {training_metrics['avg_loss']:.4f}")
    
    # Evaluate trained policy
    print("\nEvaluating trained policy...")
    evaluator = ImitationEvaluator(env)
    eval_metrics = evaluator.evaluate_policy(
        bc,
        num_episodes=20,
        deterministic=True
    )
    
    print("Evaluation Results:")
    print(f"  Mean reward: {eval_metrics['mean_reward']:.2f} ± {eval_metrics['std_reward']:.2f}")
    print(f"  Success rate: {eval_metrics['success_rate']:.1%}")
    print(f"  Mean episode length: {eval_metrics['mean_length']:.1f}")
    print(f"  95% Confidence Interval: [{eval_metrics['ci_lower']:.2f}, {eval_metrics['ci_upper']:.2f}]")
    
    # Compare with random policy
    print("\nComparing with random policy...")
    def random_policy(state, deterministic=True):
        return env.action_space.sample()
    
    random_metrics = evaluator.evaluate_policy(
        random_policy,
        num_episodes=20,
        deterministic=False
    )
    
    print("Random Policy Results:")
    print(f"  Mean reward: {random_metrics['mean_reward']:.2f} ± {random_metrics['std_reward']:.2f}")
    print(f"  Success rate: {random_metrics['success_rate']:.1%}")
    
    # Performance improvement
    improvement = eval_metrics['mean_reward'] - random_metrics['mean_reward']
    print(f"\nPerformance improvement: {improvement:.2f} reward points")
    
    # Plot training curve
    if hasattr(bc, 'training_losses') and bc.training_losses:
        plt.figure(figsize=(10, 6))
        plt.plot(bc.training_losses)
        plt.title('Behavioral Cloning Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.savefig('training_loss.png', dpi=300, bbox_inches='tight')
        print("\nTraining curve saved as 'training_loss.png'")
    
    # Clean up
    env.close()
    print("\n✅ Example completed successfully!")
    print("\nNext steps:")
    print("  - Try different algorithms: python scripts/train.py --algorithm gail")
    print("  - Launch interactive demo: streamlit run demo/app.py")
    print("  - Explore more environments: python scripts/train.py --env MountainCar-v0")


if __name__ == "__main__":
    main()
