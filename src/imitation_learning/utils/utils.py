"""Utility functions for imitation learning.

This module contains utility functions for seeding, device management,
data processing, and other common operations.
"""

import random
import numpy as np
import torch
import gymnasium as gym
from typing import Optional, Tuple, List, Dict, Any
import os


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Make deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Set gymnasium seed
    gym.utils.seeding.np_random(seed)


def get_device() -> torch.device:
    """Get the best available device (CUDA > MPS > CPU).
    
    Returns:
        PyTorch device
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def collect_expert_data(
    env: gym.Env,
    expert_policy,
    num_episodes: int = 100,
    max_steps_per_episode: int = 1000,
    deterministic: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Collect expert demonstration data.
    
    Args:
        env: Gymnasium environment
        expert_policy: Expert policy function that takes state and returns action
        num_episodes: Number of episodes to collect
        max_steps_per_episode: Maximum steps per episode
        deterministic: Whether to use deterministic expert policy
        
    Returns:
        Tuple of (states, actions, rewards) arrays
    """
    states = []
    actions = []
    rewards = []
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_states = []
        episode_actions = []
        episode_rewards = []
        
        for step in range(max_steps_per_episode):
            # Get expert action
            if callable(expert_policy):
                action = expert_policy(state, deterministic=deterministic)
            else:
                action = expert_policy
            
            # Take step
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            episode_states.append(state)
            episode_actions.append(action)
            episode_rewards.append(reward)
            
            state = next_state
            
            if done:
                break
        
        states.extend(episode_states)
        actions.extend(episode_actions)
        rewards.extend(episode_rewards)
    
    return np.array(states), np.array(actions), np.array(rewards)


def collect_policy_data(
    env: gym.Env,
    policy,
    num_episodes: int = 100,
    max_steps_per_episode: int = 1000,
    deterministic: bool = False
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Collect policy-generated data.
    
    Args:
        env: Gymnasium environment
        policy: Policy function that takes state and returns action
        num_episodes: Number of episodes to collect
        max_steps_per_episode: Maximum steps per episode
        deterministic: Whether to use deterministic policy
        
    Returns:
        Tuple of (states, actions, rewards) arrays
    """
    states = []
    actions = []
    rewards = []
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_states = []
        episode_actions = []
        episode_rewards = []
        
        for step in range(max_steps_per_episode):
            # Get policy action
            if callable(policy):
                action = policy(state, deterministic=deterministic)
            else:
                action = policy
            
            # Take step
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            episode_states.append(state)
            episode_actions.append(action)
            episode_rewards.append(reward)
            
            state = next_state
            
            if done:
                break
        
        states.extend(episode_states)
        actions.extend(episode_actions)
        rewards.extend(episode_rewards)
    
    return np.array(states), np.array(actions), np.array(rewards)


def normalize_states(states: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Normalize states to zero mean and unit variance.
    
    Args:
        states: State array of shape (n_samples, state_dim)
        
    Returns:
        Tuple of (normalized_states, mean, std)
    """
    mean = np.mean(states, axis=0)
    std = np.std(states, axis=0)
    std = np.where(std == 0, 1.0, std)  # Avoid division by zero
    
    normalized_states = (states - mean) / std
    
    return normalized_states, mean, std


def denormalize_states(normalized_states: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Denormalize states back to original scale.
    
    Args:
        normalized_states: Normalized state array
        mean: Original mean
        std: Original standard deviation
        
    Returns:
        Denormalized states
    """
    return normalized_states * std + mean


def create_expert_policy(env_name: str, seed: int = 42) -> callable:
    """Create a simple expert policy for demonstration purposes.
    
    Args:
        env_name: Name of the environment
        seed: Random seed for the expert policy
        
    Returns:
        Expert policy function
    """
    set_seed(seed)
    
    if "CartPole" in env_name:
        # Simple heuristic policy for CartPole
        def expert_policy(state, deterministic=True):
            # Move right if pole is leaning right, left if leaning left
            if state[2] > 0:
                return 1  # Right
            else:
                return 0  # Left
    
    elif "MountainCar" in env_name:
        # Simple heuristic policy for MountainCar
        def expert_policy(state, deterministic=True):
            # Always push right to build momentum
            return 2  # Right
    
    else:
        # Random policy as fallback
        def expert_policy(state, deterministic=True):
            return np.random.randint(0, 2)  # Random action
    
    return expert_policy


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    filepath: str
) -> None:
    """Save model checkpoint.
    
    Args:
        model: PyTorch model
        optimizer: PyTorch optimizer
        epoch: Current epoch
        loss: Current loss
        filepath: Path to save checkpoint
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    
    torch.save(checkpoint, filepath)


def load_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    filepath: str
) -> Tuple[int, float]:
    """Load model checkpoint.
    
    Args:
        model: PyTorch model
        optimizer: PyTorch optimizer
        filepath: Path to checkpoint file
        
    Returns:
        Tuple of (epoch, loss)
    """
    checkpoint = torch.load(filepath, map_location='cpu')
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return checkpoint['epoch'], checkpoint['loss']


def compute_returns(rewards: List[float], gamma: float = 0.99) -> List[float]:
    """Compute discounted returns from rewards.
    
    Args:
        rewards: List of rewards
        gamma: Discount factor
        
    Returns:
        List of discounted returns
    """
    returns = []
    running_return = 0
    
    for reward in reversed(rewards):
        running_return = reward + gamma * running_return
        returns.insert(0, running_return)
    
    return returns


def compute_advantages(
    rewards: List[float],
    values: List[float],
    next_values: List[float],
    gamma: float = 0.99,
    lam: float = 0.95
) -> List[float]:
    """Compute generalized advantage estimates (GAE).
    
    Args:
        rewards: List of rewards
        values: List of value estimates
        next_values: List of next state value estimates
        gamma: Discount factor
        lam: GAE lambda parameter
        
    Returns:
        List of advantage estimates
    """
    advantages = []
    running_advantage = 0
    
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * next_values[t] - values[t]
        running_advantage = delta + gamma * lam * running_advantage
        advantages.insert(0, running_advantage)
    
    return advantages


def create_summary_dict(metrics: Dict[str, Any]) -> Dict[str, str]:
    """Create a summary dictionary with formatted metrics.
    
    Args:
        metrics: Dictionary of metrics
        
    Returns:
        Dictionary with formatted metric strings
    """
    summary = {}
    
    for key, value in metrics.items():
        if isinstance(value, float):
            summary[key] = f"{value:.4f}"
        elif isinstance(value, int):
            summary[key] = f"{value}"
        else:
            summary[key] = str(value)
    
    return summary
