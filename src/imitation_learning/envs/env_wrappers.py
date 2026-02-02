"""Environment wrappers and utilities for imitation learning.

This module contains environment wrappers and utilities for data collection
and environment management.
"""

from typing import Dict, List, Tuple, Optional, Any, Callable
import numpy as np
import torch
import gymnasium as gym
from gymnasium import Wrapper
import pickle
import os


class ExpertDataCollector:
    """Collect expert demonstration data from environments.
    
    Provides functionality to collect expert demonstrations using
    various expert policies and save them for training.
    
    Args:
        env: Gymnasium environment
        expert_policy: Expert policy function
        device: Device for computation
    """
    
    def __init__(
        self,
        env: gym.Env,
        expert_policy: Callable,
        device: Optional[torch.device] = None
    ):
        self.env = env
        self.expert_policy = expert_policy
        self.device = device or torch.device("cpu")
        
    def collect_demonstrations(
        self,
        num_episodes: int = 100,
        max_steps_per_episode: int = 1000,
        deterministic: bool = True,
        save_path: Optional[str] = None
    ) -> Dict[str, np.ndarray]:
        """Collect expert demonstrations.
        
        Args:
            num_episodes: Number of episodes to collect
            max_steps_per_episode: Maximum steps per episode
            deterministic: Whether to use deterministic expert policy
            save_path: Path to save demonstrations
            
        Returns:
            Dictionary containing demonstration data
        """
        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []
        infos = []
        
        for episode in range(num_episodes):
            state, info = self.env.reset()
            episode_states = []
            episode_actions = []
            episode_rewards = []
            episode_next_states = []
            episode_dones = []
            episode_infos = []
            
            for step in range(max_steps_per_episode):
                # Get expert action
                action = self.expert_policy(state, deterministic=deterministic)
                
                # Take step
                next_state, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                
                episode_states.append(state)
                episode_actions.append(action)
                episode_rewards.append(reward)
                episode_next_states.append(next_state)
                episode_dones.append(done)
                episode_infos.append(info)
                
                state = next_state
                
                if done:
                    break
            
            states.extend(episode_states)
            actions.extend(episode_actions)
            rewards.extend(episode_rewards)
            next_states.extend(episode_next_states)
            dones.extend(episode_dones)
            infos.extend(episode_infos)
        
        demonstrations = {
            "states": np.array(states),
            "actions": np.array(actions),
            "rewards": np.array(rewards),
            "next_states": np.array(next_states),
            "dones": np.array(dones),
            "infos": infos
        }
        
        if save_path:
            self.save_demonstrations(demonstrations, save_path)
        
        return demonstrations
    
    def save_demonstrations(self, demonstrations: Dict[str, np.ndarray], filepath: str) -> None:
        """Save demonstrations to file.
        
        Args:
            demonstrations: Demonstration data
            filepath: Path to save file
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, 'wb') as f:
            pickle.dump(demonstrations, f)
    
    def load_demonstrations(self, filepath: str) -> Dict[str, np.ndarray]:
        """Load demonstrations from file.
        
        Args:
            filepath: Path to load file from
            
        Returns:
            Demonstration data
        """
        with open(filepath, 'rb') as f:
            demonstrations = pickle.load(f)
        
        return demonstrations


class EnvironmentWrapper(Wrapper):
    """Wrapper for environment modifications.
    
    Provides common environment modifications useful for imitation learning,
    such as state normalization, reward shaping, and action clipping.
    
    Args:
        env: Gymnasium environment
        normalize_states: Whether to normalize states
        normalize_rewards: Whether to normalize rewards
        clip_actions: Whether to clip actions
        state_mean: State mean for normalization
        state_std: State std for normalization
        reward_mean: Reward mean for normalization
        reward_std: Reward std for normalization
    """
    
    def __init__(
        self,
        env: gym.Env,
        normalize_states: bool = False,
        normalize_rewards: bool = False,
        clip_actions: bool = False,
        state_mean: Optional[np.ndarray] = None,
        state_std: Optional[np.ndarray] = None,
        reward_mean: Optional[float] = None,
        reward_std: Optional[float] = None
    ):
        super().__init__(env)
        
        self.normalize_states = normalize_states
        self.normalize_rewards = normalize_rewards
        self.clip_actions = clip_actions
        
        # State normalization parameters
        self.state_mean = state_mean
        self.state_std = state_std
        
        # Reward normalization parameters
        self.reward_mean = reward_mean
        self.reward_std = reward_std
        
        # Initialize normalization parameters if not provided
        if normalize_states and (state_mean is None or state_std is None):
            self._compute_state_stats()
        
        if normalize_rewards and (reward_mean is None or reward_std is None):
            self._compute_reward_stats()
    
    def _compute_state_stats(self) -> None:
        """Compute state statistics for normalization."""
        states = []
        
        # Collect states from random episodes
        for _ in range(100):
            state, _ = self.env.reset()
            states.append(state)
            
            for _ in range(50):  # Collect 50 steps
                action = self.env.action_space.sample()
                state, _, terminated, truncated, _ = self.env.step(action)
                states.append(state)
                
                if terminated or truncated:
                    break
        
        states = np.array(states)
        self.state_mean = np.mean(states, axis=0)
        self.state_std = np.std(states, axis=0)
        self.state_std = np.where(self.state_std == 0, 1.0, self.state_std)
    
    def _compute_reward_stats(self) -> None:
        """Compute reward statistics for normalization."""
        rewards = []
        
        # Collect rewards from random episodes
        for _ in range(100):
            state, _ = self.env.reset()
            
            for _ in range(50):  # Collect 50 steps
                action = self.env.action_space.sample()
                _, reward, terminated, truncated, _ = self.env.step(action)
                rewards.append(reward)
                
                if terminated or truncated:
                    break
        
        rewards = np.array(rewards)
        self.reward_mean = np.mean(rewards)
        self.reward_std = np.std(rewards)
        if self.reward_std == 0:
            self.reward_std = 1.0
    
    def reset(self, **kwargs) -> Tuple[np.ndarray, Dict]:
        """Reset environment and normalize initial state."""
        state, info = self.env.reset(**kwargs)
        
        if self.normalize_states:
            state = self._normalize_state(state)
        
        return state, info
    
    def step(self, action) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Take step and normalize state and reward."""
        # Clip action if requested
        if self.clip_actions:
            action = np.clip(action, self.env.action_space.low, self.env.action_space.high)
        
        state, reward, terminated, truncated, info = self.env.step(action)
        
        # Normalize state
        if self.normalize_states:
            state = self._normalize_state(state)
        
        # Normalize reward
        if self.normalize_rewards:
            reward = self._normalize_reward(reward)
        
        return state, reward, terminated, truncated, info
    
    def _normalize_state(self, state: np.ndarray) -> np.ndarray:
        """Normalize state using computed statistics."""
        return (state - self.state_mean) / self.state_std
    
    def _normalize_reward(self, reward: float) -> float:
        """Normalize reward using computed statistics."""
        return (reward - self.reward_mean) / self.reward_std
    
    def denormalize_state(self, normalized_state: np.ndarray) -> np.ndarray:
        """Denormalize state back to original scale."""
        return normalized_state * self.state_std + self.state_mean
    
    def denormalize_reward(self, normalized_reward: float) -> float:
        """Denormalize reward back to original scale."""
        return normalized_reward * self.reward_std + self.reward_mean


class VectorizedEnvironment:
    """Vectorized environment for parallel data collection.
    
    Provides functionality to run multiple environments in parallel
    for efficient data collection.
    
    Args:
        env_fns: List of environment creation functions
        num_envs: Number of parallel environments
    """
    
    def __init__(self, env_fns: List[Callable], num_envs: int):
        self.env_fns = env_fns
        self.num_envs = num_envs
        self.envs = [fn() for fn in env_fns[:num_envs]]
        
        # Check that all environments have the same spaces
        self.observation_space = self.envs[0].observation_space
        self.action_space = self.envs[0].action_space
        
        for env in self.envs[1:]:
            assert env.observation_space == self.observation_space
            assert env.action_space == self.action_space
    
    def reset(self) -> Tuple[np.ndarray, List[Dict]]:
        """Reset all environments."""
        states = []
        infos = []
        
        for env in self.envs:
            state, info = env.reset()
            states.append(state)
            infos.append(info)
        
        return np.array(states), infos
    
    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[Dict]]:
        """Step all environments."""
        states = []
        rewards = []
        terminateds = []
        truncateds = []
        infos = []
        
        for i, env in enumerate(self.envs):
            state, reward, terminated, truncated, info = env.step(actions[i])
            states.append(state)
            rewards.append(reward)
            terminateds.append(terminated)
            truncateds.append(truncated)
            infos.append(info)
        
        return (
            np.array(states),
            np.array(rewards),
            np.array(terminateds),
            np.array(truncateds),
            infos
        )
    
    def close(self) -> None:
        """Close all environments."""
        for env in self.envs:
            env.close()
    
    def render(self) -> None:
        """Render all environments."""
        for env in self.envs:
            env.render()
