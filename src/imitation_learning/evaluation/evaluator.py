"""Evaluation utilities for imitation learning.

This module contains evaluation functions and metrics for assessing
imitation learning performance.
"""

from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import torch
import gymnasium as gym
from scipy import stats
import matplotlib.pyplot as plt
import pandas as pd


class ImitationEvaluator:
    """Evaluator for imitation learning algorithms.
    
    Provides comprehensive evaluation metrics including performance,
    stability, and imitation quality measures.
    
    Args:
        env: Gymnasium environment
        device: Device for computation
    """
    
    def __init__(self, env: gym.Env, device: Optional[torch.device] = None):
        self.env = env
        self.device = device or torch.device("cpu")
        
    def evaluate_policy(
        self,
        policy,
        num_episodes: int = 100,
        deterministic: bool = True,
        render: bool = False
    ) -> Dict[str, float]:
        """Evaluate policy performance.
        
        Args:
            policy: Policy to evaluate
            num_episodes: Number of evaluation episodes
            deterministic: Whether to use deterministic policy
            render: Whether to render episodes
            
        Returns:
            Dictionary of evaluation metrics
        """
        episode_rewards = []
        episode_lengths = []
        success_count = 0
        
        for episode in range(num_episodes):
            state, _ = self.env.reset()
            episode_reward = 0
            episode_length = 0
            
            while True:
                if callable(policy):
                    action = policy(state, deterministic=deterministic)
                else:
                    action = policy
                
                if isinstance(action, torch.Tensor):
                    action = action.cpu().numpy()
                if hasattr(action, 'item'):
                    action = action.item()
                
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                episode_reward += reward
                episode_length += 1
                
                if render and episode == 0:  # Only render first episode
                    self.env.render()
                
                state = next_state
                
                if terminated or truncated:
                    break
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            
            # Define success criteria (environment-specific)
            if self._is_success(episode_reward, episode_length):
                success_count += 1
        
        # Compute statistics
        mean_reward = np.mean(episode_rewards)
        std_reward = np.std(episode_rewards)
        mean_length = np.mean(episode_lengths)
        success_rate = success_count / num_episodes
        
        # Confidence intervals
        ci_lower, ci_upper = stats.t.interval(
            0.95, len(episode_rewards) - 1,
            loc=mean_reward, scale=stats.sem(episode_rewards)
        )
        
        return {
            "mean_reward": mean_reward,
            "std_reward": std_reward,
            "mean_length": mean_length,
            "success_rate": success_rate,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "min_reward": np.min(episode_rewards),
            "max_reward": np.max(episode_rewards),
            "episode_rewards": episode_rewards,
            "episode_lengths": episode_lengths
        }
    
    def _is_success(self, reward: float, length: int) -> bool:
        """Determine if an episode is successful.
        
        Args:
            reward: Episode reward
            length: Episode length
            
        Returns:
            True if episode is successful
        """
        env_name = self.env.spec.id if hasattr(self.env.spec, 'id') else str(self.env)
        
        if "CartPole" in env_name:
            return reward >= 195.0  # CartPole success threshold
        elif "MountainCar" in env_name:
            return reward >= -110.0  # MountainCar success threshold
        elif "Acrobot" in env_name:
            return reward >= -100.0  # Acrobot success threshold
        else:
            # Generic success criteria
            return reward > np.mean([reward]) + np.std([reward])
    
    def compare_policies(
        self,
        policies: Dict[str, Any],
        num_episodes: int = 100,
        deterministic: bool = True
    ) -> pd.DataFrame:
        """Compare multiple policies.
        
        Args:
            policies: Dictionary of policy names to policy functions
            num_episodes: Number of evaluation episodes per policy
            deterministic: Whether to use deterministic policies
            
        Returns:
            DataFrame with comparison results
        """
        results = []
        
        for name, policy in policies.items():
            metrics = self.evaluate_policy(policy, num_episodes, deterministic)
            
            results.append({
                "Policy": name,
                "Mean Reward": metrics["mean_reward"],
                "Std Reward": metrics["std_reward"],
                "Success Rate": metrics["success_rate"],
                "Mean Length": metrics["mean_length"],
                "CI Lower": metrics["ci_lower"],
                "CI Upper": metrics["ci_upper"]
            })
        
        return pd.DataFrame(results)
    
    def compute_imitation_metrics(
        self,
        expert_states: np.ndarray,
        expert_actions: np.ndarray,
        policy_states: np.ndarray,
        policy_actions: np.ndarray
    ) -> Dict[str, float]:
        """Compute imitation quality metrics.
        
        Args:
            expert_states: Expert state trajectories
            expert_actions: Expert action trajectories
            policy_states: Policy state trajectories
            policy_actions: Policy action trajectories
            
        Returns:
            Dictionary of imitation metrics
        """
        # Action accuracy (for discrete actions)
        if expert_actions.dtype in [np.int32, np.int64]:
            action_accuracy = np.mean(expert_actions == policy_actions)
        else:
            # For continuous actions, compute MSE
            action_accuracy = -np.mean((expert_actions - policy_actions) ** 2)
        
        # State distribution similarity (using KL divergence approximation)
        state_kl_div = self._compute_kl_divergence(expert_states, policy_states)
        
        # Action distribution similarity
        action_kl_div = self._compute_kl_divergence(expert_actions, policy_actions)
        
        # Trajectory length similarity
        expert_length = len(expert_states)
        policy_length = len(policy_states)
        length_similarity = 1.0 - abs(expert_length - policy_length) / max(expert_length, policy_length)
        
        return {
            "action_accuracy": action_accuracy,
            "state_kl_divergence": state_kl_div,
            "action_kl_divergence": action_kl_div,
            "length_similarity": length_similarity
        }
    
    def _compute_kl_divergence(self, data1: np.ndarray, data2: np.ndarray) -> float:
        """Compute approximate KL divergence between two datasets.
        
        Args:
            data1: First dataset
            data2: Second dataset
            
        Returns:
            Approximate KL divergence
        """
        try:
            # Use histogram-based KL divergence estimation
            bins = 50
            hist1, _ = np.histogram(data1.flatten(), bins=bins, density=True)
            hist2, _ = np.histogram(data2.flatten(), bins=bins, density=True)
            
            # Add small epsilon to avoid log(0)
            eps = 1e-10
            hist1 = hist1 + eps
            hist2 = hist2 + eps
            
            # Normalize
            hist1 = hist1 / np.sum(hist1)
            hist2 = hist2 / np.sum(hist2)
            
            # Compute KL divergence
            kl_div = np.sum(hist1 * np.log(hist1 / hist2))
            
            return kl_div
        except:
            return float('inf')


class MetricsTracker:
    """Track and visualize training metrics.
    
    Provides functionality to track, store, and visualize training
    metrics over time.
    
    Args:
        log_dir: Directory to save logs
    """
    
    def __init__(self, log_dir: str = "logs"):
        self.log_dir = log_dir
        self.metrics_history = {}
        
    def log_metrics(self, metrics: Dict[str, float], step: int) -> None:
        """Log metrics at a given step.
        
        Args:
            metrics: Dictionary of metrics to log
            step: Current step
        """
        for key, value in metrics.items():
            if key not in self.metrics_history:
                self.metrics_history[key] = []
            self.metrics_history[key].append((step, value))
    
    def plot_metrics(self, save_path: Optional[str] = None) -> None:
        """Plot training metrics.
        
        Args:
            save_path: Path to save plot
        """
        if not self.metrics_history:
            print("No metrics to plot")
            return
        
        num_metrics = len(self.metrics_history)
        fig, axes = plt.subplots(1, num_metrics, figsize=(5 * num_metrics, 4))
        
        if num_metrics == 1:
            axes = [axes]
        
        for i, (metric_name, history) in enumerate(self.metrics_history.items()):
            steps, values = zip(*history)
            
            axes[i].plot(steps, values)
            axes[i].set_title(metric_name)
            axes[i].set_xlabel("Step")
            axes[i].set_ylabel("Value")
            axes[i].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
    
    def get_best_metric(self, metric_name: str, mode: str = "max") -> Tuple[int, float]:
        """Get the best value for a metric.
        
        Args:
            metric_name: Name of the metric
            mode: "max" or "min"
            
        Returns:
            Tuple of (step, best_value)
        """
        if metric_name not in self.metrics_history:
            raise ValueError(f"Metric {metric_name} not found")
        
        history = self.metrics_history[metric_name]
        steps, values = zip(*history)
        
        if mode == "max":
            best_idx = np.argmax(values)
        elif mode == "min":
            best_idx = np.argmin(values)
        else:
            raise ValueError("Mode must be 'max' or 'min'")
        
        return steps[best_idx], values[best_idx]
    
    def save_metrics(self, filepath: str) -> None:
        """Save metrics to file.
        
        Args:
            filepath: Path to save metrics
        """
        import json
        
        # Convert to serializable format
        serializable_history = {}
        for key, history in self.metrics_history.items():
            serializable_history[key] = [(int(step), float(value)) for step, value in history]
        
        with open(filepath, 'w') as f:
            json.dump(serializable_history, f, indent=2)
    
    def load_metrics(self, filepath: str) -> None:
        """Load metrics from file.
        
        Args:
            filepath: Path to load metrics from
        """
        import json
        
        with open(filepath, 'r') as f:
            serializable_history = json.load(f)
        
        self.metrics_history = {}
        for key, history in serializable_history.items():
            self.metrics_history[key] = [(step, value) for step, value in history]
