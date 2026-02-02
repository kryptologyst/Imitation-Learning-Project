"""Imitation learning algorithms implementation.

This module contains implementations of various imitation learning algorithms
including behavioral cloning, GAIL, AIRL, and DAgger.
"""

from typing import Dict, List, Tuple, Optional, Any
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from ..models import PolicyNetwork, DiscriminatorNetwork, ValueNetwork
from ..utils import set_seed, get_device


class BehavioralCloning:
    """Behavioral Cloning (BC) algorithm.
    
    A supervised learning approach to imitation learning where the policy
    is trained to directly predict expert actions given states.
    
    Args:
        policy: Policy network to train
        learning_rate: Learning rate for optimizer
        batch_size: Batch size for training
        num_epochs: Number of training epochs
        device: Device to run training on
    """
    
    def __init__(
        self,
        policy: PolicyNetwork,
        learning_rate: float = 1e-3,
        batch_size: int = 64,
        num_epochs: int = 100,
        device: Optional[torch.device] = None
    ) -> None:
        self.policy = policy
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.device = device or get_device()
        
        self.policy.to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)
        self.criterion = nn.CrossEntropyLoss()
        
        self.training_losses: List[float] = []
        
    def train(self, expert_states: torch.Tensor, expert_actions: torch.Tensor) -> Dict[str, float]:
        """Train the policy using behavioral cloning.
        
        Args:
            expert_states: Expert state trajectories
            expert_actions: Expert action trajectories
            
        Returns:
            Dictionary containing training metrics
        """
        self.policy.train()
        
        # Create data loader
        dataset = TensorDataset(expert_states, expert_actions)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        epoch_losses = []
        
        for epoch in range(self.num_epochs):
            epoch_loss = 0.0
            num_batches = 0
            
            for states, actions in dataloader:
                states = states.to(self.device)
                actions = actions.to(self.device)
                
                # Forward pass
                logits = self.policy(states)
                loss = self.criterion(logits, actions)
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
            
            avg_loss = epoch_loss / num_batches
            epoch_losses.append(avg_loss)
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch + 1}/{self.num_epochs}, Loss: {avg_loss:.4f}")
        
        self.training_losses.extend(epoch_losses)
        
        return {
            "final_loss": epoch_losses[-1],
            "min_loss": min(epoch_losses),
            "avg_loss": np.mean(epoch_losses)
        }
    
    def get_action(self, state: torch.Tensor, deterministic: bool = True) -> torch.Tensor:
        """Get action from the trained policy.
        
        Args:
            state: Input state
            deterministic: Whether to use deterministic action selection
            
        Returns:
            Selected action
        """
        self.policy.eval()
        with torch.no_grad():
            if state.dim() == 1:
                state = state.unsqueeze(0)
            state = state.to(self.device)
            return self.policy.get_action(state, deterministic=deterministic)


class GAIL:
    """Generative Adversarial Imitation Learning (GAIL).
    
    An adversarial approach to imitation learning where a discriminator
    distinguishes between expert and policy-generated trajectories, and
    the policy learns to fool the discriminator.
    
    Args:
        policy: Policy network to train
        discriminator: Discriminator network
        learning_rate_policy: Learning rate for policy optimizer
        learning_rate_discriminator: Learning rate for discriminator optimizer
        batch_size: Batch size for training
        num_epochs: Number of training epochs
        device: Device to run training on
    """
    
    def __init__(
        self,
        policy: PolicyNetwork,
        discriminator: DiscriminatorNetwork,
        learning_rate_policy: float = 3e-4,
        learning_rate_discriminator: float = 1e-3,
        batch_size: int = 64,
        num_epochs: int = 100,
        device: Optional[torch.device] = None
    ) -> None:
        self.policy = policy
        self.discriminator = discriminator
        self.learning_rate_policy = learning_rate_policy
        self.learning_rate_discriminator = learning_rate_discriminator
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.device = device or get_device()
        
        self.policy.to(self.device)
        self.discriminator.to(self.device)
        
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate_policy)
        self.discriminator_optimizer = optim.Adam(self.discriminator.parameters(), lr=learning_rate_discriminator)
        
        self.policy_losses: List[float] = []
        self.discriminator_losses: List[float] = []
        
    def train_discriminator(
        self, 
        expert_states: torch.Tensor, 
        expert_actions: torch.Tensor,
        policy_states: torch.Tensor,
        policy_actions: torch.Tensor
    ) -> float:
        """Train the discriminator network.
        
        Args:
            expert_states: Expert state trajectories
            expert_actions: Expert action trajectories
            policy_states: Policy-generated state trajectories
            policy_actions: Policy-generated action trajectories
            
        Returns:
            Discriminator loss
        """
        self.discriminator.train()
        
        # Create labels: 1 for expert, 0 for policy
        expert_labels = torch.ones(expert_states.size(0), 1, device=self.device)
        policy_labels = torch.zeros(policy_states.size(0), 1, device=self.device)
        
        # Combine expert and policy data
        all_states = torch.cat([expert_states, policy_states], dim=0)
        all_actions = torch.cat([expert_actions, policy_actions], dim=0)
        all_labels = torch.cat([expert_labels, policy_labels], dim=0)
        
        # Forward pass
        logits = self.discriminator(all_states, all_actions)
        loss = nn.BCEWithLogitsLoss()(logits, all_labels)
        
        # Backward pass
        self.discriminator_optimizer.zero_grad()
        loss.backward()
        self.discriminator_optimizer.step()
        
        return loss.item()
    
    def train_policy(self, states: torch.Tensor, actions: torch.Tensor) -> float:
        """Train the policy network using discriminator rewards.
        
        Args:
            states: State trajectories
            actions: Action trajectories
            
        Returns:
            Policy loss
        """
        self.policy.train()
        
        # Get discriminator rewards (negative log probability of being expert)
        with torch.no_grad():
            rewards = self.discriminator.get_reward(states, actions)
        
        # Compute policy loss (negative log probability weighted by rewards)
        log_probs = self.policy.log_prob(states, actions)
        loss = -(log_probs * rewards).mean()
        
        # Backward pass
        self.policy_optimizer.zero_grad()
        loss.backward()
        self.policy_optimizer.step()
        
        return loss.item()
    
    def train(
        self, 
        expert_states: torch.Tensor, 
        expert_actions: torch.Tensor,
        policy_states: torch.Tensor,
        policy_actions: torch.Tensor
    ) -> Dict[str, float]:
        """Train both discriminator and policy networks.
        
        Args:
            expert_states: Expert state trajectories
            expert_actions: Expert action trajectories
            policy_states: Policy-generated state trajectories
            policy_actions: Policy-generated action trajectories
            
        Returns:
            Dictionary containing training metrics
        """
        policy_losses = []
        discriminator_losses = []
        
        for epoch in range(self.num_epochs):
            # Train discriminator
            d_loss = self.train_discriminator(
                expert_states, expert_actions, policy_states, policy_actions
            )
            
            # Train policy
            p_loss = self.train_policy(policy_states, policy_actions)
            
            policy_losses.append(p_loss)
            discriminator_losses.append(d_loss)
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch + 1}/{self.num_epochs}, "
                      f"Policy Loss: {p_loss:.4f}, Discriminator Loss: {d_loss:.4f}")
        
        self.policy_losses.extend(policy_losses)
        self.discriminator_losses.extend(discriminator_losses)
        
        return {
            "final_policy_loss": policy_losses[-1],
            "final_discriminator_loss": discriminator_losses[-1],
            "avg_policy_loss": np.mean(policy_losses),
            "avg_discriminator_loss": np.mean(discriminator_losses)
        }
    
    def get_action(self, state: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        """Get action from the trained policy.
        
        Args:
            state: Input state
            deterministic: Whether to use deterministic action selection
            
        Returns:
            Selected action
        """
        self.policy.eval()
        with torch.no_grad():
            if state.dim() == 1:
                state = state.unsqueeze(0)
            state = state.to(self.device)
            return self.policy.get_action(state, deterministic=deterministic)


class AIRL:
    """Adversarial Inverse Reinforcement Learning (AIRL).
    
    An extension of GAIL that learns both a reward function and a policy.
    The discriminator outputs a reward function that can be used for
    policy optimization.
    
    Args:
        policy: Policy network to train
        discriminator: Discriminator network (reward function)
        value_network: Value network for policy optimization
        learning_rate_policy: Learning rate for policy optimizer
        learning_rate_discriminator: Learning rate for discriminator optimizer
        learning_rate_value: Learning rate for value network optimizer
        batch_size: Batch size for training
        num_epochs: Number of training epochs
        device: Device to run training on
    """
    
    def __init__(
        self,
        policy: PolicyNetwork,
        discriminator: DiscriminatorNetwork,
        value_network: ValueNetwork,
        learning_rate_policy: float = 3e-4,
        learning_rate_discriminator: float = 1e-3,
        learning_rate_value: float = 1e-3,
        batch_size: int = 64,
        num_epochs: int = 100,
        device: Optional[torch.device] = None
    ) -> None:
        self.policy = policy
        self.discriminator = discriminator
        self.value_network = value_network
        self.learning_rate_policy = learning_rate_policy
        self.learning_rate_discriminator = learning_rate_discriminator
        self.learning_rate_value = learning_rate_value
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.device = device or get_device()
        
        self.policy.to(self.device)
        self.discriminator.to(self.device)
        self.value_network.to(self.device)
        
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate_policy)
        self.discriminator_optimizer = optim.Adam(self.discriminator.parameters(), lr=learning_rate_discriminator)
        self.value_optimizer = optim.Adam(self.value_network.parameters(), lr=learning_rate_value)
        
        self.policy_losses: List[float] = []
        self.discriminator_losses: List[float] = []
        self.value_losses: List[float] = []
        
    def train_discriminator(
        self, 
        expert_states: torch.Tensor, 
        expert_actions: torch.Tensor,
        expert_next_states: torch.Tensor,
        policy_states: torch.Tensor,
        policy_actions: torch.Tensor,
        policy_next_states: torch.Tensor
    ) -> float:
        """Train the discriminator network (reward function).
        
        Args:
            expert_states: Expert state trajectories
            expert_actions: Expert action trajectories
            expert_next_states: Expert next state trajectories
            policy_states: Policy-generated state trajectories
            policy_actions: Policy-generated action trajectories
            policy_next_states: Policy-generated next state trajectories
            
        Returns:
            Discriminator loss
        """
        self.discriminator.train()
        
        # Compute rewards for expert and policy trajectories
        expert_rewards = self.discriminator.get_reward(expert_states, expert_actions)
        policy_rewards = self.discriminator.get_reward(policy_states, policy_actions)
        
        # Compute value estimates
        expert_values = self.value_network(expert_states).squeeze()
        expert_next_values = self.value_network(expert_next_states).squeeze()
        policy_values = self.value_network(policy_states).squeeze()
        policy_next_values = self.value_network(policy_next_states).squeeze()
        
        # Compute advantages
        expert_advantages = expert_rewards + expert_next_values - expert_values
        policy_advantages = policy_rewards + policy_next_values - policy_values
        
        # Discriminator loss (expert should have higher advantage)
        loss = -(expert_advantages.mean() - policy_advantages.mean())
        
        # Backward pass
        self.discriminator_optimizer.zero_grad()
        loss.backward()
        self.discriminator_optimizer.step()
        
        return loss.item()
    
    def train_value_network(
        self,
        states: torch.Tensor,
        next_states: torch.Tensor,
        rewards: torch.Tensor
    ) -> float:
        """Train the value network.
        
        Args:
            states: State trajectories
            next_states: Next state trajectories
            rewards: Reward trajectories
            
        Returns:
            Value network loss
        """
        self.value_network.train()
        
        # Compute value estimates
        values = self.value_network(states).squeeze()
        next_values = self.value_network(next_states).squeeze()
        
        # Compute targets
        targets = rewards + next_values
        
        # Compute loss
        loss = nn.MSELoss()(values, targets.detach())
        
        # Backward pass
        self.value_optimizer.zero_grad()
        loss.backward()
        self.value_optimizer.step()
        
        return loss.item()
    
    def train_policy(self, states: torch.Tensor, actions: torch.Tensor) -> float:
        """Train the policy network using learned rewards.
        
        Args:
            states: State trajectories
            actions: Action trajectories
            
        Returns:
            Policy loss
        """
        self.policy.train()
        
        # Get discriminator rewards
        with torch.no_grad():
            rewards = self.discriminator.get_reward(states, actions)
        
        # Compute policy loss
        log_probs = self.policy.log_prob(states, actions)
        loss = -(log_probs * rewards).mean()
        
        # Backward pass
        self.policy_optimizer.zero_grad()
        loss.backward()
        self.policy_optimizer.step()
        
        return loss.item()
    
    def train(
        self, 
        expert_states: torch.Tensor, 
        expert_actions: torch.Tensor,
        expert_next_states: torch.Tensor,
        policy_states: torch.Tensor,
        policy_actions: torch.Tensor,
        policy_next_states: torch.Tensor
    ) -> Dict[str, float]:
        """Train all networks.
        
        Args:
            expert_states: Expert state trajectories
            expert_actions: Expert action trajectories
            expert_next_states: Expert next state trajectories
            policy_states: Policy-generated state trajectories
            policy_actions: Policy-generated action trajectories
            policy_next_states: Policy-generated next state trajectories
            
        Returns:
            Dictionary containing training metrics
        """
        policy_losses = []
        discriminator_losses = []
        value_losses = []
        
        for epoch in range(self.num_epochs):
            # Train discriminator
            d_loss = self.train_discriminator(
                expert_states, expert_actions, expert_next_states,
                policy_states, policy_actions, policy_next_states
            )
            
            # Train value network
            all_states = torch.cat([expert_states, policy_states], dim=0)
            all_next_states = torch.cat([expert_next_states, policy_next_states], dim=0)
            all_rewards = torch.cat([
                self.discriminator.get_reward(expert_states, expert_actions),
                self.discriminator.get_reward(policy_states, policy_actions)
            ], dim=0)
            
            v_loss = self.train_value_network(all_states, all_next_states, all_rewards)
            
            # Train policy
            p_loss = self.train_policy(policy_states, policy_actions)
            
            policy_losses.append(p_loss)
            discriminator_losses.append(d_loss)
            value_losses.append(v_loss)
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch + 1}/{self.num_epochs}, "
                      f"Policy Loss: {p_loss:.4f}, Discriminator Loss: {d_loss:.4f}, "
                      f"Value Loss: {v_loss:.4f}")
        
        self.policy_losses.extend(policy_losses)
        self.discriminator_losses.extend(discriminator_losses)
        self.value_losses.extend(value_losses)
        
        return {
            "final_policy_loss": policy_losses[-1],
            "final_discriminator_loss": discriminator_losses[-1],
            "final_value_loss": value_losses[-1],
            "avg_policy_loss": np.mean(policy_losses),
            "avg_discriminator_loss": np.mean(discriminator_losses),
            "avg_value_loss": np.mean(value_losses)
        }
    
    def get_action(self, state: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        """Get action from the trained policy.
        
        Args:
            state: Input state
            deterministic: Whether to use deterministic action selection
            
        Returns:
            Selected action
        """
        self.policy.eval()
        with torch.no_grad():
            if state.dim() == 1:
                state = state.unsqueeze(0)
            state = state.to(self.device)
            return self.policy.get_action(state, deterministic=deterministic)
