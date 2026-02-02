"""Tests for imitation learning package."""

import pytest
import torch
import numpy as np
import gymnasium as gym

from src.imitation_learning import (
    BehavioralCloning, GAIL, AIRL,
    PolicyNetwork, DiscriminatorNetwork, ValueNetwork,
    ExpertDataCollector, ImitationEvaluator,
    set_seed, get_device, create_expert_policy
)


class TestPolicyNetwork:
    """Test PolicyNetwork class."""
    
    def test_initialization(self):
        """Test network initialization."""
        policy = PolicyNetwork(state_dim=4, action_dim=2)
        assert policy.state_dim == 4
        assert policy.action_dim == 2
    
    def test_forward(self):
        """Test forward pass."""
        policy = PolicyNetwork(state_dim=4, action_dim=2)
        state = torch.randn(1, 4)
        output = policy(state)
        assert output.shape == (1, 2)
    
    def test_get_action(self):
        """Test action sampling."""
        policy = PolicyNetwork(state_dim=4, action_dim=2)
        state = torch.randn(1, 4)
        
        # Test deterministic action
        action = policy.get_action(state, deterministic=True)
        assert action.shape == (1,)
        assert action.dtype == torch.long
        
        # Test stochastic action
        action = policy.get_action(state, deterministic=False)
        assert action.shape == (1,)
        assert action.dtype == torch.long
    
    def test_log_prob(self):
        """Test log probability computation."""
        policy = PolicyNetwork(state_dim=4, action_dim=2)
        state = torch.randn(1, 4)
        action = torch.tensor([0])
        
        log_prob = policy.log_prob(state, action)
        assert log_prob.shape == (1,)
        assert log_prob.dtype == torch.float32


class TestBehavioralCloning:
    """Test BehavioralCloning class."""
    
    def test_initialization(self):
        """Test algorithm initialization."""
        policy = PolicyNetwork(state_dim=4, action_dim=2)
        bc = BehavioralCloning(policy=policy)
        assert bc.policy == policy
    
    def test_train(self):
        """Test training process."""
        policy = PolicyNetwork(state_dim=4, action_dim=2)
        bc = BehavioralCloning(policy=policy, num_epochs=2)
        
        # Create dummy data
        states = torch.randn(100, 4)
        actions = torch.randint(0, 2, (100,))
        
        # Train
        metrics = bc.train(states, actions)
        
        assert "final_loss" in metrics
        assert "min_loss" in metrics
        assert "avg_loss" in metrics
        assert isinstance(metrics["final_loss"], float)
    
    def test_get_action(self):
        """Test action selection."""
        policy = PolicyNetwork(state_dim=4, action_dim=2)
        bc = BehavioralCloning(policy=policy)
        
        state = torch.randn(4)
        action = bc.get_action(state)
        assert action.shape == (1,)
        assert action.dtype == torch.long


class TestDiscriminatorNetwork:
    """Test DiscriminatorNetwork class."""
    
    def test_initialization(self):
        """Test network initialization."""
        discriminator = DiscriminatorNetwork(state_dim=4, action_dim=2)
        assert discriminator.state_dim == 4
        assert discriminator.action_dim == 2
    
    def test_forward(self):
        """Test forward pass."""
        discriminator = DiscriminatorNetwork(state_dim=4, action_dim=2)
        state = torch.randn(1, 4)
        action = torch.randn(1, 2)
        output = discriminator(state, action)
        assert output.shape == (1, 1)
    
    def test_get_reward(self):
        """Test reward computation."""
        discriminator = DiscriminatorNetwork(state_dim=4, action_dim=2)
        state = torch.randn(1, 4)
        action = torch.randn(1, 2)
        reward = discriminator.get_reward(state, action)
        assert reward.shape == (1,)
        assert reward.dtype == torch.float32


class TestGAIL:
    """Test GAIL class."""
    
    def test_initialization(self):
        """Test algorithm initialization."""
        policy = PolicyNetwork(state_dim=4, action_dim=2)
        discriminator = DiscriminatorNetwork(state_dim=4, action_dim=2)
        gail = GAIL(policy=policy, discriminator=discriminator, num_epochs=2)
        
        assert gail.policy == policy
        assert gail.discriminator == discriminator
    
    def test_train(self):
        """Test training process."""
        policy = PolicyNetwork(state_dim=4, action_dim=2)
        discriminator = DiscriminatorNetwork(state_dim=4, action_dim=2)
        gail = GAIL(policy=policy, discriminator=discriminator, num_epochs=2)
        
        # Create dummy data
        expert_states = torch.randn(50, 4)
        expert_actions = torch.randint(0, 2, (50,))
        policy_states = torch.randn(50, 4)
        policy_actions = torch.randint(0, 2, (50,))
        
        # Train
        metrics = gail.train(expert_states, expert_actions, policy_states, policy_actions)
        
        assert "final_policy_loss" in metrics
        assert "final_discriminator_loss" in metrics
        assert isinstance(metrics["final_policy_loss"], float)


class TestValueNetwork:
    """Test ValueNetwork class."""
    
    def test_initialization(self):
        """Test network initialization."""
        value_net = ValueNetwork(state_dim=4)
        assert value_net.state_dim == 4
    
    def test_forward(self):
        """Test forward pass."""
        value_net = ValueNetwork(state_dim=4)
        state = torch.randn(1, 4)
        output = value_net(state)
        assert output.shape == (1, 1)


class TestAIRL:
    """Test AIRL class."""
    
    def test_initialization(self):
        """Test algorithm initialization."""
        policy = PolicyNetwork(state_dim=4, action_dim=2)
        discriminator = DiscriminatorNetwork(state_dim=4, action_dim=2)
        value_network = ValueNetwork(state_dim=4)
        airl = AIRL(policy=policy, discriminator=discriminator, value_network=value_network, num_epochs=2)
        
        assert airl.policy == policy
        assert airl.discriminator == discriminator
        assert airl.value_network == value_network
    
    def test_train(self):
        """Test training process."""
        policy = PolicyNetwork(state_dim=4, action_dim=2)
        discriminator = DiscriminatorNetwork(state_dim=4, action_dim=2)
        value_network = ValueNetwork(state_dim=4)
        airl = AIRL(policy=policy, discriminator=discriminator, value_network=value_network, num_epochs=2)
        
        # Create dummy data
        expert_states = torch.randn(50, 4)
        expert_actions = torch.randint(0, 2, (50,))
        expert_next_states = torch.randn(50, 4)
        policy_states = torch.randn(50, 4)
        policy_actions = torch.randint(0, 2, (50,))
        policy_next_states = torch.randn(50, 4)
        
        # Train
        metrics = airl.train(
            expert_states, expert_actions, expert_next_states,
            policy_states, policy_actions, policy_next_states
        )
        
        assert "final_policy_loss" in metrics
        assert "final_discriminator_loss" in metrics
        assert "final_value_loss" in metrics


class TestImitationEvaluator:
    """Test ImitationEvaluator class."""
    
    def test_initialization(self):
        """Test evaluator initialization."""
        env = gym.make("CartPole-v1")
        evaluator = ImitationEvaluator(env)
        assert evaluator.env == env
    
    def test_evaluate_policy(self):
        """Test policy evaluation."""
        env = gym.make("CartPole-v1")
        evaluator = ImitationEvaluator(env)
        
        # Create a simple policy
        def simple_policy(state, deterministic=True):
            return 0  # Always take action 0
        
        metrics = evaluator.evaluate_policy(simple_policy, num_episodes=5)
        
        assert "mean_reward" in metrics
        assert "std_reward" in metrics
        assert "success_rate" in metrics
        assert "mean_length" in metrics
        assert isinstance(metrics["mean_reward"], float)


class TestUtilityFunctions:
    """Test utility functions."""
    
    def test_set_seed(self):
        """Test seed setting."""
        set_seed(42)
        # Test that random numbers are deterministic
        np.random.seed(42)
        val1 = np.random.random()
        set_seed(42)
        val2 = np.random.random()
        assert val1 == val2
    
    def test_get_device(self):
        """Test device detection."""
        device = get_device()
        assert isinstance(device, torch.device)
    
    def test_create_expert_policy(self):
        """Test expert policy creation."""
        expert_policy = create_expert_policy("CartPole-v1", seed=42)
        assert callable(expert_policy)
        
        # Test policy execution
        state = np.array([0.0, 0.0, 0.0, 0.0])
        action = expert_policy(state)
        assert isinstance(action, (int, np.integer))


class TestExpertDataCollector:
    """Test ExpertDataCollector class."""
    
    def test_initialization(self):
        """Test collector initialization."""
        env = gym.make("CartPole-v1")
        expert_policy = create_expert_policy("CartPole-v1")
        collector = ExpertDataCollector(env, expert_policy)
        
        assert collector.env == env
        assert collector.expert_policy == expert_policy
    
    def test_collect_demonstrations(self):
        """Test demonstration collection."""
        env = gym.make("CartPole-v1")
        expert_policy = create_expert_policy("CartPole-v1")
        collector = ExpertDataCollector(env, expert_policy)
        
        demonstrations = collector.collect_demonstrations(num_episodes=2, max_steps_per_episode=10)
        
        assert "states" in demonstrations
        assert "actions" in demonstrations
        assert "rewards" in demonstrations
        assert "next_states" in demonstrations
        assert "dones" in demonstrations
        
        assert len(demonstrations["states"]) > 0
        assert len(demonstrations["actions"]) > 0


if __name__ == "__main__":
    pytest.main([__file__])
