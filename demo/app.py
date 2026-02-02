"""Interactive Streamlit demo for imitation learning.

This demo allows users to experiment with different imitation learning
algorithms and visualize their performance.
"""

import streamlit as st
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from imitation_learning import (
    BehavioralCloning, GAIL, AIRL,
    PolicyNetwork, DiscriminatorNetwork, ValueNetwork,
    ExpertDataCollector, ImitationEvaluator,
    set_seed, get_device, create_expert_policy
)


def main():
    """Main Streamlit app."""
    st.set_page_config(
        page_title="Imitation Learning Demo",
        page_icon="🤖",
        layout="wide"
    )
    
    st.title("🤖 Imitation Learning Demo")
    st.markdown("""
    This demo showcases different imitation learning algorithms including:
    - **Behavioral Cloning (BC)**: Supervised learning approach
    - **Generative Adversarial Imitation Learning (GAIL)**: Adversarial approach
    - **Adversarial Inverse Reinforcement Learning (AIRL)**: Reward learning approach
    """)
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("Configuration")
        
        # Environment selection
        env_name = st.selectbox(
            "Environment",
            ["CartPole-v1", "MountainCar-v0", "Acrobot-v1"],
            index=0
        )
        
        # Algorithm selection
        algorithm = st.selectbox(
            "Algorithm",
            ["Behavioral Cloning", "GAIL", "AIRL"],
            index=0
        )
        
        # Training parameters
        st.subheader("Training Parameters")
        num_expert_episodes = st.slider("Expert Episodes", 10, 200, 50)
        num_epochs = st.slider("Training Epochs", 10, 200, 50)
        batch_size = st.slider("Batch Size", 16, 128, 32)
        learning_rate = st.slider("Learning Rate", 1e-4, 1e-2, 1e-3, format="%.0e")
        
        # Evaluation parameters
        st.subheader("Evaluation Parameters")
        num_eval_episodes = st.slider("Evaluation Episodes", 10, 100, 20)
        render_episodes = st.checkbox("Render Episodes", value=False)
        
        # Random seed
        seed = st.number_input("Random Seed", value=42, min_value=0, max_value=1000)
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("Training Progress")
        
        if st.button("🚀 Start Training", type="primary"):
            with st.spinner("Training in progress..."):
                # Set random seed
                set_seed(seed)
                
                # Create environment
                env = gym.make(env_name)
                env.reset(seed=seed)
                
                # Create expert policy
                expert_policy = create_expert_policy(env_name, seed)
                
                # Collect expert data
                collector = ExpertDataCollector(env, expert_policy)
                demonstrations = collector.collect_demonstrations(
                    num_episodes=num_expert_episodes,
                    deterministic=True
                )
                
                # Create networks
                state_dim = env.observation_space.shape[0]
                action_dim = env.action_space.n
                
                policy = PolicyNetwork(
                    state_dim=state_dim,
                    action_dim=action_dim,
                    hidden_dims=(64, 64)
                )
                
                # Train algorithm
                if algorithm == "Behavioral Cloning":
                    bc = BehavioralCloning(
                        policy=policy,
                        learning_rate=learning_rate,
                        batch_size=batch_size,
                        num_epochs=num_epochs
                    )
                    
                    expert_states = torch.tensor(demonstrations["states"], dtype=torch.float32)
                    expert_actions = torch.tensor(demonstrations["actions"], dtype=torch.long)
                    
                    metrics = bc.train(expert_states, expert_actions)
                    trained_policy = bc
                    
                elif algorithm == "GAIL":
                    discriminator = DiscriminatorNetwork(
                        state_dim=state_dim,
                        action_dim=action_dim,
                        hidden_dims=(64, 64)
                    )
                    
                    gail = GAIL(
                        policy=policy,
                        discriminator=discriminator,
                        learning_rate_policy=learning_rate,
                        learning_rate_discriminator=learning_rate * 3,
                        batch_size=batch_size,
                        num_epochs=num_epochs
                    )
                    
                    expert_states = torch.tensor(demonstrations["states"], dtype=torch.float32)
                    expert_actions = torch.tensor(demonstrations["actions"], dtype=torch.long)
                    policy_states = expert_states.clone()
                    policy_actions = expert_actions.clone()
                    
                    metrics = gail.train(expert_states, expert_actions, policy_states, policy_actions)
                    trained_policy = gail
                    
                elif algorithm == "AIRL":
                    discriminator = DiscriminatorNetwork(
                        state_dim=state_dim,
                        action_dim=action_dim,
                        hidden_dims=(64, 64)
                    )
                    value_network = ValueNetwork(
                        state_dim=state_dim,
                        hidden_dims=(64, 64)
                    )
                    
                    airl = AIRL(
                        policy=policy,
                        discriminator=discriminator,
                        value_network=value_network,
                        learning_rate_policy=learning_rate,
                        learning_rate_discriminator=learning_rate * 3,
                        learning_rate_value=learning_rate,
                        batch_size=batch_size,
                        num_epochs=num_epochs
                    )
                    
                    expert_states = torch.tensor(demonstrations["states"], dtype=torch.float32)
                    expert_actions = torch.tensor(demonstrations["actions"], dtype=torch.long)
                    expert_next_states = torch.tensor(demonstrations["next_states"], dtype=torch.float32)
                    policy_states = expert_states.clone()
                    policy_actions = expert_actions.clone()
                    policy_next_states = expert_next_states.clone()
                    
                    metrics = airl.train(
                        expert_states, expert_actions, expert_next_states,
                        policy_states, policy_actions, policy_next_states
                    )
                    trained_policy = airl
                
                # Evaluate policy
                evaluator = ImitationEvaluator(env)
                eval_metrics = evaluator.evaluate_policy(
                    trained_policy,
                    num_episodes=num_eval_episodes,
                    deterministic=True,
                    render=render_episodes
                )
                
                # Store results in session state
                st.session_state.training_complete = True
                st.session_state.algorithm = algorithm
                st.session_state.metrics = metrics
                st.session_state.eval_metrics = eval_metrics
                st.session_state.trained_policy = trained_policy
                st.session_state.env = env
                
                st.success("Training completed successfully!")
    
    # Display results if training is complete
    if st.session_state.get("training_complete", False):
        with col1:
            st.header("Results")
            
            # Training metrics
            st.subheader("Training Metrics")
            metrics = st.session_state.metrics
            
            if "final_loss" in metrics:
                st.metric("Final Loss", f"{metrics['final_loss']:.4f}")
            elif "final_policy_loss" in metrics:
                st.metric("Final Policy Loss", f"{metrics['final_policy_loss']:.4f}")
                if "final_discriminator_loss" in metrics:
                    st.metric("Final Discriminator Loss", f"{metrics['final_discriminator_loss']:.4f}")
            
            # Evaluation metrics
            st.subheader("Evaluation Metrics")
            eval_metrics = st.session_state.eval_metrics
            
            col_metric1, col_metric2, col_metric3 = st.columns(3)
            
            with col_metric1:
                st.metric(
                    "Mean Reward",
                    f"{eval_metrics['mean_reward']:.2f}",
                    f"±{eval_metrics['std_reward']:.2f}"
                )
            
            with col_metric2:
                st.metric(
                    "Success Rate",
                    f"{eval_metrics['success_rate']:.1%}"
                )
            
            with col_metric3:
                st.metric(
                    "Mean Length",
                    f"{eval_metrics['mean_length']:.1f}"
                )
            
            # Confidence interval
            st.info(f"95% Confidence Interval: [{eval_metrics['ci_lower']:.2f}, {eval_metrics['ci_upper']:.2f}]")
        
        with col2:
            st.header("Visualization")
            
            # Reward distribution
            fig = go.Figure()
            fig.add_trace(go.Histogram(
                x=eval_metrics['episode_rewards'],
                name='Episode Rewards',
                opacity=0.7
            ))
            fig.update_layout(
                title="Reward Distribution",
                xaxis_title="Reward",
                yaxis_title="Frequency"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Episode length distribution
            fig2 = go.Figure()
            fig2.add_trace(go.Histogram(
                x=eval_metrics['episode_lengths'],
                name='Episode Lengths',
                opacity=0.7
            ))
            fig2.update_layout(
                title="Episode Length Distribution",
                xaxis_title="Length",
                yaxis_title="Frequency"
            )
            st.plotly_chart(fig2, use_container_width=True)
        
        # Policy comparison
        st.header("Policy Comparison")
        
        if st.button("🔄 Compare with Random Policy"):
            with st.spinner("Evaluating random policy..."):
                # Evaluate random policy
                def random_policy(state, deterministic=True):
                    return env.action_space.sample()
                
                random_eval = evaluator.evaluate_policy(
                    random_policy,
                    num_episodes=num_eval_episodes,
                    deterministic=False
                )
                
                # Create comparison chart
                algorithms = [st.session_state.algorithm, "Random Policy"]
                mean_rewards = [eval_metrics['mean_reward'], random_eval['mean_reward']]
                std_rewards = [eval_metrics['std_reward'], random_eval['std_reward']]
                
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=algorithms,
                    y=mean_rewards,
                    error_y=dict(type='data', array=std_rewards),
                    name='Mean Reward'
                ))
                fig.update_layout(
                    title="Policy Comparison",
                    xaxis_title="Algorithm",
                    yaxis_title="Mean Reward"
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Display comparison metrics
                col_comp1, col_comp2 = st.columns(2)
                
                with col_comp1:
                    st.metric(
                        f"{st.session_state.algorithm} Reward",
                        f"{eval_metrics['mean_reward']:.2f}",
                        f"±{eval_metrics['std_reward']:.2f}"
                    )
                
                with col_comp2:
                    st.metric(
                        "Random Policy Reward",
                        f"{random_eval['mean_reward']:.2f}",
                        f"±{random_eval['std_reward']:.2f}"
                    )
    
    # Footer
    st.markdown("---")
    st.markdown("""
    **Disclaimer**: This implementation is for research and educational purposes only.
    It is NOT intended for production control of real-world systems.
    """)


if __name__ == "__main__":
    main()
