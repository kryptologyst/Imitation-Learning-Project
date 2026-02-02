# Imitation Learning Project

A comprehensive implementation of imitation learning algorithms for research and educational purposes. This project provides clean, well-documented implementations of Behavioral Cloning (BC), Generative Adversarial Imitation Learning (GAIL), and Adversarial Inverse Reinforcement Learning (AIRL).

## ⚠️ Important Disclaimer

**This implementation is for research and educational purposes only. It is NOT intended for production control of real-world systems.**

## Features

- **Multiple Algorithms**: Behavioral Cloning, GAIL, and AIRL implementations
- **Modern Stack**: PyTorch 2.x, Gymnasium, comprehensive type hints
- **Reproducible**: Deterministic seeding and configuration management
- **Comprehensive Evaluation**: Statistical analysis with confidence intervals
- **Interactive Demo**: Streamlit-based visualization and experimentation
- **Production Ready**: Clean code structure, testing, and documentation

## Installation

### Prerequisites

- Python 3.10+
- PyTorch 2.0+
- CUDA (optional, for GPU acceleration)
- MPS (optional, for Apple Silicon)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/kryptologyst/Imitation-Learning-Project.git
cd Imitation-Learning-Project
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Install the package in development mode:
```bash
pip install -e .
```

## Quick Start

### Command Line Training

Train a Behavioral Cloning model on CartPole:

```bash
python scripts/train.py --env CartPole-v1 --algorithm bc --num_episodes 100 --num_epochs 50
```

Train GAIL on MountainCar:

```bash
python scripts/train.py --env MountainCar-v0 --algorithm gail --num_episodes 200 --num_epochs 100
```

### Interactive Demo

Launch the Streamlit demo:

```bash
streamlit run demo/app.py
```

The demo provides an interactive interface to:
- Select environments and algorithms
- Adjust training parameters
- Visualize training progress and results
- Compare different policies

### Python API

```python
import torch
import gymnasium as gym
from imitation_learning import (
    BehavioralCloning, PolicyNetwork,
    ExpertDataCollector, ImitationEvaluator,
    set_seed, create_expert_policy
)

# Set random seed for reproducibility
set_seed(42)

# Create environment
env = gym.make("CartPole-v1")

# Create expert policy
expert_policy = create_expert_policy("CartPole-v1", seed=42)

# Collect expert demonstrations
collector = ExpertDataCollector(env, expert_policy)
demonstrations = collector.collect_demonstrations(num_episodes=100)

# Create policy network
policy = PolicyNetwork(
    state_dim=env.observation_space.shape[0],
    action_dim=env.action_space.n,
    hidden_dims=(64, 64)
)

# Train behavioral cloning
bc = BehavioralCloning(
    policy=policy,
    learning_rate=1e-3,
    batch_size=64,
    num_epochs=100
)

expert_states = torch.tensor(demonstrations["states"], dtype=torch.float32)
expert_actions = torch.tensor(demonstrations["actions"], dtype=torch.long)

metrics = bc.train(expert_states, expert_actions)

# Evaluate policy
evaluator = ImitationEvaluator(env)
eval_metrics = evaluator.evaluate_policy(bc, num_episodes=50)

print(f"Mean reward: {eval_metrics['mean_reward']:.2f}")
print(f"Success rate: {eval_metrics['success_rate']:.1%}")
```

## Algorithms

### Behavioral Cloning (BC)

Supervised learning approach that directly learns to predict expert actions given states.

**Key Features:**
- Simple and stable training
- Fast convergence on simple tasks
- No interaction with environment during training

**Use Cases:**
- Simple environments with sufficient expert data
- Baseline for comparison with other methods
- Quick prototyping and experimentation

### Generative Adversarial Imitation Learning (GAIL)

Adversarial approach where a discriminator distinguishes between expert and policy-generated trajectories.

**Key Features:**
- Handles distribution shift better than BC
- Can learn from limited expert data
- More robust to expert policy imperfections

**Use Cases:**
- Complex environments with limited expert data
- When expert policy is not perfect
- Scenarios requiring robust policy learning

### Adversarial Inverse Reinforcement Learning (AIRL)

Extension of GAIL that learns both a reward function and a policy.

**Key Features:**
- Learns interpretable reward functions
- Better transfer to new environments
- More sample efficient than GAIL

**Use Cases:**
- Transfer learning scenarios
- When reward function interpretation is important
- Multi-task learning environments

## Project Structure

```
imitation-learning-project/
├── src/
│   └── imitation_learning/
│       ├── algorithms/          # Algorithm implementations
│       ├── models/              # Neural network architectures
│       ├── envs/                # Environment wrappers and utilities
│       ├── evaluation/          # Evaluation metrics and tools
│       └── utils/               # Utility functions
├── configs/                     # Configuration files
├── scripts/                     # Training and evaluation scripts
├── demo/                        # Streamlit demo application
├── tests/                       # Unit tests
├── notebooks/                   # Jupyter notebooks for experimentation
├── assets/                      # Generated plots and visualizations
└── data/                        # Expert demonstrations and datasets
```

## Configuration

The project uses OmegaConf for configuration management. Create custom configurations by modifying `configs/default.yaml`:

```yaml
# Environment settings
env:
  name: "CartPole-v1"
  seed: 42

# Training settings
training:
  num_epochs: 100
  batch_size: 64
  learning_rate: 1e-3

# Algorithm-specific settings
algorithms:
  behavioral_cloning:
    learning_rate: 1e-3
    hidden_dims: [64, 64]
    dropout_rate: 0.1
```

## Evaluation Metrics

The project provides comprehensive evaluation metrics:

- **Performance Metrics**: Mean reward, success rate, episode length
- **Statistical Analysis**: Confidence intervals, standard deviation
- **Imitation Quality**: Action accuracy, state distribution similarity
- **Robustness**: Performance across multiple seeds and environments

## Environments

Supported environments include:

- **CartPole-v1**: Classic control problem
- **MountainCar-v0**: Sparse reward environment
- **Acrobot-v1**: Underactuated system
- **Custom Environments**: Easy to add new environments

## Development

### Running Tests

```bash
pytest tests/
```

### Code Formatting

```bash
black src/ scripts/ demo/
ruff check src/ scripts/ demo/
```

### Pre-commit Hooks

```bash
pre-commit install
pre-commit run --all-files
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass
6. Submit a pull request

## Citation

If you use this code in your research, please cite:

```bibtex
@software{imitation_learning_project,
  title={Modern Imitation Learning Implementation},
  author={Kryptologyst},
  year={2026},
  url={https://github.com/kryptologyst/Imitation-Learning-Project}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- OpenAI Gym/Gymnasium for environment interfaces
- PyTorch team for the deep learning framework
- Stable Baselines3 for algorithm inspiration
- The broader reinforcement learning community

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**: Reduce batch size or use CPU
2. **Slow Training**: Enable GPU acceleration or reduce model size
3. **Poor Performance**: Increase expert episodes or adjust hyperparameters

### Getting Help

- Check the documentation in the code
- Review the example notebooks
- Open an issue on GitHub
- Check the troubleshooting section in the docs

## Roadmap

- [ ] Add more environments (robotics, navigation)
- [ ] Implement additional algorithms (DAgger, MGAIL)
- [ ] Add multi-agent support
- [ ] Improve visualization tools
- [ ] Add distributed training support
- [ ] Create comprehensive tutorials
# Imitation-Learning-Project
