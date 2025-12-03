# 🚁 Drone Missile Dodge - Reinforcement Learning Project

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![Stable-Baselines3](https://img.shields.io/badge/Stable--Baselines3-2.0+-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

**An AI-powered drone that learns to navigate through missile fields using Deep Reinforcement Learning**

[Features](#features) • [Installation](#installation) • [Usage](#usage) • [Architecture](#architecture) • [Results](#results)

</div>

---

## 🎯 Project Overview

This project implements a **2D drone simulation** where an AI agent learns to:
- Navigate from a starting position to a target
- Dodge multiple waves of incoming missiles (up to 50+)
- Control dual thrusters for movement and rotation
- Make real-time decisions based on sensor observations

The agent is trained using **SAC (Soft Actor-Critic)**, a state-of-the-art deep reinforcement learning algorithm.

---

## ✨ Features

- 🎮 **Custom Gymnasium Environment** - Fully compatible with OpenAI Gym standards
- 🧠 **Deep Neural Network Policy** - 512→512→256 architecture for complex decision making
- 🚀 **GPU Accelerated Training** - CUDA support for 3-5x faster training
- 🎯 **Multi-Wave Missile System** - Dynamic spawning of missile waves during gameplay
- 📊 **Real-time Visualization** - PyGame-based rendering with custom sprites
- 💾 **Checkpoint System** - Periodic model saving to prevent progress loss
- 🔧 **Fine-tuning Support** - Continue training from existing models

---

## 🛠️ Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (recommended)
- Git

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/drone-missile-dodge.git
cd drone-missile-dodge

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## 📁 Project Structure

```
drone-missile-dodge/
│
├── 📄 drone_game_env.py      # Custom Gymnasium environment
├── 📄 train_rl_jetpack.py    # Training script with SAC algorithm
├── 📄 render_test_missile.py # Visualization and testing script
│
├── 🖼️ assets/                 # Visual assets
│   ├── drone1.png            # Drone sprite
│   ├── missile.png           # Missile sprite
│   ├── Tehran_sky.jpg        # Background image
│   ├── khamn.png             # Target sprite
│   └── blast.png             # Explosion effect
│
├── 🤖 models/                 # Trained models
│   └── drone_sac_horizontal_final.zip
│
├── 📊 logs/                   # Training logs (TensorBoard)
│
├── 📄 requirements.txt        # Python dependencies
├── 📄 README.md              # This file
└── 📄 LICENSE                # MIT License
```

---

## 🚀 Usage

### Training from Scratch

```bash
# Full training (2M timesteps, ~30-45 minutes on GPU)
python train_rl_jetpack.py --device cuda --n-envs 8

# Quick smoke test (20K timesteps)
python train_rl_jetpack.py --smoke --device cuda
```

### Fine-tuning Existing Model

```bash
# Continue training from checkpoint (500K additional steps)
python train_rl_jetpack.py --finetune ./drone_sac_horizontal_final.zip --device cuda --n-envs 8
```

### Testing & Visualization

```bash
# Watch the trained agent in action
python render_test_missile.py --episodes 10 --missiles 50

# Test with different missile counts
python render_test_missile.py --episodes 5 --missiles 100
```

### Command Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--device` | Training device (auto/cpu/cuda) | `cuda` |
| `--n-envs` | Number of parallel environments | `4` |
| `--finetune` | Path to model for fine-tuning | `None` |
| `--smoke` | Quick test with 20K steps | `False` |
| `--episodes` | Number of test episodes | `5` |
| `--missiles` | Number of missiles in test | `10` |

---

## 🧠 Architecture

### Environment Design

| Component | Specification |
|-----------|---------------|
| **Observation Space** | 21-dimensional vector |
| **Action Space** | 2 continuous values [0, 1] |
| **Physics** | Realistic 2D with gravity, thrust, rotation |
| **Reward** | Progress-based + success bonus |

#### Observation Vector (21 dimensions)
```
[0-5]   Drone state: x, y, angle, vx, vy, angular_velocity
[6-7]   Target delta: dx, dy
[8]     Nearest missile distance
[9-20]  3 nearest missiles: [dx, dy, vx, vy] × 3
```

#### Action Space
```
[0] Left thruster power  (0.0 - 1.0)
[1] Right thruster power (0.0 - 1.0)
```

### Neural Network Architecture

```
Input Layer (21 neurons)
    ↓
Hidden Layer 1 (512 neurons, ReLU)
    ↓
Hidden Layer 2 (512 neurons, ReLU)
    ↓
Hidden Layer 3 (256 neurons, ReLU)
    ↓
Output Layer (2 neurons - Actor)
```

### SAC Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| Learning Rate | 3e-4 (train) / 5e-5 (finetune) | Step size for optimization |
| Buffer Size | 500,000 | Experience replay capacity |
| Batch Size | 512 | Samples per update |
| Gamma | 0.99 | Discount factor |
| Tau | 0.005 | Target network update rate |
| Learning Starts | 20,000 | Steps before learning begins |

---

## 📊 Results

### Performance Metrics

| Missiles | Success Rate | Training Time |
|----------|--------------|---------------|
| 10 | 100% (20/20) | ~10 min |
| 25 | 95% (19/20) | ~20 min |
| 50 | 90% (18/20) | ~30 min |

### Training Curves

The agent typically achieves:
- **Episode reward > 2000** after 500K steps
- **Episode length ~300-400 steps** (efficient path finding)
- **Consistent success** after 1M steps

---

## 🎮 How It Works

### 1. Environment Dynamics

The drone operates in a 40m × 30m world with:
- **Gravity**: -9.81 m/s²
- **Mass**: 0.7 kg
- **Max Thrust**: 25N per thruster
- **Air Resistance**: 0.92 damping factor

### 2. Missile System

- **Initial Wave**: Layered pattern with guaranteed gap
- **Dynamic Waves**: 4-6 missiles every ~1.6 seconds
- **Speed**: 2.5 m/s horizontal movement
- **Total Waves**: Up to 5 waves per episode

### 3. Reward Function

```python
# Progress reward (main driving force)
reward += (best_distance - current_distance) * 15.0

# Success bonus
if distance_to_target < 2.0:
    reward += 300.0

# Hesitation penalty (near target but not reaching)
elif distance_to_target < 3.0:
    reward -= 2.0

# Collision penalties
missile_hit: -200.0
floor/ceiling: -100.0
```

---

## 🔧 Configuration

### Modifying Difficulty

In `drone_game_env.py`:
```python
MISSILE_SPEED_M = 2.5      # Increase for harder gameplay
MISSILE_WIDTH_M = 1.0      # Decrease for smaller hitbox
```

In `train_rl_jetpack.py`:
```python
MISSILE_COUNT = 50         # Adjust initial missile count
TOTAL_TIMESTEPS = 500_000  # Adjust training duration
```

### Custom Training

```python
from drone_game_env import Drone2DEnv
from stable_baselines3 import SAC

# Create environment
env = Drone2DEnv(render_mode=None, missile_count=30)

# Train custom model
model = SAC("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=1_000_000)
model.save("my_custom_model")
```

---

## 📈 Monitoring Training

### TensorBoard

```bash
tensorboard --logdir ./logs
```

Then open `http://localhost:6006` in your browser.

### Key Metrics to Watch
- `rollout/ep_rew_mean`: Average episode reward (should increase)
- `rollout/ep_len_mean`: Episode length (should stabilize around 300-400)
- `train/actor_loss`: Actor network loss (should decrease)
- `train/critic_loss`: Critic network loss (should stabilize)

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3) - RL algorithms
- [Gymnasium](https://gymnasium.farama.org/) - Environment API
- [PyGame](https://www.pygame.org/) - Visualization

---

## 📧 Contact

Your Name - your.email@example.com

Project Link: [https://github.com/yourusername/drone-missile-dodge](https://github.com/yourusername/drone-missile-dodge)

---

<div align="center">
Made with ❤️ and 🤖 by [Your Name]
</div>
