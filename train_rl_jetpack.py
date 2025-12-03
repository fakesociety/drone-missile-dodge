#!/usr/bin/env python3
"""
Drone Missile Dodge - Training Script

This script trains a SAC (Soft Actor-Critic) agent to control a drone
that must navigate through missile fields to reach a target.

Features:
    - GPU acceleration with CUDA support
    - Parallel environment training (vectorized)
    - Periodic checkpoint saving
    - Fine-tuning from existing models
    - Automatic evaluation after training

Author: [Your Name]
Date: 2025
License: MIT

Usage:
    # Train from scratch
    python train_rl_jetpack.py --device cuda --n-envs 8
    
    # Fine-tune existing model
    python train_rl_jetpack.py --finetune ./drone_sac_horizontal_final.zip --device cuda
    
    # Quick smoke test
    python train_rl_jetpack.py --smoke --device cuda
"""

import argparse
import os
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env
from drone_game_env import Drone2DEnv

# =============================================================================
# TRAINING CONFIGURATION
# =============================================================================

MODEL_NAME = "drone_sac_horizontal"
TOTAL_TIMESTEPS = 500_000      # Training steps (500K for fine-tuning, 2M for scratch)
MODEL_SAVE_PATH = f"./{MODEL_NAME}_final.zip"
MISSILE_COUNT = 50             # Number of missiles (difficulty level)
CHECKPOINT_FREQ = 100_000      # Save model every N steps


def train_clean(device_pref='auto', n_envs=4, load_model=None):
    """
    Train the drone agent using SAC algorithm.
    
    Args:
        device_pref (str): Device preference ('auto', 'cpu', 'cuda')
        n_envs (int): Number of parallel environments
        load_model (str): Path to existing model for fine-tuning
    
    Returns:
        None (saves model to disk)
    """
    # Determine device
    device = 'cuda' if (device_pref == 'cuda' or (device_pref == 'auto' and torch.cuda.is_available())) else 'cpu'
    print(f"🖥️  Training device: {device}")

    if device == 'cuda':
        torch.set_num_threads(1)  # Avoid CPU overhead when using GPU
    
    print(f"🔄 Using {n_envs} parallel environments")

    # Environment factory
    def make_env():
        def _init():
            return Drone2DEnv(render_mode=None, missile_count=MISSILE_COUNT, spawn_close=True)
        return _init

    vec_env = make_vec_env(make_env(), n_envs=n_envs)

    if load_model and os.path.exists(load_model):
        # Fine-tuning existing model
        print(f"Loading existing model from {load_model}")
        model = SAC.load(load_model, device=device)
        model.set_env(vec_env)
        model.learning_rate = 5e-5  # Very low LR for careful fine-tuning
        print(f"Fine-tuning with learning_rate={model.learning_rate}")
    else:
        # Training from scratch
        if os.path.exists(MODEL_SAVE_PATH):
            print(f"Deleting old model: {MODEL_SAVE_PATH}")
            os.remove(MODEL_SAVE_PATH)
        
        # Optimized SAC setup for 21D observation
        model = SAC(
            "MlpPolicy",
            vec_env,
            verbose=1,
            learning_rate=3e-4,
            buffer_size=500_000,  # Large buffer for 2.5M steps
            learning_starts=20_000,  # Start learning after seeing quality data
            batch_size=512,  # Larger batches for stable learning
            gamma=0.99,
            tau=0.005,
            ent_coef='auto',
            target_update_interval=1,  # Update target network every step
            train_freq=1,
            gradient_steps=1,
            policy_kwargs=dict(
                net_arch=[512, 512, 256],  # Deep network for complex control
                activation_fn=torch.nn.ReLU,  # Explicit activation
            ),
            device=device
        )

    print(f"Training for {TOTAL_TIMESTEPS} timesteps")
    
    # Train with periodic saves every 100K steps
    save_freq = 100_000
    for i in range(0, TOTAL_TIMESTEPS, save_freq):
        steps = min(save_freq, TOTAL_TIMESTEPS - i)
        model.learn(total_timesteps=steps, reset_num_timesteps=False)
        model.save(MODEL_SAVE_PATH)
        print(f"Checkpoint saved at {i+steps} steps: {MODEL_SAVE_PATH}")
    
    vec_env.close()
    print(f"Final model saved: {MODEL_SAVE_PATH}")

    # Quick eval
    print(f"\n{'='*50}")
    print(f"Running 20-episode evaluation")
    print(f"{'='*50}\n")
    eval_env = Drone2DEnv(render_mode=None, missile_count=MISSILE_COUNT, spawn_close=True)
    successes = 0
    failures = {'missile': 0, 'floor_ceiling': 0, 'other': 0}
    
    for ep in range(20):
        obs, info = eval_env.reset()
        terminated = False
        truncated = False
        while not terminated and not truncated:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = eval_env.step(action)
        
        reason = info.get('reason', '')
        if "Target Reached" in reason:
            successes += 1
            print(f"Episode {ep+1}: SUCCESS ✓")
        else:
            if "missile" in reason.lower():
                failures['missile'] += 1
                print(f"Episode {ep+1}: FAILED - Missile hit")
            elif "floor" in reason.lower() or "ceiling" in reason.lower():
                failures['floor_ceiling'] += 1
                print(f"Episode {ep+1}: FAILED - Floor/Ceiling crash")
            else:
                failures['other'] += 1
                print(f"Episode {ep+1}: FAILED - {reason}")
    
    eval_env.close()
    
    print(f"\n{'='*50}")
    print(f"EVALUATION RESULTS")
    print(f"{'='*50}")
    print(f"Success rate: {successes}/20 ({successes*5}%)")
    print(f"Failures:")
    print(f"  - Missile hits: {failures['missile']}")
    print(f"  - Floor/Ceiling: {failures['floor_ceiling']}")
    print(f"  - Other: {failures['other']}")
    print(f"{'='*50}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train clean horizontal-missile drone agent")
    parser.add_argument('--smoke', action='store_true', help='Short smoke test (20k steps)')
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='cuda')
    parser.add_argument('--n-envs', type=int, default=4)
    parser.add_argument('--finetune', type=str, help='Path to existing model to fine-tune')
    args = parser.parse_args()

    if args.smoke:
        print("Smoke test: 20k steps")
        TOTAL_TIMESTEPS = 20_000

    train_clean(device_pref=args.device, n_envs=args.n_envs, load_model=args.finetune)