#!/usr/bin/env python3
"""
Drone Missile Dodge - Visualization & Testing Script

This script loads a trained model and runs it in visual mode,
allowing you to watch the AI agent navigate through missile fields.

Features:
    - Real-time PyGame visualization
    - Configurable missile count and episodes
    - Blast effect display on success
    - Success/failure statistics

Author: [Your Name]
Date: 2025
License: MIT

Usage:
    # Basic test with 10 missiles
    python render_test_missile.py --episodes 5 --missiles 10
    
    # Extreme challenge with 50 missiles
    python render_test_missile.py --episodes 10 --missiles 50
    
    # Slow motion for analysis
    python render_test_missile.py --frame-delay 0.05
"""

import argparse
import time
import os
import torch
from stable_baselines3 import SAC
from drone_game_env import Drone2DEnv


def main():
    """Main function to run visualization."""
    parser = argparse.ArgumentParser(
        description='Visualize trained drone agent navigating through missiles.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python render_test_missile.py --episodes 5 --missiles 20
    python render_test_missile.py --model ./my_model.zip --missiles 50
        """
    )
    parser.add_argument('--model', type=str, default='./drone_sac_horizontal_final.zip',
                        help='Path to trained model (default: ./drone_sac_horizontal_final.zip)')
    parser.add_argument('--episodes', type=int, default=5,
                        help='Number of episodes to run (default: 5)')
    parser.add_argument('--missiles', type=int, default=10,
                        help='Number of missiles in environment (default: 10)')
    parser.add_argument('--frame-delay', type=float, default=0.0,
                        help='Seconds between frames for slow-mo (default: 0)')
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto',
                        help='Device for model inference (default: auto)')
    args = parser.parse_args()

    # Validate model exists
    if not os.path.exists(args.model):
        print(f'❌ Model not found: {args.model}')
        raise SystemExit(1)

    # Resolve device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    elif args.device == 'cuda':
        if not torch.cuda.is_available():
            raise SystemExit('❌ CUDA requested but not available')
        device = 'cuda'
    else:
        device = 'cpu'

    print(f'🖥️  Loading model on device: {device}')
    model = SAC.load(args.model, device=device)

    print(f'🎮 Creating environment with {args.missiles} missiles...')
    env = Drone2DEnv(render_mode='human', missile_count=args.missiles, spawn_close=True)

    # Statistics
    successes = 0
    failures = 0

    for ep in range(args.episodes):
        obs, info = env.reset()
        terminated = False
        truncated = False
        
        while not terminated and not truncated:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            if args.frame_delay > 0:
                time.sleep(args.frame_delay)
        
        # Show blast effect on success
        if env.show_blast:
            successes += 1
            for _ in range(45):  # Show blast for ~1.5 seconds
                env.render()
                time.sleep(1 / 30.0)
            print(f'✅ Episode {ep+1}/{args.episodes}: SUCCESS - Target reached!')
        else:
            failures += 1
            print(f'❌ Episode {ep+1}/{args.episodes}: FAILED - {info.get("reason", "Unknown")}')

    env.close()
    
    # Print summary
    print(f'\n{"="*50}')
    print(f'📊 RESULTS SUMMARY')
    print(f'{"="*50}')
    print(f'✅ Successes: {successes}/{args.episodes} ({successes/args.episodes*100:.1f}%)')
    print(f'❌ Failures:  {failures}/{args.episodes}')
    print(f'{"="*50}')


if __name__ == '__main__':
    main()
