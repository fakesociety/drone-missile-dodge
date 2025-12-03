#!/usr/bin/env python3
"""
Setup script to organize project files into proper structure.
Run this once to move assets and models to their folders.
"""

import os
import shutil

# Define file mappings
ASSETS = ['drone1.png', 'missile.png', 'Tehran_sky.jpg', 'khamn.png', 'blast.png']
MODELS = ['drone_sac_horizontal_final.zip']

def setup_project():
    """Organize project files into proper folder structure."""
    
    # Create directories if they don't exist
    os.makedirs('assets', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    # Move assets
    print("📁 Organizing assets...")
    for asset in ASSETS:
        if os.path.exists(asset):
            dest = os.path.join('assets', asset)
            if not os.path.exists(dest):
                shutil.copy(asset, dest)
                print(f"  ✅ Copied {asset} → assets/")
            else:
                print(f"  ⏭️  {asset} already in assets/")
        else:
            print(f"  ⚠️  {asset} not found")
    
    # Move models
    print("\n🤖 Organizing models...")
    for model in MODELS:
        if os.path.exists(model):
            dest = os.path.join('models', model)
            if not os.path.exists(dest):
                shutil.copy(model, dest)
                print(f"  ✅ Copied {model} → models/")
            else:
                print(f"  ⏭️  {model} already in models/")
        else:
            print(f"  ⚠️  {model} not found")
    
    print("\n✅ Project structure organized!")
    print("\nYou can now upload to GitHub:")
    print("  git init")
    print("  git add .")
    print("  git commit -m 'Initial commit: Drone Missile Dodge RL project'")
    print("  git remote add origin https://github.com/yourusername/drone-missile-dodge.git")
    print("  git push -u origin main")

if __name__ == '__main__':
    setup_project()
