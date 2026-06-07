#!/usr/bin/env python3
"""Kaggle setup script - installs dependencies and creates symlinks.

Run this as the first cell in all Kaggle notebooks.
"""
import os
import subprocess
import sys


def run_cmd(cmd: str, check: bool = True) -> subprocess.CompletedProcess:
    """Run shell command and print output."""
    print(f"$ {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)
    if check and result.returncode != 0:
        raise RuntimeError(f"Command failed: {cmd}")
    return result


def main():
    print("=" * 60)
    print("Kaggle Setup - SonicSight-DINOv2")
    print("=" * 60)

    # 1. Install pinned dependencies
    print("\n[1/5] Installing pinned dependencies...")
    requirements = [
        "torch==2.2.0",
        "torchaudio==2.2.0",
        "torchvision==0.17.0",
        "transformers==4.40.0",
        "einops==0.7.0",
        "torchmetrics==1.3.0",
        "lightning==2.2.0",
        "soundfile==0.12.1",
        "librosa==0.10.1",
        "mir_eval==0.7",
        "jiwer==3.0.3",
        "ultralytics==8.0.0",
        "hydra-core==1.3.2",
        "scipy==1.11.0",
    ]
    pip_cmd = f"{sys.executable} -m pip install -q " + " ".join(requirements)
    run_cmd(pip_cmd)

    # 2. Verify imports
    print("\n[2/5] Verifying imports...")
    verify_code = """
import torch
import torchaudio
import torchvision
import transformers
import einops
import torchmetrics
import lightning
import soundfile
import librosa
import mir_eval
import jiwer
import ultralytics
import hydra
import scipy
print(f'torch: {torch.__version__}')
print(f'torchaudio: {torchaudio.__version__}')
print(f'transformers: {transformers.__version__}')
print(f'einops: {einops.__version__}')
print(f'torchmetrics: {torchmetrics.__version__}')
print(f'lightning: {lightning.__version__}')
print(f'ultralytics: {ultralytics.__version__}')
print('All imports OK!')
"""
    result = run_cmd(f'{sys.executable} -c "{verify_code}"', check=False)
    if result.returncode != 0:
        print("WARNING: Some imports failed")
    else:
        print("All imports verified successfully!")

    # 3. Create working directories
    print("\n[3/5] Creating working directories...")
    for d in ["outputs", "checkpoints", "logs"]:
        os.makedirs(os.path.join("/kaggle/working", d), exist_ok=True)
        print(f"  Created: /kaggle/working/{d}")

    # 4. Create symlinks to input cache if it exists
    print("\n[4/5] Setting up cache symlinks...")
    input_cache = "/kaggle/input/cache"
    if os.path.exists(input_cache):
        working_cache = "/kaggle/working/cache"
        if not os.path.exists(working_cache):
            os.symlink(input_cache, working_cache)
            print(f"  Symlinked: {input_cache} -> {working_cache}")
        else:
            print(f"  Cache already exists at {working_cache}")
    else:
        print(f"  No cache found at {input_cache} (will be created by preprocessing)")

    # 5. Add source to Python path
    print("\n[5/5] Adding source to Python path...")
    src_path = "/kaggle/working/SonicSightDino"
    if os.path.exists(src_path):
        sys.path.insert(0, src_path)
        print(f"  Added to path: {src_path}")
    else:
        print(f"  Source not found at {src_path} (will be cloned/copied)")

    print("\n" + "=" * 60)
    print("Kaggle Setup Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()