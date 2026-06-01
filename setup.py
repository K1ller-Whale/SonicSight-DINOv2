"""SonicSightDino: DINOv2-Guided Audio-Visual Source Separation"""
from setuptools import setup, find_packages

setup(
    name="sonicsight-dino",
    version="0.1.0",
    description="DINOv2-guided audio-visual source separation",
    author="SonicSight Team",
    python_requires=">=3.10",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "torch>=2.0.0",
        "torchaudio>=2.0.0",
        "torchmetrics>=1.4.0",
        "lightning>=2.4.0",
        "transformers>=4.41.0",
        "einops>=0.8.0",
        "hydra-core>=1.3.0",
        "omegaconf>=2.3.0",
        "librosa>=0.10.0",
        "soundfile>=0.12.0",
        "numpy>=1.24.0",
        "scipy>=1.10.0",
    ],
    extras_require={
        "dev": ["pytest>=8.0.0", "pytest-cov>=5.0.0"],
        "eval": ["jiwer>=3.0.0", "openai-whisper>=20240930"],
    },
)
