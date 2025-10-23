from setuptools import setup, find_packages

setup(
    name="graft",
    version="0.1.0",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "transformers>=4.30.0",
        "accelerate>=0.20.0",
        "datasets>=2.16.0,<4.0.0",
        "torch-geometric>=2.3.0",
        "pyyaml>=6.0",
        "numpy>=1.24.0,<2.0.0",
        "scikit-learn>=1.2.0",
        "tqdm>=4.65.0",
        "wandb>=0.15.0",
        "colorama",
        "matplotlib",
    ],
    extras_require={
        "dev": ["pytest>=7.3.0", "black>=23.3.0", "isort>=5.12.0"],
    },
)
