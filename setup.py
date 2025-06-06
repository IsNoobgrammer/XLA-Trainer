from setuptools import setup, find_packages

setup(
    name="tpu-trainer",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=1.10.0",
        "numpy>=1.19.0",
        "tqdm>=4.62.0",
        "wandb>=0.12.0",
    ],
    python_requires=">=3.8",
)
