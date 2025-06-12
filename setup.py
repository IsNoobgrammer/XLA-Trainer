from setuptools import setup, find_packages

setup(
    name="tpu-trainer",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=1.10.0",
        "torch-xla>=1.13.0",
        "numpy>=1.19.0",
        "tqdm>=4.62.0",
        "wandb>=0.12.0",
        "transformers>=4.20.0",
        "datasets>=2.0.0",
    ],
    python_requires=">=3.8",
    description="XLA-optimized training utilities for TPU and GPU",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="TPU-Trainer Team",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
)
