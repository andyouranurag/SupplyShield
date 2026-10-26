"""
setup.py — installable package for MultimodalWarning (Person 4).

Install in editable mode:
    pip install -e .

Or as a regular install:
    pip install .
"""

from setuptools import setup, find_packages

setup(
    name="MultimodalWarning",
    version="1.0.0",
    author="Person 4",
    description="Multimodal Fusion & Early Warning library for supply-chain disruptions.",
    long_description=open("README.md").read() if __import__("pathlib").Path("README.md").exists() else "",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.24",
        "pandas>=2.0",
        "scikit-learn>=1.3",
        "pyyaml>=6.0",
    ],
    extras_require={
        "torch": ["torch>=2.0"],
        "dev":   ["pytest>=7.0", "pytest-cov"],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
