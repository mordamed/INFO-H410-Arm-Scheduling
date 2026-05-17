from setuptools import setup, find_packages

setup(
    name="arm_scheduler",
    version="1.0.0",
    description="ARM32 Instruction Scheduler for Masked Cryptography (INFO-H410)",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "python-constraint>=1.4.0",
        "numpy>=1.24",
        "matplotlib>=3.7",
        "pandas>=2.0",
        "tqdm>=4.65",
    ],
    entry_points={
        "console_scripts": [
            "arm-scheduler=experiments.run_all:main",
        ]
    },
)
