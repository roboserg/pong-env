from setuptools import setup, find_packages

setup(
    name="pong_env",  # Changed from pong_env to pong
    version="0.1.0",
    package_dir={"": "src"},  # Add this line to indicate src directory
    packages=find_packages(where="src"),  # Add where="src"
    install_requires=[
        "gymnasium",
        "stable-baselines3",
        "pygame-ce",
    ],
    python_requires=">=3.8",
)
