from setuptools import setup, find_packages

# Read dependencies from requirements.txt
with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="poker_project",
    version="0.1",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=requirements,  # Use requirements.txt for dependencies
)
