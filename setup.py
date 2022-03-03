from setuptools import find_packages, setup

setup(
    name="stancedetection",
    packages=find_packages("src", exclude=["*.tests", "*.tests.*", "tests.*", "tests"]),
    version="0.1.0",
    description="Few-Shot Cross-Lingual Stance Detection with Sentiment-Based Pre-Training",
    author="Anonymous",
    package_dir={"": "src"},
    entry_points={},
    include_package_data=True,
    license="",
)
