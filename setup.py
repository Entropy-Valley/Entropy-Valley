from setuptools import setup, find_packages
from pathlib import Path

setup(
    name="ladit",
    version="0.1.0",
    description="Length-Adaptive Decoding for Masked Diffusion Machine Translation",
    long_description=Path("README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    packages=find_packages(include=["ladit", "ladit.*"]),
    install_requires=Path("requirements.txt").read_text().splitlines(),
    python_requires=">=3.9",
    license="MIT",
)
