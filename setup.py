# setup.py
from setuptools import setup, find_packages

setup(
    name="garbage_truck_detection",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "ultralytics>=8.0.0",
        "numpy>=1.20.0",
        "opencv-python>=4.5.0",
        "supervision>=0.10.0",
        "torch>=2.0.0",
        "pyyaml>=6.0",
        "pandas>=1.3.0",
        "matplotlib>=3.4.0",
    ],
    author="Your Name",
    author_email="your.email@example.com",
    description="A system for detecting garbage trucks in videos",
    keywords="computer vision, object detection, yolo, garbage truck",
    url="https://github.com/yourusername/garbage_truck_detection",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)