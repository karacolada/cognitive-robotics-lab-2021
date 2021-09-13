from setuptools import setup, find_packages
import os
import sys

SETUP_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(SETUP_DIR)

setup(name="cognitive-robotics-lab",
      version='0.0.1',
      packages=find_packages(),
      install_requires=[
            "gym",
            "matplotlib",
            "numpy",
            "opencv-python",
			"toml",
            "torch",
            "torchvision",
            "tqdm",
      ],
      )
