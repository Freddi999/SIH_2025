import sys
import os

# Ensure src is in path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from main import start_training


if __name__ == "__main__":
    start_training()
