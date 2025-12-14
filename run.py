#!/usr/bin/env python
"""Launcher script for the Image Processing application."""
import sys
import os

# Add the project root to the path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Now import and run
from src.gui import run_app

if __name__ == "__main__":
    print("Starting Image Processing Suite...")
    print("If the window doesn't appear, check for error messages below.")
    try:
        run_app()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        input("Press Enter to exit...")

