#!/usr/bin/env python3
"""
main.py
=======

Console entry point for the Contextual Chatbot.

Usage:
    python3 main.py
    python3 main.py --thread work-notes
    python3 main.py --message "Hello there!"

Run `python3 main.py --help` for all options.
"""

import sys

from chatbot.cli import main

if __name__ == "__main__":
    sys.exit(main())
