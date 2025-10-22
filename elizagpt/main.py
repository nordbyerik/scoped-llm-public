#!/usr/bin/env python3
"""
ElizaGPT - Command-line interface for the Eliza chatbot.

This script provides an interactive terminal interface to chat with Eliza,
the classic psychotherapy simulation chatbot.
"""

import sys
from elizagpt import Eliza


def main():
    """Main entry point for the ElizaGPT CLI."""
    eliza = Eliza()
    eliza.start_conversation()
    return 0


if __name__ == "__main__":
    sys.exit(main())
