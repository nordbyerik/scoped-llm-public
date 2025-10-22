#!/usr/bin/env python3
"""
Example script demonstrating how to use ElizaGPT.

This shows both interactive and programmatic usage of the Eliza chatbot.
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from elizagpt import Eliza


def programmatic_example():
    """Demonstrate programmatic usage of Eliza."""
    print("=" * 70)
    print("PROGRAMMATIC USAGE EXAMPLE")
    print("=" * 70)
    print()
    
    eliza = Eliza()
    
    # Example conversation
    conversations = [
        "Hello",
        "I feel worried about my future",
        "My mother never understood me",
        "I need help",
        "I want to be happy"
    ]
    
    for user_input in conversations:
        response = eliza.respond(user_input)
        print(f"You: {user_input}")
        print(f"Eliza: {response}")
        print()


def interactive_example():
    """Demonstrate interactive usage of Eliza."""
    print("\n" + "=" * 70)
    print("INTERACTIVE MODE")
    print("=" * 70)
    print("Starting interactive conversation...")
    print("(This will start the full Eliza experience)\n")
    
    eliza = Eliza()
    eliza.start_conversation()


if __name__ == "__main__":
    # Show programmatic example first
    programmatic_example()
    
    # Ask if user wants to try interactive mode
    print("Would you like to try interactive mode? (yes/no)")
    choice = input().strip().lower()
    
    if choice in ['yes', 'y']:
        interactive_example()
    else:
        print("\nThank you for trying ElizaGPT!")
