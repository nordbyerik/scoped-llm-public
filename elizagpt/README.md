# ElizaGPT

A faithful Python implementation of the classic ELIZA chatbot from the 1960s.

## About ELIZA

ELIZA was one of the first chatbot programs, created by Joseph Weizenbaum at MIT in 1964-1966. It simulated a Rogerian psychotherapist by using pattern matching and substitution methodology to create the illusion of understanding. Despite its simplicity, ELIZA demonstrated how a relatively simple program could appear to engage in meaningful conversation.

## Features

- **Classic Pattern Matching**: Uses regex patterns to recognize user input
- **Pronoun Reflection**: Transforms statements like "I am sad" to "How long have you been sad?"
- **Psychotherapy Simulation**: Responds in the style of a Rogerian psychotherapist
- **Interactive CLI**: Simple command-line interface for conversations
- **Pure Python**: No external dependencies required

## Installation

ElizaGPT is a standalone Python module. No additional packages are required beyond Python 3.10+.

```bash
cd elizagpt
python main.py
```

## Usage

### Interactive Mode

To start a conversation with Eliza:

```bash
python elizagpt/main.py
```

Or from within Python:

```python
from elizagpt import Eliza

eliza = Eliza()
eliza.start_conversation()
```

### Programmatic Usage

You can also use Eliza programmatically:

```python
from elizagpt import Eliza

eliza = Eliza()
response = eliza.respond("I feel sad today")
print(response)  # e.g., "Do you often feel sad today?"
```

## Example Conversation

```
You: Hello
Eliza: Hello... I'm glad you could drop by today.

You: I feel worried about my future
Eliza: Do you often feel worried about your future?

You: I think I need help
Eliza: Why do you need help?

You: My mother never understood me
Eliza: Tell me more about your mother.

You: quit
Eliza: Thank you for talking with me.
```

## How It Works

ELIZA operates using a simple but effective pattern-matching algorithm:

1. **Pattern Recognition**: User input is matched against a series of regex patterns
2. **Keyword Detection**: Special keywords like "mother", "father", "dream" trigger specific responses
3. **Pronoun Reflection**: Personal pronouns are swapped (I → you, my → your, etc.)
4. **Response Generation**: A random response from the matched pattern's response set is selected
5. **Contextual Substitution**: Captured phrases are reflected and inserted into the response

## Technical Details

### Pattern Priority

Patterns are matched in order, with more specific patterns listed first. This ensures that:
- Direct questions about needs, abilities, and feelings are caught early
- Family-related keywords trigger appropriate therapeutic responses
- Generic patterns serve as fallbacks

### Reflection Dictionary

The reflection mechanism transforms user statements by swapping:
- Pronouns: "I" ↔ "you", "my" ↔ "your"
- Verb forms: "am" ↔ "are", "was" ↔ "were"

This creates the illusion that Eliza understands and empathizes with the user.

## History and Impact

ELIZA was revolutionary for its time and had significant cultural impact:
- Demonstrated natural language processing concepts
- Showed how simple pattern matching could create compelling interactions
- Raised questions about AI, consciousness, and human-computer interaction
- Inspired countless chatbots and conversational AI systems

Despite its creator's intentions to demonstrate the superficiality of human-computer communication, many users developed emotional connections to ELIZA, illustrating the human tendency to anthropomorphize technology.

## Limitations

As a faithful recreation of the original, ElizaGPT inherits ELIZA's limitations:
- No actual understanding or intelligence
- Cannot maintain long-term context
- Responses are purely pattern-based
- May produce nonsensical responses with unusual input

## License

This implementation is provided for educational purposes, demonstrating classic AI techniques.

## References

- Weizenbaum, Joseph (1966). "ELIZA—a computer program for the study of natural language communication between man and machine". Communications of the ACM. 9 (1): 36–45.
- [Original ELIZA paper](https://web.stanford.edu/class/cs124/p36-weizenabaum.pdf)

## Author

Created as a demonstration of classic AI pattern-matching techniques, faithful to the original ELIZA design philosophy.
