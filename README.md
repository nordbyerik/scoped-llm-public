# To Run
For solid performance, try to run this on A series nvidia gpus

Run the commands in the setup.sh file.

Create a local .env file which contains
- WANDB_API_KEY
- HUGGINGFACE_TOKEN
- ANTHROPIC_KEY
- REPLICATE_KEY

Run main.py

# Main Files Which Are Relevant
utils/activation_utils/steering_layer.py <- Allows for activations to be collected and/or steered

utils/dataset_utils <- Contains code to load in datasets

utils/evaluation_utils <- Contains code to evaluate models

llm_controllers <- Contains various classes for different types of steerers & scopers

folder_of_shame <- Contains previous experiments

# Main TODOs:
- Add in additional dataset loading & their respective evals
- Clean up code (i.e. use **kwargs instead of listing out full parameters)
- Add in more scoping methods & move the hackathon code into this framework