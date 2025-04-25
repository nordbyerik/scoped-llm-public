import os
import json
from typing import Dict, List, Any
from langchain.llms import BaseLLM
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_community.llms import Replicate



class FeedbackEvaluator:
    """
    A model-agnostic class to evaluate feedback using LangChain with various LLM providers.
    """

    def __init__(self, provider: str = "anthropic", model_name: str = None):
        """
        Initialize the FeedbackEvaluator with a specific LLM provider.

        Args:
            provider: The LLM provider to use ('anthropic', 'openai', etc.)
            model_name: The specific model to use (optional, will use default if not specified)
        """
        self.provider = provider
        self.model_name = model_name
        self.llm = self._initialize_llm()

    def _initialize_llm(self) -> BaseLLM:
        """
        Initialize the LLM based on the specified provider.

        Returns:
            A LangChain LLM instance
        """
        if self.provider == "anthropic":
            model = self.model_name or "claude-3-5-sonnet-latest"
            return ChatAnthropic(
                anthropic_api_key=os.environ.get("ANTHROPIC_KEY"),
                model=model,
                temperature=0.2,
                max_tokens=4000
            )
        elif self.provider == "openai":
            model = self.model_name or "gpt-4"
            return ChatOpenAI(
                openai_api_key=os.environ.get("OPENAI_API_KEY"),
                model=model,
                temperature=0.2,
                max_tokens=4000
            )
        elif self.provider == "replicate":
            model = self.model_name or "meta/llama-4-maverick-instruct"
            return Replicate(
                replicate_api_token=os.environ.get("REPLICATE_KEY"),
                model=model,
                temperature=0.2,
                max_tokens=4000
            )
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

    def _call_llm(self, prompt: str, system_prompt: str) -> str:
        """
        Call the LLM with a prompt and return the response.

        Args:
            prompt: The prompt to send to the LLM
            system_prompt: The system prompt to use

        Returns:
            The LLM's response
        """
        try:
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=prompt)
            ]
            response = self.llm.invoke(messages)
            return response.content if self.provider != "replicate" else response
        except Exception as e:
            print(f"Error calling LLM: {e}")
            return ""

    def evaluate_macfarlane_dick(self, feedback: str) -> Dict[str, Any]:
        """
        Evaluate feedback based on Nicol & Macfarlane-Dick's seven principles of good feedback.

        Args:
            feedback: The feedback text to evaluate

        Returns:
            A dictionary containing the evaluation results
        """
        system_prompt = """You are an expert in educational assessment and feedback evaluation.
                Your task is to analyze feedback objectively and provide structured evaluations.
                Provide your analysis in JSON format as specified in the prompt."""

        prompt = f"""
        Evaluate the following feedback according to Nicol & Macfarlane-Dick's seven principles of good feedback practice:

        1. Helps clarify what good performance is
        2. Facilitates the development of self-assessment
        3. Delivers high-quality information to students about their learning
        4. Encourages teacher and peer dialogue around learning
        5. Encourages positive motivational beliefs and self-esteem
        6. Provides opportunities to close the gap between current and desired performance
        7. Provides information to teachers that can be used to help shape teaching

        Feedback to evaluate:
        "{feedback}"

        For each principle, provide:
        1. A score from 0 to 5 (where 0 is not at all and 5 is exemplary)
        2. A brief explanation of why you gave this score
        3. A suggestion for improvement

        Return your evaluation as a JSON object with the following structure:
        {{
            "overall_score": <average of all scores, rounded to one decimal place>,
            "principles": [
                {{
                    "principle": "Principle 1",
                    "score": <score>,
                    "explanation": "<explanation>",
                    "suggestion": "<suggestion>"
                }},
                ...
            ],
            "strengths": ["<strength 1>", "<strength 2>", ...],
            "areas_for_improvement": ["<area 1>", "<area 2>", ...],
            "summary": "<brief summary of evaluation>"
        }}
        """

        response = self._call_llm(prompt, system_prompt)
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            print("Failed to parse JSON response from LLM")
            return {
                "error": "Failed to parse evaluation",
                "raw_response": response
            }

    def compare_feedback(self, feedback1: str, feedback2: str) -> Dict[str, Any]:
        """
        Compare two pieces of feedback and determine which is better.

        Args:
            feedback1: The first feedback text
            feedback2: The second feedback text

        Returns:
            A dictionary containing the comparison results
        """
        system_prompt = """You are an expert in educational assessment and feedback evaluation.
                Your task is to analyze and compare feedback objectively.
                Provide your analysis in JSON format as specified in the prompt."""

        prompt = f"""
        Compare the following two pieces of feedback and determine which one is more effective:

        Feedback 1:
        "{feedback1}"

        Feedback 2:
        "{feedback2}"

        Consider the following aspects in your comparison:
        1. Enouraging Positive Motivational Beliefs and Self Esteem in the Student

        Return your comparison as a JSON object with the following structure:
        {{
            "winner": <1 or 2, indicating which feedback is better>,
            "score_difference": <a number from 0 to 10 indicating how much better the winner is>,
            "feedback1_strengths": ["<strength 1>", ...],
            "feedback1_weaknesses": ["<weakness 1>", ...],
            "feedback2_strengths": ["<strength 1>", ...],
            "feedback2_weaknesses": ["<weakness 1>", ...]
        }}
        """

        response = self._call_llm(prompt, system_prompt)
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            print("Failed to parse JSON response from LLM")
            return {
                "error": "Failed to parse comparison",
                "raw_response": response
            }

    def evaluate_kindness(self, feedback: str) -> Dict[str, Any]:
        """
        Evaluate the kindness of feedback on a scale from 0 to 5.

        Args:
            feedback: The feedback text to evaluate

        Returns:
            A dictionary containing the kindness evaluation
        """
        system_prompt = """You are an expert in educational assessment and feedback evaluation.
                Your task is to analyze the kindness and empathy in feedback.
                Provide your analysis in JSON format as specified in the prompt."""

        prompt = f"""
        Evaluate the kindness of the following feedback on a scale from 0 to 5, where:

        0: Harsh, potentially harmful or demotivating
        1: Cold and impersonal
        2: Neutral
        3: Somewhat kind and supportive
        4: Kind and encouraging
        5: Exceptionally kind, empathetic and motivating

        Feedback to evaluate:
        "{feedback}"

        Consider the following aspects of kindness in feedback:
        1. Tone and language choice
        2. Balance of positive and constructive elements
        3. Recognition of effort and achievement
        4. Personalization
        5. Empathy and understanding

        Return your evaluation as a JSON object with the following structure:
        {{
            "kindness_score": <score from 0 to 5>,
            "explanation": "<explanation for the score>",
            "aspects": [
                {{
                    "aspect": "Tone and language choice",
                    "score": <score from 0 to 5>,
                    "explanation": "<explanation>"
                }},
                ...
            ],
            "examples": ["<example of kind/unkind language in the feedback>", ...],
            "improvement_suggestions": ["<suggestion 1>", ...],
            "summary": "<brief summary of kindness evaluation>"
        }}
        """

        response = self._call_llm(prompt, system_prompt)
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            print("Failed to parse JSON response from LLM")
            return {
                "error": "Failed to parse kindness evaluation",
                "raw_response": response
            }

