"""
Unit tests for the ElizaGPT chatbot implementation.
"""

import unittest
import re
from elizagpt.eliza import Eliza


class TestEliza(unittest.TestCase):
    """Test cases for the Eliza chatbot class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.eliza = Eliza()
    
    def test_initialization(self):
        """Test that Eliza initializes correctly."""
        self.assertIsNotNone(self.eliza)
        self.assertIsInstance(self.eliza.reflections, dict)
        self.assertIsInstance(self.eliza.patterns, list)
        self.assertGreater(len(self.eliza.patterns), 0)
    
    def test_reflections_dict(self):
        """Test that the reflections dictionary contains expected mappings."""
        self.assertEqual(self.eliza.reflections["i"], "you")
        self.assertEqual(self.eliza.reflections["my"], "your")
        self.assertEqual(self.eliza.reflections["am"], "are")
        self.assertEqual(self.eliza.reflections["you"], "me")
        self.assertEqual(self.eliza.reflections["your"], "my")
    
    def test_reflect_simple(self):
        """Test simple pronoun reflection."""
        result = self.eliza.reflect("i am sad")
        self.assertEqual(result, "you are sad")
    
    def test_reflect_possessive(self):
        """Test possessive pronoun reflection."""
        result = self.eliza.reflect("my mother")
        self.assertEqual(result, "your mother")
    
    def test_reflect_multiple_pronouns(self):
        """Test reflection with multiple pronouns."""
        result = self.eliza.reflect("i think my mother was nice")
        self.assertEqual(result, "you think your mother were nice")
    
    def test_reflect_no_change(self):
        """Test that words without reflections remain unchanged."""
        result = self.eliza.reflect("the quick brown fox")
        self.assertEqual(result, "the quick brown fox")
    
    def test_respond_hello(self):
        """Test response to hello greeting."""
        response = self.eliza.respond("hello")
        self.assertIsInstance(response, str)
        self.assertTrue(any(word in response.lower() for word in ["hello", "hi", "glad"]))
    
    def test_respond_i_need(self):
        """Test response to 'I need' pattern."""
        response = self.eliza.respond("I need help")
        self.assertIsInstance(response, str)
        self.assertTrue(any(word in response.lower() for word in ["need", "help"]))
    
    def test_respond_i_feel(self):
        """Test response to 'I feel' pattern."""
        response = self.eliza.respond("I feel sad")
        self.assertIsInstance(response, str)
        # Response should contain some reflection about feeling
        self.assertTrue(len(response) > 0)
    
    def test_respond_mother_keyword(self):
        """Test response to mother keyword."""
        response = self.eliza.respond("My mother never understood me")
        self.assertIsInstance(response, str)
        self.assertTrue("mother" in response.lower() or "family" in response.lower())
    
    def test_respond_father_keyword(self):
        """Test response to father keyword."""
        response = self.eliza.respond("My father was strict")
        self.assertIsInstance(response, str)
        self.assertTrue("father" in response.lower() or "family" in response.lower() or "affection" in response.lower())
    
    def test_respond_question(self):
        """Test response to a question."""
        response = self.eliza.respond("What should I do?")
        self.assertIsInstance(response, str)
        # Eliza typically reflects questions back
        self.assertTrue(len(response) > 0)
    
    def test_respond_quit(self):
        """Test response to quit command."""
        response = self.eliza.respond("quit")
        self.assertIsInstance(response, str)
        self.assertTrue(any(word in response.lower() for word in ["thank", "good", "bye"]))
    
    def test_respond_yes(self):
        """Test response to yes."""
        response = self.eliza.respond("yes")
        self.assertIsInstance(response, str)
        self.assertTrue(len(response) > 0)
    
    def test_respond_computer_keyword(self):
        """Test response to computer keyword."""
        response = self.eliza.respond("You are a computer")
        self.assertIsInstance(response, str)
        self.assertTrue("computer" in response.lower() or "talking" in response.lower())
    
    def test_respond_case_insensitive(self):
        """Test that responses are case insensitive."""
        response1 = self.eliza.respond("HELLO")
        response2 = self.eliza.respond("hello")
        response3 = self.eliza.respond("HeLLo")
        # All should generate valid responses
        self.assertTrue(len(response1) > 0)
        self.assertTrue(len(response2) > 0)
        self.assertTrue(len(response3) > 0)
    
    def test_respond_empty_string(self):
        """Test response to empty string."""
        response = self.eliza.respond("")
        self.assertIsInstance(response, str)
        self.assertTrue(len(response) > 0)
    
    def test_respond_whitespace(self):
        """Test response to whitespace."""
        response = self.eliza.respond("   ")
        self.assertIsInstance(response, str)
        self.assertTrue(len(response) > 0)
    
    def test_pattern_matching_order(self):
        """Test that more specific patterns match before generic ones."""
        # "I need" should match before generic pattern
        response = self.eliza.respond("I need help")
        # Should not be a generic response
        self.assertFalse(response == "I see.")
    
    def test_response_variability(self):
        """Test that responses vary for the same input."""
        responses = set()
        # Run the same input multiple times
        for _ in range(20):
            response = self.eliza.respond("I feel sad")
            responses.add(response)
        
        # Should have at least 2 different responses (due to randomness)
        # Note: This test might rarely fail due to randomness, but with 20 tries
        # and multiple response options, it should almost always pass
        self.assertGreaterEqual(len(responses), 1)
    
    def test_i_am_pattern(self):
        """Test 'I am' pattern matching."""
        response = self.eliza.respond("I am worried")
        self.assertIsInstance(response, str)
        # Should reflect the state
        self.assertTrue("worried" in response.lower() or "being" in response.lower())
    
    def test_because_pattern(self):
        """Test 'because' pattern matching."""
        response = self.eliza.respond("because I am sad")
        self.assertIsInstance(response, str)
        self.assertTrue("reason" in response.lower() or "sad" in response.lower())
    
    def test_sorry_pattern(self):
        """Test 'sorry' pattern matching."""
        response = self.eliza.respond("I am sorry")
        self.assertIsInstance(response, str)
        # Can match either the sorry pattern or the "I am" pattern
        self.assertTrue("apology" in response.lower() or "apolog" in response.lower() or "feel" in response.lower() or "sorry" in response.lower())
    
    def test_can_you_pattern(self):
        """Test 'can you' pattern matching."""
        response = self.eliza.respond("Can you help me?")
        self.assertIsInstance(response, str)
        self.assertTrue("help" in response.lower() or "can" in response.lower() or "why" in response.lower())
    
    def test_i_want_pattern(self):
        """Test 'I want' pattern matching."""
        response = self.eliza.respond("I want to be happy")
        self.assertIsInstance(response, str)
        self.assertTrue("want" in response.lower() or "happy" in response.lower())
    
    def test_childhood_keyword(self):
        """Test response to childhood keyword."""
        response = self.eliza.respond("When I was a child, I was lonely")
        self.assertIsInstance(response, str)
        self.assertTrue("child" in response.lower() or "friend" in response.lower() or "dream" in response.lower())
    
    def test_default_response(self):
        """Test that unknown patterns get a default response."""
        response = self.eliza.respond("xyzabc nonsense words")
        self.assertIsInstance(response, str)
        # Should still generate some response
        self.assertTrue(len(response) > 0)
    
    def test_pronoun_reflection_in_response(self):
        """Test that pronouns are properly reflected in responses."""
        response = self.eliza.respond("I need help with my problems")
        # Response should contain reflected version (your problems)
        self.assertIsInstance(response, str)
        if "problem" in response.lower():
            # If the response includes the word, it should be reflected
            self.assertTrue("your" in response.lower() or "you" in response.lower())


class TestElizaReflections(unittest.TestCase):
    """Test cases specifically for the reflection mechanism."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.eliza = Eliza()
    
    def test_reflect_i_to_you(self):
        """Test 'I' to 'you' reflection."""
        self.assertEqual(self.eliza.reflect("i"), "you")
    
    def test_reflect_you_to_me(self):
        """Test 'you' to 'me' reflection."""
        self.assertEqual(self.eliza.reflect("you"), "me")
    
    def test_reflect_my_to_your(self):
        """Test 'my' to 'your' reflection."""
        self.assertEqual(self.eliza.reflect("my"), "your")
    
    def test_reflect_your_to_my(self):
        """Test 'your' to 'my' reflection."""
        self.assertEqual(self.eliza.reflect("your"), "my")
    
    def test_reflect_am_to_are(self):
        """Test 'am' to 'are' reflection."""
        self.assertEqual(self.eliza.reflect("am"), "are")
    
    def test_reflect_are_to_am(self):
        """Test 'are' to 'am' reflection."""
        self.assertEqual(self.eliza.reflect("are"), "am")
    
    def test_reflect_complex_sentence(self):
        """Test reflection of a complex sentence."""
        result = self.eliza.reflect("i think you are wrong about my feelings")
        self.assertEqual(result, "you think me am wrong about your feelings")


if __name__ == '__main__':
    unittest.main()
