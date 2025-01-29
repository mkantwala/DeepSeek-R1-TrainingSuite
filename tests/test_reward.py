import unittest
from reward.reward_calculator import MathRewardCalculator, CodeRewardCalculator


class TestRewardSystem(unittest.TestCase):
    def setUp(self):
        self.math_calculator = MathRewardCalculator()
        self.code_calculator = CodeRewardCalculator()

    def test_math_reward(self):
        # Test correct format and answer
        response = "<think>Calculate 2+2</think><answer>4</answer>"
        reward = self.math_calculator.calculate_reward(response, "4")
        self.assertEqual(reward['total'], 1.3)

        # Test incorrect answer
        response = "<think>Calculate 2+2</think><answer>5</answer>"
        reward = self.math_calculator.calculate_reward(response, "4")
        self.assertEqual(reward['total'], 0.3)

    def test_code_reward(self):
        test_cases = [("add(2,3)", "5"), ("multiply(3,4)", "12")]
        correct_code = "def add(a,b): return a+b\ndef multiply(a,b): return a*b"

        # Test correct code
        response = f"<think>Implement functions</think><answer>{correct_code}</answer>"
        reward = self.code_calculator.calculate_reward(response, test_cases)
        self.assertAlmostEqual(reward['total'], 1.3, delta=0.01)


if __name__ == '__main__':
    unittest.main()