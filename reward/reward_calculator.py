import re
import subprocess
import tempfile
from math import isclose
from typing import List, Tuple


class MathRewardCalculator:
    def __init__(self):
        self.pattern = re.compile(
            r"<think>(?P<reasoning>.*?)</think>\s*<answer>(?P<solution>.*?)</answer>",
            re.DOTALL
        )

    def calculate_reward(self, response: str, ground_truth: str) -> dict:
        match = self.pattern.search(response)
        if not match:
            return {"total": 0.0, "accuracy": 0.0, "format": 0.0}

        formatted = 0.3  # Base format score
        accuracy = self._verify_solution(match.group("solution"), ground_truth)
        return {
            "total": accuracy + formatted,
            "accuracy": accuracy,
            "format": formatted
        }

    def _verify_solution(self, solution: str, ground_truth: str) -> float:
        try:
            # Extract numerical answer
            numbers = [float(x) for x in re.findall(r"-?\d+\.?\d*", solution)]
            gt_number = float(re.search(r"-?\d+\.?\d*", ground_truth).group())

            # Check proximity
            for num in numbers:
                if isclose(num, gt_number, rel_tol=1e-3):
                    return 1.0
            return 0.0
        except:
            return 0.0


class CodeRewardCalculator(MathRewardCalculator):
    def _verify_solution(self, code: str, test_cases: List[Tuple[str, str]]) -> float:
        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir) / "test_code.py"
            with open(test_file, "w") as f:
                f.write(code + "\n\n")
                for inp, exp in test_cases:
                    f.write(f'print({inp})  # Expected: {exp}\n')

            try:
                result = subprocess.run(
                    ["python", str(test_file)],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                outputs = result.stdout.strip().split('\n')
                correct = 0
                for output, (_, expected) in zip(outputs, test_cases):
                    if output.strip() == expected.strip():
                        correct += 1
                return correct / len(test_cases)
            except (subprocess.TimeoutExpired, subprocess.CalledProcessError):
                return 0.0