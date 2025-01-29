import unittest
import torch
from training.grpo_trainer import GRPOTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer

class TestTrainingComponents(unittest.TestCase):
    def setUp(self):
        self.model = AutoModelForCausalLM.from_pretrained("deepseek-ai/deepseek-math-7b-base")
        self.tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/deepseek-math-7b-base")
        self.config = {
            'group_size': 2,
            'lr': 2e-5,
            'epsilon': 0.2,
            'beta': 0.01,
            'max_length': 512
        }

    def test_trainer_initialization(self):
        trainer = GRPOTrainer(
            model=self.model,
            ref_model=self.model,
            tokenizer=self.tokenizer,
            config=self.config
        )
        self.assertIsNotNone(trainer.optimizer)

    def test_loss_computation(self):
        dummy_batch = {
            'input_ids': [torch.randint(0, 1000, (10,))],
            'responses': ["<think>Test</think><answer>42</answer>"],
            'rewards': [1.0]
        }
        trainer = GRPOTrainer(
            model=self.model,
            ref_model=self.model,
            tokenizer=self.tokenizer,
            config=self.config
        )
        loss = trainer.compute_loss(dummy_batch)
        self.assertIsInstance(loss, torch.Tensor)

if __name__ == '__main__':
    unittest.main()