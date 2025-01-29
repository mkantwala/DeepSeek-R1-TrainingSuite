import unittest
from data.dataset_loader import MathDatasetLoader

class TestDataPipeline(unittest.TestCase):
    def setUp(self):
        self.loader = MathDatasetLoader()
        self.dataset = self.loader.load("math_dataset", split_ratio=0.9)

    def test_dataset_split(self):
        self.assertIn('train', self.dataset)
        self.assertIn('val', self.dataset)
        self.assertGreater(len(self.dataset['train']), 0)
        self.assertGreater(len(self.dataset['val']), 0)

    def test_example_format(self):
        example = self.dataset['train'][0]
        self.assertIn('input_ids', example)
        self.assertIn('ground_truth', example)
        self.assertIn('task_type', example)
        self.assertIsInstance(example['input_ids'], torch.Tensor)

    def test_tokenization(self):
        example = self.dataset['train'][0]
        decoded = self.loader.tokenizer.decode(example['input_ids'])
        self.assertTrue(decoded.startswith("User: "))
        self.assertIn("Assistant:", decoded)

if __name__ == '__main__':
    unittest.main()