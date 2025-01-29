from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer


class MathDatasetLoader:
    def __init__(self, tokenizer_name="deepseek-ai/deepseek-math-7b-base"):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def load(self, dataset_name, split_ratio=0.95):
        """Load and tokenize dataset with train/validation split"""
        dataset = load_dataset(dataset_name)
        tokenized_data = self._process_dataset(dataset)
        return self._split_dataset(tokenized_data, split_ratio)

    def _process_dataset(self, dataset):
        processed = []
        for example in dataset['train']:
            if self._validate_example(example):
                processed.append(self._tokenize_example(example))
        return processed

    def _validate_example(self, example):
        required_keys = ['problem', 'solution', 'type']
        return all(key in example for key in required_keys) and \
            len(example['solution']) > 50

    def _tokenize_example(self, example):
        return {
            'input_ids': self.tokenizer(
                f"User: {example['problem']}\nAssistant: ",
                return_tensors='pt',
                truncation=True,
                max_length=2048
            )['input_ids'][0],
            'ground_truth': example['solution'],
            'task_type': example['type']
        }

    def _split_dataset(self, data, split_ratio):
        split_idx = int(len(data) * split_ratio)
        return DatasetDict({
            'train': data[:split_idx],
            'val': data[split_idx:]
        })