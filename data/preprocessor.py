import re
from datasets import load_dataset
from transformers import AutoTokenizer


class MathDataPreprocessor:
    def __init__(self, tokenizer_name):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.template = """A conversation between User and Assistant.
User: {prompt}
Assistant: <think>{reasoning}</think><answer>{solution}</answer>"""

    def load_dataset(self, dataset_name="competition_math"):
        dataset = load_dataset(dataset_name)
        return self.process_data(dataset['train'])

    def process_data(self, raw_data):
        processed = []
        for ex in raw_data:
            if self.quality_check(ex):
                processed.append(self.format_example(ex))
        return self.train_val_split(processed)

    def quality_check(self, example):
        required_keys = ['problem', 'solution', 'level']
        return all(key in example for key in required_keys) and \
            len(example['solution']) > 50 and \
            "boxed" in example['solution']

    def format_example(self, example):
        return self.template.format(
            prompt=example['problem'],
            reasoning=example.get('reasoning', ''),
            solution=example['solution']
        )

    def train_val_split(self, data, split_ratio=0.95):
        split_idx = int(len(data) * split_ratio)
        return {'train': data[:split_idx], 'val': data[split_idx:]}