from accelerate import Accelerator
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer


class TrainingSystem:
    def __init__(self, config):
        self.config = config
        self.accelerator = Accelerator()
        self.tokenizer = AutoTokenizer.from_pretrained(config['model_name'])
        self.model = AutoModelForCausalLM.from_pretrained(config['model_name'])
        self.reward_calculator = MathRewardCalculator()

    def train(self, train_data):
        # Prepare data loader
        train_loader = DataLoader(
            train_data,
            batch_size=self.config['batch_size'],
            shuffle=True
        )

        # Prepare reference model
        ref_model = AutoModelForCausalLM.from_pretrained(self.config['model_name'])
        ref_model.requires_grad_(False)

        # Initialize trainer
        trainer = GRPOTrainer(
            model=self.model,
            ref_model=ref_model,
            tokenizer=self.tokenizer,
            config=self.config
        )

        # Prepare components for distributed training
        model, optimizer, train_loader = self.accelerator.prepare(
            self.model, trainer.optimizer, train_loader
        )

        # Training loop
        for epoch in range(self.config['epochs']):
            for batch in train_loader:
                # Generate responses
                with torch.no_grad():
                    outputs = model.generate(
                        input_ids=batch['input_ids'],
                        max_length=self.config['max_length'],
                        num_return_sequences=1
                    )
                responses = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

                # Calculate rewards
                rewards = []
                for response, gt in zip(responses, batch['ground_truth']):
                    rewards.append(self.reward_calculator.calculate_reward(response, gt)['total'])

                # Update policy
                loss = trainer.compute_loss({
                    'input_ids': batch['input_ids'],
                    'responses': responses,
                    'rewards': rewards
                })

                self.accelerator.backward(loss)
                optimizer.st()
                optimizer.zero_grad()

                # Logging
                if self.accelerator.is_main_process:
                    print(f"Epoch {epoch} Loss: {loss.item()} Avg Reward: {sum(rewards) / len(rewards)}")