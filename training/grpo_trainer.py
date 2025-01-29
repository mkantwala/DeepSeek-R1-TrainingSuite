import torch
from torch.nn import KLDivLoss
from torch.optim import AdamW
from transformers import AutoModelForCausalLM


class GRPOTrainer:
    def __init__(self, model, ref_model, tokenizer, config):
        self.model = model
        self.ref_model = ref_model
        self.tokenizer = tokenizer
        self.config = config
        self.optimizer = AdamW(model.parameters(), lr=config['lr'])
        self.kl_loss = KLDivLoss(reduction="batchmean")

    def compute_loss(self, batch):
        queries = batch["input_ids"]
        responses = batch["responses"]
        rewards = torch.tensor(batch["rewards"], device=self.model.device)

        # Process in groups
        group_losses = []
        for i in range(0, len(responses), self.config['group_size']):
            group_responses = responses[i:i + self.config['group_size']]
            group_rewards = rewards[i:i + self.config['group_size']]

            with torch.no_grad():
                ref_logits = self.ref_model(**self._prepare_batch(queries[i], group_responses)).logits

            current_logits = self.model(**self._prepare_batch(queries[i], group_responses)).logits

            # Calculate policy gradient loss
            advantages = self._calculate_advantages(group_rewards)
            ratios = torch.exp(current_logits - ref_logits)
            clipped_ratios = torch.clamp(ratios, 1 - self.config['epsilon'], 1 + self.config['epsilon'])
            policy_loss = -torch.min(ratios * advantages, clipped_ratios * advantages).mean()

            # KL penalty
            kl_penalty = self.kl_loss(
                torch.nn.functional.log_softmax(current_logits, dim=-1),
                torch.nn.functional.softmax(ref_logits, dim=-1)
            )

            group_loss = policy_loss + self.config['beta'] * kl_penalty
            group_losses.append(group_loss)

        return torch.stack(group_losses).mean()

    def _calculate_advantages(self, rewards):
        mean_reward = rewards.mean()
        std_reward = rewards.std() + 1e-8
        return (rewards - mean_reward) / std_reward

    def _prepare_batch(self, query, responses):
        texts = [f"{self.tokenizer.decode(query)} {resp}" for resp in responses]
        return self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.config['max_length']
        ).to(self.model.device)