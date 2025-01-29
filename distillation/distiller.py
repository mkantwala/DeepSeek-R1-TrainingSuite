import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer


class SafetyDistiller:
    def __init__(self, teacher_model, student_model, tokenizer, temperature=0.7):
        self.teacher = teacher_model
        self.student = student_model
        self.tokenizer = tokenizer
        self.temperature = temperature
        self.kl_loss = torch.nn.KLDivLoss(reduction="batchmean")

    def distill(self, dataset, epochs=3, batch_size=32):
        train_data = self._generate_training_data(dataset)
        loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

        optimizer = torch.optim.AdamW(self.student.parameters(), lr=5e-6)

        for epoch in range(epochs):
            total_loss = 0
            for batch in loader:
                loss = self._distillation_step(batch)
                total_loss += loss.item()

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.student.parameters(), 1.0)
                optimizer.step()

            print(f"Epoch {epoch + 1} Loss: {total_loss / len(loader):.4f}")

    def _generate_training_data(self, dataset):
        distilled_data = []
        for example in dataset:
            with torch.no_grad():
                outputs = self.teacher.generate(
                    input_ids=example['input_ids'].unsqueeze(0),
                    max_length=2048,
                    temperature=self.temperature,
                    do_sample=True,
                    top_p=0.9
                )
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            if self._safety_check(response):
                distilled_data.append({
                    'input_ids': example['input_ids'],
                    'labels': outputs[0]
                })
        return distilled_data

    def _safety_check(self, text):
        safety_keywords = ['malicious', 'harmful', 'dangerous']
        return not any(keyword in text.lower() for keyword in safety_keywords)

    def _distillation_step(self, batch):
        teacher_outputs = self.teacher(
            input_ids=batch['input_ids'],
            labels=batch['labels']
        )
        student_outputs = self.student(
            input_ids=batch['input_ids'],
            labels=batch['labels']
        )

        teacher_probs = torch.nn.functional.softmax(teacher_outputs.logits / self.temperature, dim=-1)
        student_log_probs = torch.nn.functional.log_softmax(student_outputs.logits / self.temperature, dim=-1)

        kl_loss = self.kl_loss(student_log_probs, teacher_probs)
        return kl_loss