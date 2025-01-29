import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from data.dataset_loader import MathDatasetLoader
from distillation.distiller import SafetyDistiller


def main():
    # Configuration
    teacher_model_name = "trained_teacher"
    student_model_name = "deepseek-ai/deepseek-math-7b-base"
    dataset_name = "math_dataset"

    # Load models
    teacher = AutoModelForCausalLM.from_pretrained(teacher_model_name)
    student = AutoModelForCausalLM.from_pretrained(student_model_name)
    tokenizer = AutoTokenizer.from_pretrained(student_model_name)

    # Load dataset
    loader = MathDatasetLoader()
    dataset = loader.load(dataset_name)

    # Initialize distiller
    distiller = SafetyDistiller(
        teacher_model=teacher,
        student_model=student,
        tokenizer=tokenizer,
        temperature=0.7
    )

    # Run distillation
    distiller.distill(dataset['val'], epochs=5, batch_size=16)

    # Save distilled model
    student.save_pretrained("distilled_student")
    tokenizer.save_pretrained("distilled_student")


if __name__ == "__main__":
    main()