from transformers import AutoModelForCausalLM, AutoConfig
from peft import get_peft_config, get_peft_model


def load_model_with_lora(model_name, lora_config=None):
    """Load base model with LoRA configuration"""
    config = AutoConfig.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, config=config)

    if lora_config:
        peft_config = get_peft_config({
            "peft_type": "LORA",
            "task_type": "CAUSAL_LM",
            "inference_mode": False,
            **lora_config
        })
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

    return model


def freeze_base_model(model):
    """Freeze all base model parameters except attention layers"""
    for param in model.parameters():
        param.requires_grad = False

    for layer in model.transformer.h:
        layer.attention.requires_grad_(True)

    return model