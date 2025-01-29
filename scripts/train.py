import yaml
from data.preprocessor import MathDataPreprocessor
from training.training_system import TrainingSystem


def load_config(config_path):
    with open(config_path) as f:
        return yaml.safe_load(f)


def main():
    config = load_config("configs/base_config.yaml")

    # Prepare data
    preprocessor = MathDataPreprocessor(config['model_name'])
    data = preprocessor.load_dataset()

    # Initialize training system
    trainer = TrainingSystem(config)

    # Start training
    trainer.train(data['train'])


if __name__ == "__main__":
    main()