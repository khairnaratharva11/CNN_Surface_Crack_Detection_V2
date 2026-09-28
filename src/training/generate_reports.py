import os
import yaml
import sys
import tensorflow as tf

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from src.data.dataset import load_config, get_data_generators
from src.models.architectures import get_model
from src.training.evaluate import evaluate_and_plot

def main():
    config = load_config()
    arch_name = config['model']['arch']
    
    print("Loading data...")
    _, _, test_data = get_data_generators()
    
    print("Loading model...")
    img_size = tuple(config['model']['image_size']) + (3,)
    model = get_model(arch_name, input_shape=img_size)
    
    models_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../', config['paths']['models_dir']))
    model_path = os.path.join(models_dir, f'{arch_name}_best.keras')
    
    if os.path.exists(model_path):
        model.load_weights(model_path)
        print(f"Loaded weights from {model_path}")
        evaluate_and_plot(model, test_data, None, config, arch_name)
    else:
        print(f"Model weights not found at {model_path}")

if __name__ == "__main__":
    main()
