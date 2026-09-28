import os
import yaml
import json
import argparse
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

# We can import modules relative to the script path or by modifying sys.path if run directly.
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.data.dataset import load_config, get_data_generators
from src.models.architectures import get_model
from src.training.evaluate import evaluate_and_plot

def main():
    parser = argparse.ArgumentParser(description="Train Crack Detection Model")
    parser.add_argument("--arch", type=str, default=None, help="Model architecture (baseline or mobilenetv2)")
    args = parser.parse_args()

    config = load_config()
    arch_name = args.arch if args.arch else config['model']['arch']
    
    tf.random.set_seed(config['model']['random_seed'])
    
    print(f"Loading data generators...")
    train_data, val_data, test_data = get_data_generators()
    
    print(f"Building {arch_name} model...")
    img_size = tuple(config['model']['image_size']) + (3,)
    model = get_model(arch_name, input_shape=img_size)
    
    # If transfer learning, compile model
    optimizer = Adam(learning_rate=config['model']['learning_rate'])
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    
    models_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../', config['paths']['models_dir']))
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, f'{arch_name}_best.keras')
    
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        ModelCheckpoint(model_path, monitor='val_accuracy', save_best_only=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-5, verbose=1)
    ]
    
    print("Starting training...")
    history = model.fit(
        train_data,
        epochs=config['model']['epochs'],
        validation_data=val_data,
        callbacks=callbacks
    )
    
    # Fine-tuning if MobileNetV2
    if arch_name == 'mobilenetv2':
        print("Unfreezing base model for fine-tuning...")
        base_model = model.layers[1]
        base_model.trainable = True
        
        # Recompile with very low learning rate
        model.compile(optimizer=Adam(learning_rate=1e-5), loss='binary_crossentropy', metrics=['accuracy'])
        
        callbacks_ft = [
            EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
            ModelCheckpoint(os.path.join(models_dir, f'{arch_name}_finetuned_best.keras'), monitor='val_accuracy', save_best_only=True, verbose=1)
        ]
        
        history_ft = model.fit(
            train_data,
            epochs=5,  # Fine tune for 5 epochs
            validation_data=val_data,
            callbacks=callbacks_ft
        )
        # Update history with fine-tuned metrics
        for k in history.history.keys():
            history.history[k].extend(history_ft.history[k])
            
        model_path = os.path.join(models_dir, f'{arch_name}_finetuned_best.keras')
        
    print("Evaluating model...")
    # Load best weights before evaluating
    model.load_weights(model_path)
    evaluate_and_plot(model, test_data, history, config, arch_name)
    
    # Save Model Card
    model_card = {
        "architecture": arch_name,
        "input_size": img_size,
        "class_indices": train_data.class_indices,
        "threshold_used": config['inference']['confidence_threshold']
    }
    with open(os.path.join(models_dir, f'{arch_name}_model_card.json'), 'w') as f:
        json.dump(model_card, f, indent=4)
        
if __name__ == "__main__":
    main()
