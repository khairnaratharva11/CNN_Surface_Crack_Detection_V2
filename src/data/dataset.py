import os
import yaml
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def load_config():
    config_path = os.path.join(os.path.dirname(__file__), '../../config.yaml')
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def get_data_generators():
    config = load_config()
    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../', config['paths']['dataset_v1_dir']))
    
    img_size = tuple(config['model']['image_size'])
    batch_size = config['model']['batch_size']
    
    # Adding more robust augmentations for real-world footage
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=20,
        zoom_range=0.2,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        brightness_range=[0.8, 1.2],  # Handle different lighting
        channel_shift_range=20.0      # Handle camera noise/color shifts
    )
    
    validation_datagen = ImageDataGenerator(rescale=1.0 / 255)
    test_datagen = ImageDataGenerator(rescale=1.0 / 255)
    
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'validation')
    test_dir = os.path.join(data_dir, 'test')
    
    train_data = train_datagen.flow_from_directory(
        train_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode="binary"
    )
    
    validation_data = validation_datagen.flow_from_directory(
        val_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode="binary"
    )
    
    test_data = test_datagen.flow_from_directory(
        test_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode="binary",
        shuffle=False
    )
    
    return train_data, validation_data, test_data
