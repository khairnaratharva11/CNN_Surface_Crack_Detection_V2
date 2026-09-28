import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, GlobalAveragePooling2D, Input
from tensorflow.keras.applications import MobileNetV2

def build_v1_baseline(input_shape=(128, 128, 3)):
    """The original V1 CNN architecture for baseline comparison."""
    model = Sequential()
    
    model.add(Conv2D(32, (3, 3), activation="relu", input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Conv2D(64, (3, 3), activation="relu"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Conv2D(128, (3, 3), activation="relu"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Conv2D(256, (3, 3), activation="relu"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Flatten())
    
    model.add(Dense(256, activation="relu"))
    model.add(Dropout(0.5))
    
    model.add(Dense(128, activation="relu"))
    model.add(Dropout(0.3))
    
    model.add(Dense(1, activation="sigmoid"))
    return model

def build_v2_mobilenet(input_shape=(128, 128, 3)):
    """CPU-optimized MobileNetV2 for real-time inference."""
    base_model = MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=input_shape
    )
    
    # Freeze the base model layers initially
    base_model.trainable = False
    
    inputs = Input(shape=input_shape)
    x = base_model(inputs, training=False)
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.2)(x)
    outputs = Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs, outputs)
    return model

def get_model(arch_name, input_shape=(128, 128, 3)):
    if arch_name == 'baseline':
        return build_v1_baseline(input_shape)
    elif arch_name == 'mobilenetv2':
        return build_v2_mobilenet(input_shape)
    else:
        raise ValueError(f"Unknown architecture: {arch_name}")
