import os
import streamlit as st
import tensorflow as tf
from src.models.architectures import get_model

@st.cache_resource
def load_crack_model(arch_name, image_size, models_dir):
    """
    Loads the Keras model using Streamlit's resource caching to ensure
    it is only loaded once per session.
    """
    model_path = os.path.join(models_dir, f'{arch_name}_finetuned_best.keras')
    
    if not os.path.exists(model_path):
        model_path = os.path.join(models_dir, f'{arch_name}_best.keras')
        
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}. Please train the model first.")
        
    print(f"Loading model from: {model_path}")
    
    input_shape = tuple(image_size) + (3,)
    model = get_model(arch_name, input_shape=input_shape)
    model.load_weights(model_path)
    return model
