import cv2
import numpy as np
import tensorflow as tf

def generate_gradcam(model, patch, class_index=0):
    """
    Generate Grad-CAM for a MobileNetV2 patch.
    patch: preprocessed float32 image (1, 128, 128, 3)
    """
    # Assuming MobileNetV2 is the inner model if built via Sequential/Model wrapper
    # We find the last convolutional layer. In MobileNetV2 it's usually 'out_relu'
    inner_model = model.layers[1] if hasattr(model.layers[1], 'layers') else model
    
    # Try to find the last conv layer dynamically
    last_conv_layer_name = None
    for layer in reversed(inner_model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            last_conv_layer_name = layer.name
            break
            
    if not last_conv_layer_name:
        return np.zeros(patch.shape[1:3])
        
    grad_model = tf.keras.models.Model(
        [inner_model.inputs], 
        [inner_model.get_layer(last_conv_layer_name).output, inner_model.output]
    )
    
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(patch)
        loss = predictions[:, class_index]
        
    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()
