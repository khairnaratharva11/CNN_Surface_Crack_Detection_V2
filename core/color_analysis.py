import cv2
import numpy as np
import json
import os
import math

def hex_to_rgb(hex_code):
    hex_code = hex_code.lstrip('#')
    return tuple(int(hex_code[i:i+2], 16) for i in (0, 2, 4))

def rgb_to_hex(rgb):
    return "#{:02x}{:02x}{:02x}".format(int(rgb[0]), int(rgb[1]), int(rgb[2]))

def color_distance(c1, c2):
    # Simple Euclidean distance
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(c1, c2)))

def load_color_palette():
    palette_path = os.path.join(os.path.dirname(__file__), 'colors.json')
    if not os.path.exists(palette_path):
        return {}
    with open(palette_path, 'r') as f:
        return json.load(f)

def extract_dominant_colors(image_path, mask=None, k=3):
    """
    Extracts dominant colors from the image. If mask is provided (where 255 is the crack),
    we invert it to sample the non-crack surface pixels.
    """
    if not os.path.exists(image_path):
        return None
        
    img = cv2.imread(image_path)
    if img is None:
        return None
        
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    if mask is not None:
        # We want to sample the surface, so we exclude the crack (mask == 255)
        # and we also exclude very dark shadows.
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        valid_pixels_mask = cv2.bitwise_and(cv2.bitwise_not(mask), (gray > 20).astype(np.uint8)*255)
        pixels = img[valid_pixels_mask == 255]
    else:
        pixels = img.reshape((-1, 3))
        
    if len(pixels) < k:
        return None
        
    pixels = np.float32(pixels)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    
    centers = np.uint8(centers)
    
    # Find the most dominant cluster
    counts = np.bincount(labels.flatten())
    dominant_idx = np.argmax(counts)
    dominant_color = centers[dominant_idx].tolist()
    
    dominant_hex = rgb_to_hex(dominant_color)
    
    palette = load_color_palette()
    nearest_name = "Unknown"
    min_dist = float('inf')
    
    for name, rgb in palette.items():
        dist = color_distance(dominant_color, rgb)
        if dist < min_dist:
            min_dist = dist
            nearest_name = name
            
    return {
        "dominant_color_rgb": dominant_color,
        "dominant_color_hex": dominant_hex,
        "nearest_color_name": nearest_name
    }
