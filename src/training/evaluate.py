import os
import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_curve, auc, 
    precision_recall_curve, average_precision_score, accuracy_score
)
import seaborn as sns

def evaluate_and_plot(model, test_data, history, config, arch_name):
    reports_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../', config['paths']['reports_dir']))
    os.makedirs(reports_dir, exist_ok=True)
    
    # 1. Plot Training History
    if history:
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label='Train Acc')
        plt.plot(history.history['val_accuracy'], label='Val Acc')
        plt.title('Accuracy')
        plt.xlabel('Epochs')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='Train Loss')
        plt.plot(history.history['val_loss'], label='Val Loss')
        plt.title('Loss')
        plt.xlabel('Epochs')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(reports_dir, f'{arch_name}_training_curves.png'))
        plt.close()
        
    # 2. Evaluate on Test Data
    print("Evaluating on test data...")
    test_data.reset()
    y_true = test_data.classes
    y_pred_prob = model.predict(test_data).ravel()
    
    # Choose threshold based on F1/Recall (using 0.5 as default or validation based)
    threshold = config['inference']['confidence_threshold']
    y_pred = (y_pred_prob > threshold).astype(int)
    
    # 3. Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=list(test_data.class_indices.keys()), 
                yticklabels=list(test_data.class_indices.keys()))
    plt.title(f'Confusion Matrix (Threshold={threshold})')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.savefig(os.path.join(reports_dir, f'{arch_name}_confusion_matrix.png'))
    plt.close()
    
    # 4. ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_pred_prob)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(reports_dir, f'{arch_name}_roc_curve.png'))
    plt.close()
    
    # 5. Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_true, y_pred_prob)
    ap = average_precision_score(y_true, y_pred_prob)
    
    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision, color='blue', lw=2, label=f'PR curve (AP = {ap:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.savefig(os.path.join(reports_dir, f'{arch_name}_pr_curve.png'))
    plt.close()
    
    # 6. Classification Report & Metrics JSON
    report = classification_report(y_true, y_pred, target_names=list(test_data.class_indices.keys()), output_dict=True)
    report_text = classification_report(y_true, y_pred, target_names=list(test_data.class_indices.keys()))
    print("Classification Report:")
    print(report_text)
    
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "roc_auc": roc_auc,
        "average_precision": ap,
        "classification_report": report,
        "threshold_used": threshold
    }
    
    with open(os.path.join(reports_dir, f'{arch_name}_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=4)
        
    print(f"Evaluation complete. Reports saved to {reports_dir}")
    
    return metrics
