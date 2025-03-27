import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, confusion_matrix
import pandas as pd

def plot_training_curves(train_losses, validation_losses, train_accuracies, validation_accuracies, plot_path):
    epochs = list(range(1, len(train_losses) + 1))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(epochs, train_losses, label="Train Loss", color="blue")
    ax1.plot(epochs, validation_losses, label="Validation Loss", color="red")
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.set_title('Loss vs Epochs')
    ax1.legend()

    ax2.plot(epochs, train_accuracies, label="Train Accuracy", color="blue")
    ax2.plot(epochs, validation_accuracies, label="Validation Accuracy", color="red")
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Accuracy vs Epochs')
    ax2.legend()

    plt.tight_layout()
    plt.savefig(plot_path)
    print(f"Training curves saved to {plot_path}")
    plt.close()

def plot_roc_curve(model, dataset_loader, device, plot_path):
    model.eval()
    all_labels = []
    all_preds = []
    
    with torch.no_grad():
        for images, labels in dataset_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            probabilities = torch.softmax(outputs, dim=1)  # Convert logits to probabilities

            all_labels.append(labels.cpu())
            all_preds.append(probabilities.cpu())
    
    # Concatenate all labels and predictions
    all_labels = torch.cat(all_labels)
    all_preds = torch.cat(all_preds)
    
    # For multiclass classification (each class's ROC curve)
    n_classes = all_preds.shape[1]  # Number of classes (should be 10 for MNIST)
    fpr = {}
    tpr = {}
    roc_auc = {}
    
    # Compute ROC curve and AUC for each class
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(all_labels.numpy(), all_preds[:, i].numpy(), pos_label=i)
        roc_auc[i] = auc(fpr[i], tpr[i])
        plt.plot(fpr[i], tpr[i], label=f'Class {i} ROC curve (AUC = {roc_auc[i]:.4f})')

    # Plot ROC curve for all classes
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve for MNIST')
    plt.legend(loc='lower right')

    plt.savefig(plot_path)
    print(f"ROC curves saved to {plot_path}")
    plt.close()

def plot_confusion_matrix(model, dataset_loader, device, class_names, plot_path):
    model.eval()  # Set the model to evaluation mode

    all_labels = []
    all_preds = []

    with torch.no_grad():  # No need to track gradients during evaluation
        for images, labels in dataset_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)  # Get the predicted class with max probability

            all_labels.append(labels.cpu())
            all_preds.append(predicted.cpu())

    # Concatenate all labels and predictions
    all_labels = torch.cat(all_labels)
    all_preds = torch.cat(all_preds)

    # Generate confusion matrix
    cm = confusion_matrix(all_labels.numpy(), all_preds.numpy())

    # Convert to pandas DataFrame for better visualization
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)

    # Plot the confusion matrix using seaborn's heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues", cbar=False, xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')

    plt.savefig(plot_path)
    print(f"ROC curves saved to {plot_path}")
    plt.close()

