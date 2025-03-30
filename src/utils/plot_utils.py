import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, confusion_matrix
import pandas as pd

# ==== Plot Training Curves ====

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

# ==== Plot ROC Curve ====

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

# ==== Plot Confusion Matrix ====

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

# ==== Plot Training Curves ====

def display_predictions(correct_samples, incorrect_samples, correct_labels, incorrect_labels, correct_preds, incorrect_preds, digit, plot_path, mode="both"):
    """
    Display correctly and/or incorrectly classified images based on mode.

    Args:
        correct_samples (list): List of correctly classified images.
        incorrect_samples (list): List of incorrectly classified images.
        correct_labels (list): List of true labels for correct predictions.
        incorrect_labels (list): List of true labels for incorrect predictions.
        correct_preds (list): List of predicted labels for correct predictions.
        incorrect_preds (list): List of predicted labels for incorrect predictions.
        digit (int): The digit being visualized.
        mode (str): "positive" to show correct, "negative" to show incorrect, "both" to show both.
    """

    if mode not in ["positive", "negative", "both"]:
        raise ValueError("Invalid mode. Choose from 'positive', 'negative', or 'both'.")

    num_correct = len(correct_samples)
    num_incorrect = len(incorrect_samples)

    if mode == "positive" and num_correct == 0:
        print(f"No correctly classified samples found for digit {digit}.")
        return
    if mode == "negative" and num_incorrect == 0:
        print(f"No misclassified samples found for digit {digit}.")
        return

    num_cols = max(num_correct, num_incorrect)
    num_rows = 1 if mode in ["positive", "negative"] else 2

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(10, 4))
    fig.suptitle(f"Predictions for Digit {digit} ({mode.capitalize()} Samples)", fontsize=14)

    axes = np.atleast_2d(axes)  # Ensure axes is always 2D

    if mode in ["positive", "both"]:
        for i, img in enumerate(correct_samples):
            axes[0, i].imshow(img.squeeze(), cmap="gray")
            axes[0, i].axis("off")
            axes[0, i].set_title(f"True: {correct_labels[i]}\nPred: {correct_preds[i]}", fontsize=8)

    if mode in ["negative", "both"]:
        for i, img in enumerate(incorrect_samples):
            row_idx = 0 if mode == "negative" else 1  # Row 0 if only showing negative, else row 1
            axes[row_idx, i].imshow(img.squeeze(), cmap="gray")
            axes[row_idx, i].axis("off")
            axes[row_idx, i].set_title(f"True: {incorrect_labels[i]}\nPred: {incorrect_preds[i]}", fontsize=8)

    plt.tight_layout()
    plt.savefig(plot_path)
    print(f"Predictions saved to {plot_path}")
    plt.close()

