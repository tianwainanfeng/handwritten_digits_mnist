import os
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
import matplotlib.pyplot as plt
from src.config_loader import CONFIG
from src.log_parser import parse_train_log
from src.utils.plot_utils import plot_training_curves, plot_roc_curve, plot_confusion_matrix
from src.data.load_data import get_mnist_data
from models.simple_cnn import SimpleCNN  # Import the CNN model

# ==== Load Configuration ====

SEED = CONFIG["random_seed"]
BATCH_SIZE = CONFIG["training"]["batch_size"]
VALIDATION_SPLIT = CONFIG["training"]["validation_split"]
USE_SUBSET = CONFIG["training"]["use_subset"]
SUBSET_SIZE = CONFIG["training"]["subset_size"]

DEVICE = CONFIG["device"]

# ==== Seed ====

# Set random seed for reproducibility
torch.manual_seed(SEED)
np.random.seed(SEED)

# ==== Main Execution ====

if __name__ == "__main__":
    
    # ==== Training optimization ====

    log_file = os.path.join(CONFIG["output"]["logs"], "train.log")
    epochs, train_losses, train_accuracies, validation_losses, validation_accuracies = parse_train_log(log_file)
    plot_training_curves(train_losses, validation_losses, train_accuracies, validation_accuracies, CONFIG["output"]["plots"] + "training_curves.png")

    # ==== Validation ROC curve ====
    
    # Initialize the model
    model = SimpleCNN().to(DEVICE)  # Ensure to load model on correct device

    # Load the best model
    best_model_path = os.path.join(CONFIG["output"]["models"], "best_model.pth")
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path))
        model.to(DEVICE)  # Ensure model is on the correct device
        print(f"Loaded best model from {best_model_path}")
    else:
        print("Warning: Best model file not found! Using last trained model.")
    
    # Load validation data (same as validation data in train.py)
    (train_images, train_labels) = get_mnist_data("train")
    
    # Reshape images for CNN (N, 1, 28, 28)
    train_images = torch.tensor(train_images, dtype=torch.float32).view(-1, 1, 28, 28)
    train_labels = torch.tensor(train_labels, dtype=torch.long)
    
    if USE_SUBSET:
        train_images, train_labels = train_images[:SUBSET_SIZE], train_labels[:SUBSET_SIZE]
        print(f"Using subset of {SUBSET_SIZE} samples for training and testing.")
    
    # Split train data into train and validation
    if VALIDATION_SPLIT > 0:
        total_train_samples = len(train_images)
        train_size = int((1 - VALIDATION_SPLIT) * total_train_samples)
        validation_size = total_train_samples - train_size
    
        train_data, validation_data = random_split(TensorDataset(train_images, train_labels), [train_size, validation_size])
        validation_loader = DataLoader(validation_data, batch_size=BATCH_SIZE, shuffle=False)
    else:
        train_data = TensorDataset(train_images, train_labels)
        validation_loader = None
    
    if not validation_loader:
        print("Validation dataset is empty")
        exit()

    # Now call the ROC curve plotting function with the best model
    plot_roc_curve(model, validation_loader, DEVICE, CONFIG["output"]["plots"] + "roc_curve_validation.png")

    # ==== Validation Confusion Matrix ====

    # Class names for MNIST (digits 0-9)
    class_names = [str(i) for i in range(10)]

    # Plot confusion matrix using the function we defined earlier
    plot_confusion_matrix(model, validation_loader, DEVICE, class_names, os.path.join(CONFIG["output"]["plots"], "confusion_matrix_validation.png"))

