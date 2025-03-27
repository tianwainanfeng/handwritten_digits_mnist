import os
import time
import numpy as np
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from torchvision import transforms
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

from src.config_loader import CONFIG
from src.utils.plot_utils import plot_training_curves, plot_roc_curve, plot_confusion_matrix
from src.data.load_data import get_mnist_data
from models.simple_cnn import SimpleCNN  # Import the CNN model

# ==== Setup Logging ====

log_file = os.path.join(CONFIG["output"]["logs"] + "train.log")
os.makedirs(os.path.dirname(log_file), exist_ok=True)  # Ensure the log directory exists

logging.basicConfig(
    filename=log_file,
    filemode="w",  # Overwrite log file for each run
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO
)

# ==== Load Configuration ====

SEED = CONFIG["random_seed"]
BATCH_SIZE = CONFIG["training"]["batch_size"]
EPOCHS = CONFIG["training"]["epochs"]
LEARNING_RATE = CONFIG["training"]["learning_rate"]
VALIDATION_SPLIT = CONFIG["training"]["validation_split"]
EARLY_STOP = CONFIG["training"]["early_stop"]
USE_SUBSET = CONFIG["training"]["use_subset"]
SUBSET_SIZE = CONFIG["training"]["subset_size"]

DEVICE = CONFIG["device"]

# ==== Seed ====

# Set random seed for reproducibility
torch.manual_seed(SEED)
np.random.seed(SEED)

# ==== Load MNIST Data ====

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

train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)

# ==== Initialize Model, Loss, and Optimizer ====

model = SimpleCNN().to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

#Track Metrics
train_losses, validation_losses = [], []
train_accuracies, validation_accuracies = [], []

best_validation_loss = float('inf')
best_epoch = 0
best_model_path = os.path.join(CONFIG["output"]["models"], "best_model.pth")

# ==== Training Loop ====

# Train the model
def train_model():
    global best_validation_loss, best_epoch
    logging.info("\nStart training...")
    
    total_start_time = time.time() # Start total training time

    for epoch in range(EPOCHS):
        model.train()  # Set the model to training mode
        epoch_start_time  = time.time() # Start epoch time
        total_loss = 0.0
        correct, total = 0, 0

        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        epoch_time = time.time() - epoch_start_time # Time for this epoch
        avg_loss = total_loss / len(train_loader)
        accuracy = 100. * correct / total

        train_losses.append(avg_loss)
        train_accuracies.append(accuracy)
        
        logging.info(f"Epoch {epoch+1}: Train Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%, Time: {epoch_time:.2f}s")
        print(f"Epoch {epoch+1}: Train Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%, Time: {epoch_time:.2f}s")
        
        if VALIDATION_SPLIT > 0:
            # Evaluate after each epoch
            validation_loss, validation_accuracy = evaluate_model()
        
            validation_losses.append(validation_loss)
            validation_accuracies.append(validation_accuracy)

            logging.info(f"Epoch [{epoch+1}/{EPOCHS}]: Validation Loss: {validation_loss:.4f}, Accuracy: {validation_accuracy:.4f}")
            print(f"Epoch [{epoch+1}/{EPOCHS}]: Validation Loss: {validation_loss:.4f}, Accuracy: {validation_accuracy:.4f}")

            # Save best model
            if validation_loss < best_validation_loss:
                best_validation_loss = validation_loss
                best_epoch = epoch
                save_model(best_model_path)
            elif epoch - best_epoch >= EARLY_STOP:
                logging.info(f"Best model saved at epoch {best_epoch+1} with validation_loss: {best_validation_loss:.4f}")
                print(f"Best model saved at epoch {best_epoch+1} with validation_loss: {best_validation_loss:.4f}")
                print("Early stopping triggered.")
                break

    total_training_time = time.time() - total_start_time; # Total training time
    avg_epoch_time = total_training_time / EPOCHS # Average time per epoch

    logging.info(f"Total Training Time: {total_training_time:.2f}s, Average Time per Epoch: {avg_epoch_time:.2f}s")
    print(f"Total Training Time: {total_training_time:.2f}s, Average Time per Epoch: {avg_epoch_time:.2f}s")
    
    logging.info("Finish training.")

# ==== Evaluation ====

# Evaluate the model on the test data
def evaluate_model():
    model.eval()  # Set the model to evaluation mode
    
    total_loss = 0.0
    correct, total = 0, 0

    with torch.no_grad():  # No need to track gradients during evaluation
        for images, labels in validation_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / len(validation_loader)
    accuracy = 100.0 * correct / total

    return avg_loss, accuracy

# ==== Save Model ====

def save_model(model_save_path):
    torch.save(model.state_dict(), model_save_path)
    logging.info(f"Model saved to {model_save_path}")
    print(f"Model saved to {model_save_path}")

# ==== Main Execution ====

if __name__ == "__main__":
    train_model()

    # ---- Plot training curves ----

    plot_training_curves(train_losses, validation_losses, train_accuracies, validation_accuracies, CONFIG["output"]["plots"] + "training_curves.png")
    
    # ---- Validation ROC curves ----

    best_model_path = os.path.join(CONFIG["output"]["models"], "best_model.pth")
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path))
        model.to(DEVICE)  # Ensure model is on the correct device
        print(f"Loaded best model from {best_model_path}")
    else:
        print("Warning: Best model file not found! Using last trained model.")

    # Now call the ROC curve plotting function with the best model
    plot_roc_curve(model, validation_loader, DEVICE, CONFIG["output"]["plots"] + "roc_curve_validation.png")

    # ---- Validation Confusion Matrix ----
    # Class names for MNIST (digits 0-9)
    class_names = [str(i) for i in range(10)]

    plot_confusion_matrix(model, validation_loader, DEVICE, class_names, os.path.join(CONFIG["output"]["plots"], "confusion_matrix_validation.png"))

