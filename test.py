import os
import torch
import logging
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt

from src.config_loader import CONFIG
from src.data.load_data import get_mnist_data
from src.utils.plot_utils import plot_confusion_matrix, plot_roc_curve
from models.simple_cnn import SimpleCNN  # Import the CNN model

# ==== Setup Logging ====

log_file = os.path.join(CONFIG["output"]["logs"] + "test.log")
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
DEVICE = CONFIG["device"]
MODEL_PATH = os.path.join(CONFIG["output"]["models"], "best_model.pth")

# ==== Seed ====

# Set random seed for reproducibility
torch.manual_seed(SEED)
np.random.seed(SEED)

# ==== Load Test Data ====

(test_images, test_labels) = get_mnist_data("test")

test_images = torch.tensor(test_images, dtype=torch.float32).view(-1, 1, 28, 28)
test_labels = torch.tensor(test_labels, dtype=torch.long)
test_loader = DataLoader(TensorDataset(test_images, test_labels), batch_size=BATCH_SIZE, shuffle=False)

num_test_images = len(test_images)
logging.info(f"Total test images: {num_test_images}")
print(f"Total test images: {num_test_images}")

# Count occurrences of each digit (0-9)
digit_counts = torch.bincount(test_labels, minlength=10)

# Print counts for each digit
for digit, count in enumerate(digit_counts):
    logging.info(f"\tDigit {digit}: {count} samples")
    print(f"\tDigit {digit}: {count} samples")

# ==== Load Best Model ====

model = SimpleCNN().to(DEVICE)

if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH))
    model.to(DEVICE)  # Ensure model is on the correct device
    logging.info(f"Loaded best model from {MODEL_PATH}")
    print(f"Loaded best model from {MODEL_PATH}")
else:
    print("Error: Best model file not found!")
    logging.error("Error: Best model file not found!")
    exit()

# ==== Evaluate Model on Test Data ====

def evaluate_model():
    model.eval()  # Set the model to evaluation mode

    total_loss = 0.0
    correct, total = 0, 0

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            total_loss += torch.nn.CrossEntropyLoss()(outputs, labels).item()

            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(test_loader)
    accuracy = 100.0 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%, Test Loss: {avg_loss:.4f}")

    return avg_loss, accuracy, all_labels, all_preds

# ==== Main Execution ====

if __name__ == "__main__":
    
    # Get test results
    test_loss, test_accuracy, test_labels, test_preds = evaluate_model()
    
    logging.info(f"Test Loss: {test_loss:.4f}")
    logging.info(f"Test Accuracy: {test_accuracy:.2f}%")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.2f}%")
    
    # ---- Test Confusion Matrix ----
    
    class_names = [str(i) for i in range(10)]  # Classes: 0-9 for MNIST
    
    # Plot confusion matrix
    plot_confusion_matrix(model, test_loader, DEVICE, class_names, os.path.join(CONFIG["output"]["plots"], "confusion_matrix_test.png"))
    
    # ---- Test ROC Curve ----
    
    # Plot ROC curve
    plot_roc_curve(model, test_loader, DEVICE, os.path.join(CONFIG["output"]["plots"], "roc_curve_test.png"))
    
    # ---- Classification Report ----
    
    # Generate and log classification report
    report = classification_report(test_labels, test_preds, target_names=class_names)
    logging.info("\nClassification Report:\n" + report)
    print("\nClassification Report:\n", report)
