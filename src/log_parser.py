import os
import re
import matplotlib.pyplot as plt
from src.config_loader import CONFIG

def parse_train_log(log_file):
    epochs, train_losses, train_accuracies = [], [], []
    val_losses, val_accuracies = [], []

    with open(log_file, "r") as file:
        for line in file:
            # Match training loss and accuracy
            train_match = re.search(r"Epoch (\d+): Train Loss: ([\d.]+), Accuracy: ([\d.]+)%", line)
            val_match = re.search(r"Epoch \[(\d+)/\d+\]: Validation Loss: ([\d.]+), Accuracy: ([\d.]+)", line)

            if train_match:
                epoch = int(train_match.group(1))
                train_loss = float(train_match.group(2))
                train_accuracy = float(train_match.group(3))
                epochs.append(epoch)
                train_losses.append(train_loss)
                train_accuracies.append(train_accuracy)

            if val_match:
                val_loss = float(val_match.group(2))
                val_accuracy = float(val_match.group(3))
                val_losses.append(val_loss)
                val_accuracies.append(val_accuracy)

    return epochs, train_losses, train_accuracies, val_losses, val_accuracies

if __name__ == "__main__":
    log_file = os.path.join(CONFIG["output"]["logs"] + "train.log")
    epochs, train_losses, train_accuracies, val_losses, val_accuracies = parse_train_log(log_file)
    print(f"Parsed {len(epochs)} epochs from {log_file}.")
    print("Epochs:", epochs)
    print("Train Losses:", train_losses)
    print("Train Accuracies:", train_accuracies)
    print("Validation Losses:", val_losses)
    print("Validation Accuracies:", val_accuracies)
