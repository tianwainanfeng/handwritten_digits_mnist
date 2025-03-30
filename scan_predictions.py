"""
example: python scan_predictions.py -d 3 -m positive -f prediction_3_positive.png -n 10
"""
import os
import argparse
import torch
import numpy as np
from torchvision import transforms
import matplotlib.pyplot as plt

from src.config_loader import CONFIG
from src.data.load_data import get_mnist_data
from src.utils.scan_utils import scan_predictions
from src.utils.plot_utils import display_predictions
from models.simple_cnn import SimpleCNN  # Import the CNN model

# ==== Main Function ====

def main(digit, num_samples, filename, mode):

    # ==== Load Configuration ====
    
    DEVICE = CONFIG["device"]
    BATCH_SIZE = CONFIG["training"]["batch_size"]
    MODEL_PATH = os.path.join(CONFIG["output"]["models"], "best_model.pth")
    OUTPUT_PLOT_PATH = CONFIG["output"]["plots"]
    
    # ==== Load MNIST Data ====
    
    (test_images, test_labels) = get_mnist_data("test")
    
    test_images = torch.tensor(test_images, dtype=torch.float32).view(-1, 1, 28, 28)
    test_labels = torch.tensor(test_labels, dtype=torch.long)
    
    # ==== Load Model ====
    
    model = SimpleCNN()

    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH))
        model.to(DEVICE)  # Ensure model is on the correct device
        print(f"Loaded best model from {MODEL_PATH}")
    else:
        print("Error: Best model file not found!")
        exit()
    
    model.eval()

    # ==== Run scanning ====

    """ Run scanning for correctly and incorrectly predicted samples of a given digit. """
    correct_samples, incorrect_samples, correct_labels, incorrect_labels, correct_preds, incorrect_preds = scan_predictions(model, test_images, test_labels, DEVICE, digit, num_samples)

    if mode == "positive" and not correct_samples:
        print(f"No correctly classified samples found for digit {digit}. Try another digit.")
        return
    if mode == "negative" and not incorrect_samples:
        print(f"No misclassified samples found for digit {digit}. Try another digit.")
        return
    plot_path = os.path.join(OUTPUT_PLOT_PATH, filename)
    display_predictions(correct_samples, incorrect_samples, correct_labels, incorrect_labels, correct_preds, incorrect_preds, digit, plot_path, mode)

# ==== Main Execution ====

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scan and visualize MNIST predictions.")
    parser.add_argument("-d", "--digit", type=int, default=0, help="Digit to scan (0-9).")
    parser.add_argument("-n", "--num_samples", type=int, default=5, help="Number of samples to display.")
    parser.add_argument("-f", "--filename", type=str, default="prediction.png", help="Choose a filename.")
    parser.add_argument("-m", "--mode", type=str, choices=["positive", "negative", "both"], default="both",
                        help="Choose to display 'positive' (correct predictions), 'negative' (misclassified), or 'both'.")

    args = parser.parse_args()
    main(args.digit, args.num_samples, args.filename, args.mode)

