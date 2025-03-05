import os
import struct
import logging
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from src.config_loader import CONFIG

# ==== Load Configuration ====

TRAIN_IMAGES = CONFIG["data"]["raw"]["train_images"]
TEST_IMAGES = CONFIG["data"]["raw"]["test_images"]
TRAIN_LABELS = CONFIG["data"]["raw"]["train_labels"]
TEST_LABELS = CONFIG["data"]["raw"]["test_labels"]

OUTPUT_PATH_IMAGES = CONFIG["output"]["images"]
OUTPUT_PATH_LOGS = CONFIG["output"]["logs"]

LOG_FILE_PATH = os.path.join(OUTPUT_PATH_LOGS, "read_and_display.log")

# Ensure output directories exist
os.makedirs(OUTPUT_PATH_IMAGES, exist_ok=True)
os.makedirs(OUTPUT_PATH_LOGS, exist_ok=True)

# ==== Setup Logging ====

logging.basicConfig(
    filename=LOG_FILE_PATH,
    filemode="w", # Overwrite log file each time, use "a" for appending
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO
)

def log_and_print(message):
    """Prints to console and logs to file."""
    print(message)
    logging.info(message)

# ==== Function ====

def load_mnist_images(filename):
    """Loads MNIST images from IDX file format."""
    with open(filename, 'rb') as f:
        magic, num_images, rows, cols = struct.unpack(">IIII", f.read(16))
        assert magic == 2051, "Invalid magic number for MNIST image file!"
        
        # Read image data
        image_data = np.frombuffer(f.read(), dtype=np.uint8)
        images = image_data.reshape(num_images, rows, cols)
        
    return images

def load_mnist_labels(filename):
    """Loads MNIST labels from IDX file format."""
    with open(filename, "rb") as f:
        magic, num_items = struct.unpack(">II", f.read(8))
        assert magic == 2049, "Invalid magic numbr for MNIST label file!"
        
        # Read lablel data
        labels = np.frombuffer(f.read(), dtype=np.uint8)

    return labels

def display_and_save_images(images, num_images=10, output_prefix="digit_image"):
    """Displays and saves the first few images."""
    for i in range(min(num_images, len(images))):
        if i > 3:
            continue
        plt.imshow(images[i], cmap='gray')
        plt.title(f"Digit Image {i + 1}")
        plt.axis('off')
        save_path = os.path.join(OUTPUT_PATH_IMAGES, f"{output_prefix}_single_{i + 1}.png")
        plt.savefig(save_path)
        print(f"Saved: {save_path}")

    # Display multiple images in a grid
    fig, axes = plt.subplots(2, 5, figsize=(10, 5))
    for i, ax in enumerate(axes.flat[:num_images]):
        ax.imshow(images[i], cmap='gray')
        ax.set_title(f"Digit {i + 1}")
        ax.axis('off')

    grid_save_path = os.path.join(OUTPUT_PATH_IMAGES, f"{output_prefix}_first_{num_images}.png")
    plt.savefig(grid_save_path)
    print(f"Saved: {grid_save_path}")

def count_labels(labels, dataset_name="Dataset"):
    """Counts and prints label statistics and logs output."""
    label_counts = Counter(labels)
    log_and_print(f"\n{dataset_name} Label Counts:")
    for digit in range(10):
        log_and_print(f"Digit {digit}: {label_counts[digit]} images")

# ==== Main Execution ====

if __name__ == "__main__":

    # Load images
    log_and_print("\nLoading MNIST Training Images...")
    train_images = load_mnist_images(TRAIN_IMAGES)
    log_and_print(f"Total Training Images: {len(train_images)}")

    log_and_print("\nLoading MNIST Testing Images...")
    test_images = load_mnist_images(TEST_IMAGES)
    log_and_print(f"Total Testing Images: {len(test_images)}")

    log_and_print("\nDisplaying and Saving Sample Images...")
    display_and_save_images(train_images, num_images=10, output_prefix="digit_image")
    
    # Load labels
    log_and_print("\nLoading MNIST Training Labels...")
    train_labels = load_mnist_labels(TRAIN_LABELS)

    log_and_print("\nLoading MNIST Testing Labels...")
    test_labels = load_mnist_labels(TEST_LABELS)
    
    count_labels(train_labels, "Training Set")
    count_labels(test_labels, "Test Set")

    log_and_print("\nProcessing Completed Successfully!")
