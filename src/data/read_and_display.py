import os
import logging
import matplotlib.pyplot as plt
from collections import Counter
from src.config_loader import CONFIG
from src.data.load_data import get_mnist_data

# ==== Load Configuration ====

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

    # Load MNIST data
    (train_images, train_labels) = get_mnist_data("train")
    (test_images, test_labels) = get_mnist_data("test")

    # images
    log_and_print(f"Total Training Images: {len(train_images)}")
    log_and_print(f"Total Testing Images: {len(test_images)}")

    log_and_print("\nDisplaying and Saving Sample Images...")
    display_and_save_images(train_images, num_images=10, output_prefix="digit_image")
    
    # labels
    count_labels(train_labels, "Training Set")
    count_labels(test_labels, "Test Set")

    log_and_print("\nProcessing Completed Successfully!")
