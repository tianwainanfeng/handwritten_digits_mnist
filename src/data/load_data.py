from src.config_loader import CONFIG
from src.data.mnist_data_loader import load_mnist_images, load_mnist_labels

# ==== Load Configuration ====

TRAIN_IMAGES = CONFIG["data"]["raw"]["train_images"]
TEST_IMAGES = CONFIG["data"]["raw"]["test_images"]
TRAIN_LABELS = CONFIG["data"]["raw"]["train_labels"]
TEST_LABELS = CONFIG["data"]["raw"]["test_labels"]

# ==== Load MNIST Data ====

def get_mnist_data(dataset):
    if dataset not in ["train", "test"]:
        raise ValueError("data parameter must be either 'train' or 'test'")

    if dataset == "train":
        print("Loading MNIST Training Data...")
        train_images = load_mnist_images(TRAIN_IMAGES)
        train_labels = load_mnist_labels(TRAIN_LABELS)
        
        return (train_images, train_labels)

    if dataset == "test":
        print("Loading MNIST Testing Data...")
        test_images = load_mnist_images(TEST_IMAGES)
        test_labels = load_mnist_labels(TEST_LABELS)

        return (test_images, test_labels)
