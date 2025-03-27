import struct
import numpy as np

# ==== Loader Functions ====

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
