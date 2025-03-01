import struct
import numpy as np
import matplotlib.pyplot as plt

# ==== Configuration ====

# Input
filename = "mnist_raw_dataset/train-images.idx3-ubyte"

# Output
output_path = "outputs/"

# ==== Function ====

def load_mnist_images(filename):
    with open(filename, 'rb') as f:
        magic, num_images, rows, cols = struct.unpack(">IIII", f.read(16))
        assert magic == 2051, "Invalid magic number for MNIST image file!"
        
        # Read image data
        image_data = np.frombuffer(f.read(), dtype=np.uint8)
        images = image_data.reshape(num_images, rows, cols)
        
    return images

# ==== Load the images ====

images = load_mnist_images(filename)

print("Size of Images:", len(images))
print("Shape of Images:", images.shape)

# Display the first image
plt.imshow(images[0], cmap='gray')
plt.title("Digit Imagei 1")
plt.axis('off')
plt.savefig(output_path + "image_single_digit_1.png")
#plt.show() # Optional

# Display the second image
plt.imshow(images[1], cmap='gray')
plt.title("Digit Imagei 2")
plt.axis('off')
plt.savefig(output_path + "image_single_digit_2.png")
#plt.show() # Optional

# Display the first 10 images
fig, axes = plt.subplots(2, 5, figsize=(10, 5)) # Create a figure with 2 rows and 5 columns

for i, ax in enumerate(axes.flat):  # Flatten the 2D grid of subplots
    ax.imshow(images[i], cmap='gray')
    ax.set_title(f"Digit Image {i + 1}")
    ax.axis('off')  # Hide axes for better visualization

# Save the entire figure as one image
plt.savefig(output_path + "image_first_10_digits.png")
#plt.show()  # Show the canvas
