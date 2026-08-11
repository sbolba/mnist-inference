import numpy as np
import matplotlib.pyplot as plt

images = np.load('models/mnist_test_images_and_labels/test_images.npy')
labels = np.load('models/mnist_test_images_and_labels/test_labels.npy')

idx = int(input("Enter the index of the image to view (0-9): "))
plt.imshow(images[idx].reshape(28, 28), cmap='gray')
plt.title(f"Label: {labels[idx]}")
plt.show()