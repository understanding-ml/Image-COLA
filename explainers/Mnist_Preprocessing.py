import tensorflow.keras as keras
import pandas as pd
from matplotlib import pyplot as plt
from tensorflow.keras.utils import to_categorical
from scipy.spatial.distance import euclidean
import numpy as np

class Preprocessing:
    def __init__(self, file_path, test_idx=15, xmin=-0.5, xmax=0.5):
        self.file_path = file_path
        self.test_idx = test_idx
        self.xmin = xmin
        self.xmax = xmax
        self.all_results_flat = None
        self.X = None
        self.max_p_img = None

    def load_and_preprocess_data(self):
        df = pd.read_excel(self.file_path)
        self.all_results_flat = df.values

        (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

        x_train = x_train.astype('float32') / 255
        x_test = x_test.astype('float32') / 255
        x_train = x_train.reshape(-1,784,)
        x_test = x_test.reshape(-1,784,)
        y_train = to_categorical(y_train)
        y_test = to_categorical(y_test)

        xmin, xmax = -.5, .5
        x_train = ((x_train - x_train.min()) / (x_train.max() - x_train.min())) * (xmax - xmin) + xmin
        x_test = ((x_test - x_test.min()) / (x_test.max() - x_test.min())) * (xmax - xmin) + xmin
        x_train_flat = x_train.reshape(-1, 784)  

        x_test_flat = x_test.reshape(-1, 784)  

        idx = 15
        self.X = x_test_flat[idx].reshape(1, 784)

        return self.X, self.all_results_flat

    def calculate_distances(self):
        X_flattened = self.X.flatten()
        distances = np.array([euclidean(X_flattened, img.flatten()) for img in self.all_results_flat])
        return distances

    def find_most_diverse_images(self, distances, top_n=10):
        most_diverse_indices = np.argsort(distances)[-top_n:]
        return most_diverse_indices

    def visualize_distances_with_highlights(self, distances, highlighted_indices):
        plt.figure(figsize=(12, 6))
        plt.bar(range(len(distances)), distances, color='gray')
        plt.title("Distances between X and Counterfactual Images")
        for idx in highlighted_indices:
            plt.bar(idx, distances[idx], color='red', label='Most Diverse' if idx == highlighted_indices[0] else "")
        plt.xlabel('Image Index')
        plt.ylabel('Distance')
        plt.legend(loc="upper right")
        plt.show()

    def display_most_diverse_images(self, largest_distance_images):
        fig, axes = plt.subplots(1, len(largest_distance_images), figsize=(10, 5))
        for i, idx in enumerate(largest_distance_images):
            ax = axes[i]
            ax.imshow(self.all_results_flat[idx].reshape(28, 28), cmap='gray')
            ax.axis('off')
            ax.set_title(f"Image {idx}")
        plt.tight_layout()
        plt.show()

    def preprocess_and_find_diverse_images(self, top_n=10):
        self.load_and_preprocess_data()
        distances = self.calculate_distances()
        largest_distance_images = self.find_most_diverse_images(distances, top_n=top_n)
        # self.visualize_distances_with_highlights(distances, largest_distance_images)
        # self.display_most_diverse_images(largest_distance_images)
        return self.X, self.all_results_flat[largest_distance_images], largest_distance_images
