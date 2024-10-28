# Joint_Probabilities.py

import numpy as np
import torch
import ot
from OptimalTransport import WassersteinDivergence

def lock_pixels_if_static_across_all_images(X, most_diverse_images, Thresh=0):
    """
    Locks pixels that remain unchanged across all images by setting their weights to zero.

    Args:
        X (np.array): Original image in flattened form.
        most_diverse_images (list of np.array): List of diverse images.
        Thresh (float): Threshold for pixel change.

    Returns:
        np.array: Weights for each pixel.
    """
    weights = np.ones(X.shape, dtype=float)
    static_pixels = np.all([(np.abs(X - img.reshape(X.shape)) <= Thresh) 
                            for img in most_diverse_images], axis=0)
    weights[static_pixels] = 0 
    return weights.flatten()

def compute_joint_distribution_with_weights(X, most_diverse_images, delta=0.1, Thresh=0, reg=1):
    """
    Computes the joint distribution with weights based on Wasserstein distance.

    Args:
        X (np.array): Original image in flattened form.
        most_diverse_images (list of np.array): List of diverse images.
        delta (float): Regularization parameter for Wasserstein distance.
        Thresh (float): Threshold for pixel change.
        reg (float): Regularization term.

    Returns:
        list of float: Distances based on Wasserstein distance.
    """
    ot_model = WassersteinDivergence(reg=reg)
    X_tensor = torch.tensor(X, dtype=torch.float32)
    
    weights = lock_pixels_if_static_across_all_images(X, most_diverse_images, Thresh)
    weights_tensor = torch.tensor(weights, dtype=torch.float32)

    distances = []
    for R_image in most_diverse_images:
        R_tensor = torch.tensor(R_image, dtype=torch.float32)
        dist = ot_model.distance(X_tensor, R_tensor, weights_tensor, delta)
        distances.append(dist.item())  
    return distances

def softmax(distances, temperature=0.05):
    """
    Applies the softmax function to the distances to calculate joint probabilities.

    Args:
        distances (list of float): List of distances.
        temperature (float): Temperature for scaling the softmax.

    Returns:
        np.array: Softmax probabilities.
    """
    exp_distances = np.exp(-np.array(distances) / temperature) 
    return exp_distances / np.sum(exp_distances)

def create_max_image(self, largest_distance_images_, joint_probabilities):
    """
    Finds the image with the highest joint probability.
    """
    largest_distance_images_ = np.array(largest_distance_images_)
    max_prob_index = np.argmax(joint_probabilities)
    self.max_p_img = largest_distance_images_[max_prob_index]
    return self.max_p_img
