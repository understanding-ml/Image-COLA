import numpy as np
import torch
import ot
import matplotlib.pyplot as plt
from itertools import combinations

class GenericImageExplainer:
    def __init__(self, model, target_label=3, delta=0.1, reg=1):
        """
        Args:
        - model: A pre-trained model used for predictions
        - target_label: Target label for generating counterfactual images
        - delta: Wasserstein distance parameter
        - reg: Regularization parameter for Wasserstein distance
        """
        self.model = model
        self.target_label = target_label
        self.delta = delta
        self.reg = reg
        self.max_p_img = None
        self.p = None
        self.varphi = None

    def compute_distances(self, X, R_images, Thresh=0):
        weights = self.lock_pixels_if_static(X, R_images, Thresh)
        distances = [self.wasserstein_distance(X, R, weights, self.delta) for R in R_images]
        return np.array(distances)

    def lock_pixels_if_static(self, X, R_images, Thresh=0):
        weights = np.ones(X.shape)
        static_pixels = np.all([(np.abs(X - img) <= Thresh) for img in R_images], axis=0)
        weights[static_pixels] = 0
        return weights

    def wasserstein_distance(self, y_s, y_t, weights, delta):
        y_s_filtered, y_t_filtered = y_s[weights > 0], y_t[weights > 0]
        proj_y_s, proj_y_t = torch.ones(len(y_s_filtered)) / len(y_s_filtered), torch.ones(len(y_t_filtered)) / len(y_t_filtered)
        trimmed_M_y = ot.dist(y_s_filtered.reshape(-1, 1), y_t_filtered.reshape(-1, 1), metric="sqeuclidean")
        trimmed_nu = ot.emd(proj_y_s, proj_y_t, trimmed_M_y)
        return np.sum(trimmed_nu * trimmed_M_y) * (1 / (1 - 2 * delta))

    def softmax(self, distances, temperature=0.05):
        exp_distances = np.exp(-distances / temperature)
        return exp_distances / np.sum(exp_distances)

    def compute_joint_probabilities(self, X, R_images):
        distances = self.compute_distances(X, R_images)
        return self.softmax(distances)

    def obtain_counterfactual_image(self, X, R_images):
        joint_probabilities = self.compute_joint_probabilities(X, R_images)
        max_prob_index = np.argmax(joint_probabilities)
        self.max_p_img = R_images[max_prob_index]
        return self.max_p_img

    def generate_minimal_replacement(self, X, R_images):
        counterfactual_img = self.obtain_counterfactual_image(X, R_images)
        successful_combination = None

        for num_replacements in range(1, len(R_images)):
            found_successful = False
            for pixel_indices in combinations(self.positive_shap_values_with_coords, num_replacements):
                temp_image = X.copy()
                for rank, (value, (i, j)) in enumerate(pixel_indices):
                    temp_image[i, j] = counterfactual_img[i, j]

                prediction = self.model.predict(temp_image.reshape(1, -1))
                if np.argmax(prediction) == self.target_label:
                    successful_combination = [(rank, (i, j)) for rank, (value, (i, j)) in enumerate(pixel_indices)]
                    return temp_image, successful_combination

        return counterfactual_img, successful_combination


def run_generic_explainer(model, X, R_images, target_label=3):
    explainer = GenericImageExplainer(model, target_label=target_label)
    optimized_image, successful_combination = explainer.generate_minimal_replacement(X, R_images)
    return optimized_image, successful_combination
