import numpy as np
import matplotlib.pyplot as plt

class PSHAPExplainer:
    def __init__(self, cnn):
        """
        Initializes the PSHAPExplainer with a given model.
        
        Args:
        - cnn: A pre-trained model used to calculate predictions.
        """
        self.cnn = cnn  

    def shap_values(self, X, R_images, target_label, joint_probabilities):
        """
        Calculates p-SHAP values for the given image X, counterfactual images R, 
        and joint probabilities.
        
        Args:
        - X: The original image in flattened form (e.g., 784 for a 28x28 image).
        - R_images: List of counterfactual images (also flattened).
        - target_label: The target class label for which SHAP values are calculated.
        - joint_probabilities: Array of joint probabilities for each counterfactual.
        
        Returns:
        - p-SHAP values as a 2D array matching the image shape (28x28).
        """
        # Reshape X and R images to 28x28 format
        X_reshaped = X.reshape(28, 28)
        R_images = np.array(R_images).reshape(-1, 28, 28)
        num_counterfactuals = R_images.shape[0]

        joint_probabilities /= np.sum(joint_probabilities)  

        shap_values = np.zeros((28, 28))

        for k in range(num_counterfactuals):
            R_image_reshaped = R_images[k]

            prob_with = self.cnn.predict(R_image_reshaped.reshape(1, 784))[0][target_label]

            for i in range(28):
                for j in range(28):
                    
                    perturbed_image = R_image_reshaped.copy()
                    perturbed_image[i, j] = X_reshaped[i, j]


                    prob_without = self.cnn.predict(perturbed_image.reshape(1, 784))[0][target_label]


                    marginal_contribution = (prob_with - prob_without) * joint_probabilities[k]
                    shap_values[i, j] += marginal_contribution


        total_shap_value = np.sum(shap_values)
        shap_values /= (total_shap_value if total_shap_value != 0 else 1e-10)

        return shap_values
    def get_positive_shap_values_with_coords(self, p_shap_values, num_values=20):
        """
        Extracts the top positive SHAP values and their coordinates.
        
        Args:
        - p_shap_values: SHAP values array (28x28).
        - num_values: Number of top positive values to retrieve.
        
        Returns:
        - List of tuples (value, (row, col)) for the top positive SHAP values.
        """
        masked_p_shap_values = np.copy(p_shap_values)
        masked_p_shap_values[masked_p_shap_values <= 0] = np.nan

        mean_val = np.nanmean(masked_p_shap_values)
        std_val = np.nanstd(masked_p_shap_values)
        normalized_shap_values = (masked_p_shap_values - mean_val) / std_val

        sorted_shap_values = np.sort(normalized_shap_values, axis=None)[::-1]
        sorted_indices = np.argsort(normalized_shap_values, axis=None)[::-1]
        
        coordinates = np.unravel_index(sorted_indices, (28, 28))
        sorted_shap_with_coords = list(zip(sorted_shap_values, zip(coordinates[0], coordinates[1])))
        
        positive_shap_values_with_coords = [
            (value, coord) for value, coord in sorted_shap_with_coords if value > 0
        ][:num_values]
        
        return positive_shap_values_with_coords

    def plot_shap_heatmaps(self, p_shap_values):
        """
        Plots heatmaps of the p-SHAP values and their normalized form.
        
        Args:
        - p_shap_values: SHAP values array (28x28).
        """
        masked_p_shap_values = np.copy(p_shap_values)
        masked_p_shap_values[masked_p_shap_values <= 0] = np.nan
        masked_p_shap_values = np.nan_to_num(masked_p_shap_values, nan=0)
        
        mean_val = np.nanmean(masked_p_shap_values)
        std_val = np.nanstd(masked_p_shap_values)
        normalized_shap_values = (masked_p_shap_values - mean_val) / std_val

        fig, axs = plt.subplots(1, 2, figsize=(12, 6))

        im1 = axs[0].imshow(p_shap_values, cmap='hot')
        fig.colorbar(im1, ax=axs[0], orientation='vertical')
        axs[0].set_title('p-SHAP Values for Optimized Image (Weighted by Joint Probabilities)')

        im2 = axs[1].imshow(normalized_shap_values, cmap='hot')
        fig.colorbar(im2, ax=axs[1], orientation='vertical')
        axs[1].set_title('Standardized Weighted p-SHAP Values')

        plt.tight_layout()
        plt.show()
