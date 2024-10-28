# CounterfactualReplacement.py

import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations

class CounterfactualOptimizer:
    def __init__(self, model, X, R, max_p_img, positive_shap_values_with_coords):
        self.model = model
        self.X = X
        self.R = R
        self.max_p_img = max_p_img
        self.positive_shap_values_with_coords = positive_shap_values_with_coords

    def greedy_minimal_replacement(self, joint_max_image, shap_values=None, target_label=3, use_shap=True):
        """
        Perform a greedy minimal pixel replacement to achieve the target prediction.

        Args:
            joint_max_image: The counterfactual image to start with.
            shap_values: Array of SHAP values, only required if `use_shap=True`.
            target_label: The target label to achieve.
            use_shap: Boolean indicating if SHAP values should prioritize replacements.

        Returns:
            Optimized image after minimal replacements, count of changed pixels, and change matrix.
        """
        c_ik = np.where(self.X.reshape(28, 28) != joint_max_image.reshape(28, 28), 1, 0)
        PN_optimized = joint_max_image.copy()
        X_reshaped = self.X.reshape(28, 28)

        replacement_coords = (
            [(i, j) for i in range(28) for j in range(28) if c_ik[i, j] == 1]
            if not (use_shap and shap_values is not None)
            else [(idx // 28, idx % 28) for idx in np.argsort(-shap_values.flatten()) if c_ik[idx // 28, idx % 28] == 1]
        )

        for i, j in replacement_coords:
            PN_temp = PN_optimized.copy()
            PN_temp[i, j] = X_reshaped[i, j]
            prediction_temp = self.cnn.predict(PN_temp.reshape(1, 784))
            if prediction_temp.argmax() == target_label:
                PN_optimized[i, j] = X_reshaped[i, j]

        c_ik_new = np.where(X_reshaped != PN_optimized.reshape(28, 28), 1, 0)
        changed_pixels = np.sum(c_ik_new)

        # Display the final optimized image
        plt.imshow(PN_optimized.reshape(28, 28), cmap='gray')
        plt.title('Optimized Pertinent Negative Image')
        plt.axis('off')
        plt.show()

        # Display the change matrix
        plt.imshow(c_ik_new, cmap='gray')
        plt.title('Change Matrix (c_ik)')
        plt.axis('off')
        plt.show()

        return PN_optimized, changed_pixels, c_ik_new


    def minimal_combination_replacement(self, target_label=3, plot_steps=True):
        """
        Finds the minimal number of pixel replacements required to change the prediction to the target label.

        Args:
        - target_label: The target label to check (default = 3).
        - plot_steps: Boolean flag for whether to plot intermediate successful images (default = True).

        Returns:
        - X_modified: Modified image after replacement.
        - min_replacements: The minimal number of pixels replaced to achieve the target prediction.
        - successful_combination: List of pixel coordinates that achieved the target prediction.
        """
        max_pixels = len(self.positive_shap_values_with_coords)
        final_successful_combination = None
        X_modified = None  # To store the last successful image
        min_replacements = max_pixels  # Track the smallest number of replacements needed

        # Iterate over decreasing number of replacements (from max_pixels down to 1)
        for num_replacements in range(max_pixels, 0, -1):
            print(f"\nTesting {num_replacements} replacements")
            found_successful = False  # Track success within each C level

            for pixel_indices in combinations(self.positive_shap_values_with_coords, num_replacements):
                temp_image = self.X.reshape(28, 28).copy()

                # Apply replacements based on positive SHAP values
                for rank, (value, (i, j)) in enumerate(pixel_indices):
                    temp_image[i, j] = self.max_p_img[i, j]

                # Predict with the modified image
                prediction = self.model.predict(temp_image.reshape(1, 784))
                predicted_class = np.argmax(prediction)

                # Record the first successful prediction per C and prepare final successful plot
                if predicted_class == target_label:
                    found_successful = True
                    final_successful_combination = [(rank, (i, j)) for rank, (value, (i, j)) in enumerate(pixel_indices)]
                    X_modified = temp_image.copy()
                    min_replacements = num_replacements  # Update with the current replacement count
                    print(f"Successful with {num_replacements} replacements.")

                    # Plot only if `plot_steps` is True, and it's the first success for this C value
                    if plot_steps:
                        plt.figure(figsize=(6, 6))
                        plt.imshow(temp_image, cmap='gray')
                        plt.title(f'Pixel Replacement Step with {num_replacements} Replacements')
                        plt.colorbar()
                        for rank, (_, (i, j)) in enumerate(pixel_indices):
                            plt.text(j, i, f'{rank+1}', color='red', fontsize=8, ha='center', va='center', fontweight='bold')
                        plt.show()
                    break  # Stop after finding the first successful prediction for this `C`

            # If a successful modification was found for this `C`, proceed to the next smaller C value
            if found_successful:
                continue
            # Stop checking further if no success was found at all after testing all C values
            else:
                break

        # Final plot if no intermediate steps were plotted
        if not plot_steps and X_modified is not None:
            plt.figure(figsize=(8, 8))
            plt.imshow(X_modified, cmap='gray')
            plt.colorbar()
            plt.title(f'Final Modified Image After {min_replacements} Replacements')

            for rank, (i, j) in final_successful_combination:
                plt.text(j, i, f'{rank+1}', color='red', fontsize=8, ha='center', va='center', fontweight='bold')

            plt.show()

        print("Coordinates of replaced pixels that achieved target prediction (with SHAP ranks):")
        print(final_successful_combination)

        # Return the optimized image with the minimal replacements
        return X_modified, min_replacements, final_successful_combination