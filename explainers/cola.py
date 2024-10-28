from pshap import PSHAPExplainer
from Mnist_Preprocessing import Preprocessing
import tensorflow as tf
tf.get_logger().setLevel(40)  
tf.compat.v1.disable_v2_behavior()  
from OptimalTransport import WassersteinDivergence
import tensorflow.keras as keras
from Joint_Probabilities import lock_pixels_if_static_across_all_images, compute_joint_distribution_with_weights, softmax, create_max_image
from Mnist_cnn_model import load_mnist_cnn_model
import torch
import ot
from itertools import combinations
import numpy as np
import matplotlib.pyplot as plt
from CounterfactualReplacement import CounterfactualOptimizer



class ImageCOLA:

    def __init__(self, X, R, delta=0.1, reg=1):
        self.X = X
        self.R = R
        self.delta = delta
        self.reg = reg
        self.cnn = load_mnist_cnn_model()
        self.p = None
        self.varphi = None
        self.optimizer = None
        self.positive_shap_values_with_coords = None
        self.q = None
        self.max_p_img = None    

    
    def compute_matching(self):
        """
        This corresponds to Lines 1-2 of COLA, with the counterfactual image R given.
        Compute joint probability weights for the counterfactual images R and store in self.p.
        """
        distances = compute_joint_distribution_with_weights(self.X, self.R, self.delta)
        self.p = softmax(distances, temperature=0.05)

        # Reshape R images and call create_max_image
        R_reshaped = np.array([img.reshape(28, 28) for img in self.R])
        self.max_p_img = create_max_image(self, largest_distance_images_ = R_reshaped, joint_probabilities = self.p)
        c_ik = np.where(self.X.reshape(28,28) != self.max_p_img.reshape(28,28), 1, 0)
        return self.p, self.max_p_img, np.sum(c_ik)


    def compute_shapley(self, target_label, joint_probabilities):
        """
        Compute p-SHAP values for the target label.
        
        Args:
        - target_label: The target class label.
        - joint_probabilities: List of joint probabilities for each R image.
        
        Returns:
        - p-SHAP values.
        """
        pshap_explainer = PSHAPExplainer(self.cnn)
        self.varphi = pshap_explainer.shap_values(self.X, self.R, target_label, joint_probabilities)
        self.positive_shap_values_with_coords = pshap_explainer.get_positive_shap_values_with_coords(self.varphi)
        return self.varphi, self.positive_shap_values_with_coords


    def compute_pixel_values(self, largest_distance_images_, joint_probabilities):
        """
        Corresponds to Line 5 of COLA, returning the optimized joint_max_image after minimal replacements.
        """

        self.max_p_img = self.create_max_image(self, largest_distance_images_, joint_probabilities=self.p)
        joint_max_image = self.max_p_img 
        
        self.optimizer = CounterfactualOptimizer(
            model=self.cnn, 
            X=self.X, 
            R=self.R, 
            max_p_img=self.max_p_img, 
            positive_shap_values_with_coords=self.positive_shap_values_with_coords
        )
        
        optimized_image, changed_pixels, change_matrix = self.optimizer.greedy_minimal_replacement(
            joint_max_image, 
            shap_values=self.shap_values, 
            target_label=3, 
            use_shap=(self.shap_values is not None)
        )
        
        self.q = change_matrix  
        return optimized_image  


    def obtain_counterfactual_image(self, C, target_label=3):
        """
        Constructs a counterfactual image with minimal modifications to achieve the target prediction.
        
        Arguments:
        - C: Placeholder parameter for future enhancements.
        - target_label: The target label to achieve (default = 3).
        
        Returns:
        - z: Optimized counterfactual image after minimal replacements.
        """
        self.optimizer = CounterfactualOptimizer(
            model=self.cnn, 
            X=self.X, 
            R=self.R, 
            max_p_img=self.max_p_img, 
            positive_shap_values_with_coords=self.positive_shap_values_with_coords
        )

        z, min_replacements, successful_combination = self.optimizer.minimal_combination_replacement(
            target_label=target_label, plot_steps=False
        )
        
        # Final plot for the optimized image with successful replacements annotated
        plt.figure(figsize=(8, 8))
        plt.imshow(z, cmap='gray')
        plt.colorbar()
        plt.title(f'Final Optimized Counterfactual Image After {min_replacements} Replacements')
        
        for rank, (i, j) in successful_combination:
            plt.text(j, i, f'{rank+1}', color='red', fontsize=8, ha='center', va='center', fontweight='bold')
        
        plt.show()

        print("Coordinates of replaced pixels that achieved target prediction (with SHAP ranks):")
        print(successful_combination)

        return z, min_replacements,successful_combination

