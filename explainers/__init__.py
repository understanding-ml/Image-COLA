# Import the necessary classes
import cola
from cola import ImageCOLA

# Instantiate Preprocessing to get X and R
preprocessor = cola.Preprocessing(file_path="all_results_PN.xlsx")
X, R, indice = preprocessor.preprocess_and_find_diverse_images()
cola_instance = ImageCOLA(X, R)

# Call compute_matching on the instance to get p
p, p_max_img, total_pixel_change = cola_instance.compute_matching()

# Calculate the shap_values based on the counterfactual R
target_label = 3  
varphi = cola_instance.compute_shapley(target_label=target_label, joint_probabilities=p)

# cola_instance.plot_shap_heatmaps(varphi)

# Obtain C & final modified image
optimized_image, min_replacements, final_successful_combination = cola_instance.obtain_counterfactual_image(C=20, target_label=3)
print(f"ImageCOLA algorithm reduced modified pixel amount from {total_pixel_change} to {min_replacements}")

