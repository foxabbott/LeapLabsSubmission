import torch
from main import BasicPGD
from PIL import Image

"""
Messy testing tools - ran out of time, would otherwise use Python's unittest framework
"""

def run_tests(num_images, target_index_list, epsilon=0.01, iterations=200):
    """
    Tests PGD code across multiple images and target classes.
    
    For each image-target pair:
    1. Generates an adversarial example using PGD
    2. Checks if the attack successfully fooled the model
    3. Verifies if the perturbation stays within the epsilon bound
    
    Returns:
        tuple: Contains two dictionaries:
            1. success_dict: Records if attack succeeded
            2. epsilon_bounded_dict: Records if perturbation stayed within epsilon bound
            
            Both have structure:
            {
                image_index: {
                    target_class: bool
                }
            }
    """

    success_dict = {}
    epsilon_bounded_dict = {}
    adv_dir = "adv_images"
    for image_index in range(1, num_images + 1):
        success_dict[image_index] = {}
        epsilon_bounded_dict[image_index] = {}
        image_path = f"images/image_{image_index}.png"
        for target_class in target_index_list:
            print(f"Building adversarial example for image {image_index} with target class {target_class}.")
            generator = BasicPGD()
            success = generator.generate(image_path, target_class, "adv_images", epsilon, iterations)
            success_dict[image_index][target_class] = success

            original = generator.preprocess(Image.open(image_path))
            adversarial = generator.preprocess(
                Image.open(f"{adv_dir}/image_{image_index}_target_{target_class}.png")
            )
            
            max_diff = torch.max(torch.abs(original - adversarial))
            epsilon_bounded_dict[image_index][target_class] = (max_diff < epsilon + 1e-5).item()

    return success_dict, epsilon_bounded_dict

if __name__ == "__main__":
    success, epsilon_bounded = run_tests(10, [30, 300, 500, 740], epsilon=0.1)
    import pdb; pdb.set_trace()