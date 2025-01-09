import argparse
import torchvision.models as models
import torchvision.transforms as transforms
import torch
import os
from PIL import Image

class BasicPGD:
    def __init__(self, device=None):
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = models.resnet18(pretrained=True)
        self.model.eval()
        
        # From Pytorch website
        normalization_mean = [0.485, 0.456, 0.406]
        normalization_std = [0.229, 0.224, 0.225]
        
        self.preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=normalization_mean, std=normalization_std),
        ])

        # To convert back to image
        self.unnormalize = transforms.Normalize(
            mean=[-i / j for i, j in zip(normalization_mean, normalization_std)],
            std=[1 / i for i in normalization_std]
        )

        # For bounding generated tensors
        self.min_clamp = torch.tensor([-i / j for i, j in zip(normalization_mean, normalization_std)]).view(1, 3, 1, 1).to(self.device)
        self.max_clamp = torch.tensor([(1 - i) / j for i, j in zip(normalization_mean, normalization_std)]).view(1, 3, 1, 1).to(self.device)


    def generate(self, image_path, target_class, output_dir="adv_images", epsilon=0.01, iterations=10):
        # Load and preprocess image
        original_image = Image.open(image_path).convert("RGB")
        input_tensor = self.preprocess(original_image).unsqueeze(0).to(self.device)
        input_tensor.requires_grad = True
        original_image = input_tensor.clone().detach()

        # Get the target class as a tensor
        target = torch.tensor([target_class]).to(self.device)

        for i in range(iterations):

            # Forward pass
            output = self.model(input_tensor)

            # Get the predicted class
            predicted_class = torch.argmax(output, dim=1)
            print(f"Iteration {i+1}: Predicted class is {predicted_class.item()}")

            if predicted_class.item() == target:
                # PGD terminates when the modal class is the target
                print(f"Success after {i} PGD iterations")
                break

            # Calculate loss: difference between loss for the current predicted class and the target class
            # Idea: nudge the noise towards region where logits are lower for predicted class and higher for target class
            loss_predicted = torch.nn.CrossEntropyLoss()(output, predicted_class)
            loss_target = -torch.nn.CrossEntropyLoss()(output, target)
            loss = loss_predicted + loss_target

            target_prob = torch.nn.functional.softmax(output, dim=1)[0, target_class].item()
            print(f"Iteration {i+1}: Probability of target class {target_class} = {target_prob}")

            # Backward pass
            self.model.zero_grad()
            loss.backward()

            # Add adversarial noise
            noise = epsilon * input_tensor.grad.sign()
            input_tensor = input_tensor + noise

            # Clamp to normalized range
            delta = torch.clamp(input_tensor - original_image, min=-epsilon, max=epsilon)
            input_tensor = torch.clamp(original_image + delta, min=self.min_clamp, max=self.max_clamp).detach().requires_grad_(True)

        # Convert back to image
        adversarial_image = input_tensor.squeeze(0).cpu()
        adversarial_image = self.unnormalize(adversarial_image).clip(0, 1)
        adversarial_image = transforms.ToPILImage()(adversarial_image)

        # Ensure outdir exists
        os.makedirs(output_dir, exist_ok=True)

        # Save adversarial image
        image_num = image_path.split('.')[0].split('_')[-1]        
        output_path = os.path.join(output_dir, f"image_{image_num}_target_{target_class}.png")
        adversarial_image.save(output_path)
        print(f"Adv imaged saved to {output_path}...")
        

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("image_path", type=str, help="Path to input image")
    parser.add_argument("target_class", type=int, help="Target class index")
    parser.add_argument("output_dir", type=str, help="Save path for adversarial image.")
    parser.add_argument("--epsilon", type=float, default=0.01, help="Max strength of adversarial noise.")
    parser.add_argument("--iterations", type=int, default=10, help="Number of PGD iterations.")

    args = parser.parse_args()

    generator = BasicPGD()
    generator.generate(args.image_path, args.target_class, args.output_dir, args.epsilon, args.iterations)
