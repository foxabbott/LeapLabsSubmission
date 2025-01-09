import argparse
import os
from torchvision import datasets, transforms
from torchvision.transforms import ToPILImage

def download_images(dataset_name="CIFAR10", num_images=10, output_dir="images"):
    """
    Download a specified number of images from CIFAR dataset, save as png files.

    Args:
        dataset_name (str): Name of the dataset
        num_images (int): Number of images to download.
        output_dir (str): Directory to save the images.
    """

    # Define transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize to match model input size
        transforms.ToTensor(),
    ])

    # Select dataset
    if dataset_name == "CIFAR10":
        dataset = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)
    else:
        # Raise error for now if not CIFAR10
        raise ValueError(f"Dataset {dataset_name} is not supported.")

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Save images
    to_pil = ToPILImage()
    for i in range(min(num_images, len(dataset))):
        image, label = dataset[i]
        image = to_pil(image)
        image_path = os.path.join(output_dir, f"image_{i+1}.png")
        image.save(image_path)
        print(f"Saved {image_path}")

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_name", type=str, default="CIFAR10", help="Dataset name. Will bug if not CIFAR10")
    parser.add_argument("num_images", type=int, default=100, help="Total number of images to save")
    parser.add_argument("output_dir", type=str, default="images", help="Save directory for images.")

    args = parser.parse_args()
    download_images(dataset_name=args.dataset_name, num_images=args.num_images, output_dir=args.output_dir)
