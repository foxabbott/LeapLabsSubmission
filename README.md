# LeapLabsSubmission
Simple codebase for generative adversarial images, for LeapLabs interview process

A Python library for generating adversarial examples using Projected Gradient Descent (PGD). This tool allows you to create imperceptibly modified images that cause image classification models to misclassify them as any target class you specify.

## Features

- Generate adversarial examples using a modified PGD algorithm
- Uses ResNet18 as the base model for classification
- Configurable perturbation strength and iteration count
- Includes utilities for downloading test images and running batch tests

## Installation

1. Clone this repository
2. Install the required dependencies:
```bash
pip install torch torchvision Pillow
```

## Usage

### Downloading Test Images

To download sample images from CIFAR10:

```bash
python image_generator.py CIFAR10 10 images
```

This will download 10 images from CIFAR10 and save them to the `images` directory.

### Generating Adversarial Examples

To generate an adversarial example:

```bash
python main.py path/to/image.png target_class output_directory --epsilon 0.1 --iterations 100
```

Parameters:
- `path/to/image.png`: Path to the input image
- `target_class`: The desired target class index (0-999 for ImageNet classes)
- `output_directory`: Directory where the adversarial image will be saved
- `--epsilon`: Maximum perturbation strength (default: 0.01) (recommended: 0.1)
- `--iterations`: Number of PGD iterations (default: 10) (recommended: 100)

### Running Tests

To test the adversarial example generation across multiple images and target classes:

```bash
python test.py
```

## Technical Details

The implementation uses a modified version of PGD with two key differences from the standard algorithm:

1. Uses sigmoid-scaled gradients instead of sign gradients
2. Loss is calculated as the difference between softmax values of the current maximum class and target class

The perturbations are bounded by the epsilon parameter to ensure the changes remain imperceptible to human observers.

## Project Structure

- `main.py`: Core implementation of the PGD algorithm
- `image_generator.py`: Utility for downloading test images from CIFAR10
- `test.py`: Test suite for verifying the implementation
