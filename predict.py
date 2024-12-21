import torch
import os
from PIL import Image
from fastai.vision.learner import create_body
from torchvision.models.resnet import resnet34
from fastai.vision.models.unet import DynamicUnet
from skimage.color import rgb2lab, lab2rgb
from torchvision import transforms
import numpy as np
import cv2

# Convert LAB color space to RGB
def lab_to_rgb(L, ab):
    """
    Converts LAB color space images to RGB.
    Args:
        L (torch.Tensor): L channel tensor with shape (B, 1, H, W)
        ab (torch.Tensor): AB channels tensor with shape (B, 2, H, W)
    Returns:
        img_rgb (numpy.ndarray): RGB image in numpy format (H, W, 3)
    """
    L = (L + 1.) * 50.  # Scale L channel back to [0, 100]
    ab = ab * 110.  # Scale AB channels back to [-110, 110]

    # Combine L and AB channels and convert to numpy for processing
    Lab = torch.cat([L, ab], dim=1).permute(0, 2, 3, 1).cpu().detach().numpy()

    # Convert LAB to RGB for the first image in the batch
    img_rgb = lab2rgb(Lab[0])
    return img_rgb


# Set device to GPU if available, else CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Build U-Net model with a ResNet34 backbone
def build_res_unet(n_input=1, n_output=2, size=224):
    """
    Creates a U-Net model using ResNet34 as the backbone.
    Args:
        n_input (int): Number of input channels (e.g., 1 for grayscale)
        n_output (int): Number of output channels (e.g., 2 for AB channels)
        size (int): Image size (assumes square images)
    Returns:
        net_G (torch.nn.Module): U-Net model
    """
    # Create the backbone with pretrained weights
    body = create_body(resnet34(), pretrained=True, n_in=n_input, cut=-2)

    # Create the U-Net model
    net_G = DynamicUnet(body, n_output, (size, size)).to(device)
    return net_G


# Load the pre-trained model weights
net_G = build_res_unet(n_input=1, n_output=2, size=224)
model_path = "models_gen/best.pt"
net_G.load_state_dict(torch.load(model_path, map_location=device))

# Define paths
input_folder = r"student_dataset\test_color\images"
output_folder = r"student_dataset\test_color\pred_imgs"

# Ensure the output folder exists
os.makedirs(output_folder, exist_ok=True)

# Initialize list to store predictions
all_preds = []

# Process each image in the input folder
for filename in os.listdir(input_folder):
    # Load and preprocess the image
    img_path = os.path.join(input_folder, filename)
    img = Image.open(img_path).convert("RGB")  # Ensure the image is in RGB mode
    img = img.resize((224, 224), Image.BICUBIC)
    img = np.array(img)

    # Convert grayscale to RGB if needed
    if len(img.shape) == 2 or img.shape[2] == 1:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    # Convert RGB to LAB color space
    img_lab = rgb2lab(img).astype("float32")
    img_lab = transforms.ToTensor()(img_lab)  # Convert to tensor

    # Normalize L and AB channels to range [-1, 1]
    L = img_lab[[0], ...] / 50. - 1.
    ab = img_lab[[1, 2], ...] / 110.

    # Prepare L channel for model input
    L = torch.tensor(L).unsqueeze(0).to(device)

    # Predict AB channels
    prd = net_G(L)

    # Convert LAB back to RGB
    fake_img = lab_to_rgb(L, prd)
    fake_img = (fake_img * 255).astype(np.uint8)  # Scale to [0, 255]

    # Save the predicted image
    output_path = os.path.join(output_folder, filename)
    Image.fromarray(fake_img).save(output_path)

    # Store predictions
    all_preds.append(fake_img)

# Save all predictions as a numpy array
all_preds = np.array(all_preds)
output_array_path = "prediction.npy"
np.save(output_array_path, all_preds)

print(f"All predictions saved to {output_array_path}. Shape: {all_preds.shape}")
