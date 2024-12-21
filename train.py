from fastai.vision.learner import create_body
from torchvision.models.resnet import resnet34
from fastai.vision.models.unet import DynamicUnet
import glob
import time
import numpy as np
from PIL import Image
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
from skimage.color import rgb2lab, lab2rgb
import random
import torch
from torch import nn, optim
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class GANLoss(nn.Module):
    """
    Custom loss function for GANs.
    Supports 'vanilla' GAN loss using BCEWithLogitsLoss or 'lsgan' using MSELoss.
    """
    def __init__(self, gan_mode='vanilla', real_label=1.0, fake_label=0.0):
        super().__init__()
        self.register_buffer('real_label', torch.tensor(real_label))
        self.register_buffer('fake_label', torch.tensor(fake_label))
        if gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode == 'lsgan':
            self.loss = nn.MSELoss()

    def get_labels(self, preds, target_is_real):
        """
        Generate labels for real or fake predictions.
        """
        labels = self.real_label if target_is_real else self.fake_label
        return labels.expand_as(preds)

    def __call__(self, preds, target_is_real):
        """
        Compute the loss for the given predictions and target labels.
        """
        labels = self.get_labels(preds, target_is_real)
        loss = self.loss(preds, labels)
        return loss


class PatchDiscriminator(nn.Module):
    """
    PatchGAN Discriminator used for distinguishing real and fake images.
    """
    def __init__(self, input_c, num_filters=64, n_down=3):
        super().__init__()
        model = [self.get_layers(input_c, num_filters, norm=False)]
        model += [self.get_layers(num_filters * 2 ** i, num_filters * 2 ** (i + 1), s=1 if i == (n_down - 1) else 2)
                  for i in range(n_down)]
        model += [self.get_layers(num_filters * 2 ** n_down, 1, s=1, norm=False, act=False)]

        self.model = nn.Sequential(*model)

    def get_layers(self, ni, nf, k=4, s=2, p=1, norm=True, act=True):
        """
        Helper function to create layers for the discriminator.
        """
        layers = [nn.Conv2d(ni, nf, k, s, p, bias=not norm)]
        if norm:
            layers += [nn.BatchNorm2d(nf)]
        if act:
            layers += [nn.LeakyReLU(0.2, True)]
        return nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward pass of the discriminator.
        """
        return self.model(x)


class MainModel(nn.Module):
    """
    Main model that encapsulates the generator, discriminator, and optimization steps.
    """
    def __init__(self, net_G=None, lr_G=2e-4, lr_D=2e-4, beta1=0.5, beta2=0.999, lambda_L1=100.):
        super().__init__()
        self.device = device
        self.lambda_L1 = lambda_L1

        self.net_G = net_G.to(self.device)
        self.net_D = init_model(PatchDiscriminator(input_c=3, n_down=3, num_filters=64), self.device)
        self.GANcriterion = GANLoss(gan_mode='vanilla').to(self.device)
        self.L1criterion = nn.L1Loss()
        self.opt_G = optim.Adam(self.net_G.parameters(), lr=lr_G, betas=(beta1, beta2))
        self.opt_D = optim.Adam(self.net_D.parameters(), lr=lr_D, betas=(beta1, beta2))

    def set_requires_grad(self, model, requires_grad=True):
        """
        Enable or disable gradient computation for a model.
        """
        for p in model.parameters():
            p.requires_grad = requires_grad

    def setup_input(self, data):
        """
        Prepare input data for training.
        """
        self.L = data['L'].to(self.device)
        self.ab = data['ab'].to(self.device)

    def forward(self):
        """
        Forward pass for the generator.
        """
        self.fake_color = self.net_G(self.L)

    def backward_D(self):
        """
        Compute and backpropagate the discriminator loss.
        """
        fake_image = torch.cat([self.L, self.fake_color], dim=1)
        fake_preds = self.net_D(fake_image.detach())
        self.loss_D_fake = self.GANcriterion(fake_preds, False)
        real_image = torch.cat([self.L, self.ab], dim=1)
        real_preds = self.net_D(real_image)
        self.loss_D_real = self.GANcriterion(real_preds, True)
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        self.loss_D.backward()

    def backward_G(self):
        """
        Compute and backpropagate the generator loss.
        """
        fake_image = torch.cat([self.L, self.fake_color], dim=1)
        fake_preds = self.net_D(fake_image)
        self.loss_G_GAN = self.GANcriterion(fake_preds, True)
        self.loss_G_L1 = self.L1criterion(self.fake_color, self.ab) * self.lambda_L1
        self.loss_G = self.loss_G_GAN + self.loss_G_L1
        self.loss_G.backward()

    def optimize(self):
        """
        Perform optimization for both generator and discriminator.
        """
        self.forward()
        # Update discriminator
        self.net_D.train()
        self.set_requires_grad(self.net_D, True)
        self.opt_D.zero_grad()
        self.backward_D()
        self.opt_D.step()

        # Update generator
        self.net_G.train()
        self.set_requires_grad(self.net_D, False)
        self.opt_G.zero_grad()
        self.backward_G()
        self.opt_G.step()

class AverageMeter:
    """
    Helper class to compute and store the average, current value, and sum of a metric.
    """
    def __init__(self):
        self.reset()

    def reset(self):
        """
        Reset all metrics to zero.
        """
        self.count, self.avg, self.sum = [0.] * 3

    def update(self, val, count=1):
        """
        Update the metrics with a new value.
        Args:
            val: New value to update.
            count: Number of samples for this value.
        """
        self.count += count
        self.sum += count * val
        self.avg = self.sum / self.count


def create_loss_meters():
    """
    Create AverageMeter instances for each loss to track during training.
    Returns:
        Dictionary of AverageMeter objects for each loss type.
    """
    loss_D_fake = AverageMeter()
    loss_D_real = AverageMeter()
    loss_D = AverageMeter()
    loss_G_GAN = AverageMeter()
    loss_G_L1 = AverageMeter()
    loss_G = AverageMeter()

    return {'loss_D_fake': loss_D_fake,
            'loss_D_real': loss_D_real,
            'loss_D': loss_D,
            'loss_G_GAN': loss_G_GAN,
            'loss_G_L1': loss_G_L1,
            'loss_G': loss_G}


def update_losses(model, loss_meter_dict, count):
    """
    Update the loss meters with the current loss values from the model.
    Args:
        model: MainModel object with computed losses.
        loss_meter_dict: Dictionary of AverageMeter objects for tracking losses.
        count: Number of samples in the current batch.
    """
    for loss_name, loss_meter in loss_meter_dict.items():
        loss = getattr(model, loss_name)
        loss_meter.update(loss.item(), count=count)


def lab_to_rgb(L, ab):
    """
    Convert L and ab channels to RGB.
    Args:
        L: Lightness channel, normalized to [-1, 1].
        ab: Color channels, normalized to [-1, 1].
    Returns:
        Batch of RGB images.
    """
    L = (L + 1.) * 50.  # Rescale to [0, 100]
    ab = ab * 110.      # Rescale to [-110, 110]
    Lab = torch.cat([L, ab], dim=1).permute(0, 2, 3, 1).cpu().numpy()
    rgb_imgs = []
    for img in Lab:
        img_rgb = lab2rgb(img)
        rgb_imgs.append(img_rgb)
    return np.stack(rgb_imgs, axis=0)


def visualize(model, data, save=True):
    """
    Visualize the results of the model on a batch of data.
    Args:
        model: MainModel object.
        data: Batch of input data.
        save: Whether to save the resulting visualization as an image file.
    """
    model.net_G.eval()  # Set generator to evaluation mode
    with torch.no_grad():
        model.setup_input(data)
        model.forward()
    model.net_G.train()  # Switch back to training mode
    fake_color = model.fake_color.detach()
    real_color = model.ab
    L = model.L
    fake_imgs = lab_to_rgb(L, fake_color)
    real_imgs = lab_to_rgb(L, real_color)
    mse = np.square(np.subtract(np.array(fake_imgs), np.array(real_imgs))).mean()
    print(f"Validation Error: {mse}")

    fig = plt.figure(figsize=(15, 8))
    for j in range(5):
        i = random.randint(0, len(L) - 1)
        ax = plt.subplot(3, 5, j + 1)
        ax.imshow(L[i][0].cpu(), cmap='gray')  # Grayscale input
        ax.axis("off")
        ax = plt.subplot(3, 5, j + 1 + 5)
        ax.imshow(fake_imgs[i])  # Predicted color
        ax.axis("off")
        ax = plt.subplot(3, 5, j + 1 + 10)
        ax.imshow(real_imgs[i])  # Ground truth
        ax.axis("off")
    if save:
        fig.savefig(f"results/colorization_{time.time()}.png")
        plt.close(fig)


def log_results(loss_meter_dict):
    """
    Print the average loss values tracked in the loss_meter_dict.
    Args:
        loss_meter_dict: Dictionary of AverageMeter objects for tracking losses.
    """
    for loss_name, loss_meter in loss_meter_dict.items():
        print(f"{loss_name}: {loss_meter.avg:.5f}")


def init_weights(net, init='kaiming', gain=0.02):
    """
    Initialize the weights of a model.
    Args:
        net: Model to initialize.
        init: Type of initialization ('norm', 'xavier', or 'kaiming').
        gain: Scaling factor for initialization.
    """
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and 'Conv' in classname:
            if init == 'norm':
                nn.init.normal_(m.weight.data, mean=0.0, std=gain)
            elif init == 'xavier':
                nn.init.xavier_normal_(m.weight.data, gain=gain)
            elif init == 'kaiming':
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')

            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)
        elif 'BatchNorm2d' in classname:
            nn.init.normal_(m.weight.data, 1., gain)
            nn.init.constant_(m.bias.data, 0.)

    net.apply(init_func)
    print(f"Model initialized with {init} initialization")
    return net


def init_model(model, device):
    """
    Initialize the model and apply weight initialization.
    Args:
        model: Model to initialize.
        device: Device to load the model onto (e.g., CPU or GPU).
    Returns:
        Initialized model.
    """
    model = model.to(device)
    model = init_weights(model)
    return model


class ColorizationDataset(Dataset):
    """
    Custom Dataset for loading and processing images for colorization.
    Converts images from RGB to LAB color space and normalizes them.
    """
    def __init__(self, paths, split='train'):
        if split == 'train':
            self.transforms = transforms.Compose([transforms.Resize((SIZE, SIZE), Image.BICUBIC),
                                                  transforms.RandomHorizontalFlip()])
        elif split == 'val':
            self.transforms = transforms.Resize((SIZE, SIZE), Image.BICUBIC)

        self.split = split
        self.size = SIZE
        self.paths = paths

    def __getitem__(self, idx):
        """
        Get an item from the dataset.
        Args:
            idx: Index of the image.
        Returns:
            Dictionary with 'L' (grayscale) and 'ab' (color) tensors.
        """
        img = Image.open(self.paths[idx]).convert("RGB")
        img = self.transforms(img)
        img = np.array(img)
        img_lab = rgb2lab(img).astype("float32")
        img_lab = transforms.ToTensor()(img_lab)
        L = img_lab[[0], ...] / 50. - 1.
        ab = img_lab[[1, 2], ...] / 110.

        return {'L': L, 'ab': ab}

    def __len__(self):
        """
        Return the size of the dataset.
        """
        return len(self.paths)


def make_dataloaders(batch_size=16, n_workers=1, pin_memory=True, **kwargs):
    """
    Create a DataLoader for the dataset.
    Args:
        batch_size: Number of samples per batch.
        n_workers: Number of subprocesses for data loading.
        pin_memory: Whether to pin memory for faster data transfer to GPU.
        kwargs: Additional arguments for the dataset.
    Returns:
        DataLoader object.
    """
    dataset = ColorizationDataset(**kwargs)
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=n_workers,
                            pin_memory=pin_memory)
    return dataloader

def train_model(model, train_dl, epochs, val_dl, display_every=1):
    """
    Train the MainModel with the provided data loaders.
    Args:
        model: MainModel object to train.
        train_dl: DataLoader for the training dataset.
        epochs: Number of training epochs.
        val_dl: DataLoader for the validation dataset.
        display_every: Frequency (in iterations) to log results and visualize outputs.
    """
    for e in range(epochs):
        # Create loss meters to track the losses in this epoch
        loss_meter_dict = create_loss_meters()
        i = 0
        for data in tqdm(train_dl, desc=f"Epoch {e + 1}/{epochs}"):
            # Fetch a batch from the validation dataset for visualization
            data_val = next(iter(val_dl))

            # Set up inputs and optimize the model
            model.setup_input(data)
            model.optimize()

            # Update the loss meters
            update_losses(model, loss_meter_dict, count=data['L'].size(0))
            i += 1

            # Display losses and visualization at specified intervals
            if i % display_every == 0:
                print(f"\nEpoch {e + 1}/{epochs}")
                print(f"Iteration {i}/{len(train_dl)}")
                log_results(loss_meter_dict)  # Log the current losses
                visualize(model, data_val, save=True)  # Visualize the model's outputs

        # Save the model's state after every epoch
        torch.save(model.net_G.state_dict(), f"models/model_{e}.pt")
        torch.save(model.net_D.state_dict(), f"models/disc_model_{e}.pt")


def build_res_unet(n_input=1, n_output=2, size=224):
    """
    Build a ResNet-based U-Net generator for image-to-image tasks.
    Args:
        n_input: Number of input channels (e.g., 1 for grayscale).
        n_output: Number of output channels (e.g., 2 for ab channels).
        size: Size of the output image.
    Returns:
        U-Net model based on ResNet-34 backbone.
    """
    # Create the body using a pretrained ResNet-34 and attach a Dynamic U-Net head
    body = create_body(resnet34(), pretrained=True, n_in=n_input, cut=-2)
    net_G = DynamicUnet(body, n_output, (size, size)).to(device)
    return net_G


def pretrain_generator(net_G, train_dl, opt, criterion, epochs):
    """
    Pretrain the generator with a simple L1 loss before adversarial training.
    Args:
        net_G: U-Net generator model.
        train_dl: DataLoader for the training dataset.
        opt: Optimizer for the generator.
        criterion: Loss function (e.g., L1 loss) for pretraining.
        epochs: Number of pretraining epochs.
    """
    for e in range(epochs):
        loss_meter = AverageMeter()
        for data in tqdm(train_dl, desc=f"Pretraining Epoch {e + 1}/{epochs}"):
            # Extract inputs and targets
            L, ab = data['L'].to(device), data['ab'].to(device)

            # Forward pass and loss computation
            preds = net_G(L)
            loss = criterion(preds, ab)

            # Backward pass and optimization
            opt.zero_grad()
            loss.backward()
            opt.step()

            # Update the loss meter
            loss_meter.update(loss.item(), L.size(0))

        print(f"Pretraining Epoch {e + 1}/{epochs}")
        print(f"L1 Loss: {loss_meter.avg:.5f}")


if __name__ == '__main__':
    """
    Main execution block to prepare the data, pretrain the generator, 
    and train the complete model.
    """
    # Define the path to the training images
    path = r"train\images"
    paths = glob.glob(path + "/*.jpg")

    # Shuffle the dataset and split into training and validation sets
    np.random.seed(9)
    random.shuffle(paths)
    train_paths = paths[:int(len(paths) * 0.90)]
    val_paths = paths[int(len(paths) * 0.90):]

    print(f"Number of training images: {len(train_paths)}")
    print(f"Number of validation images: {len(val_paths)}")

    # Image size for resizing
    SIZE = 224

    # Create training and validation DataLoaders
    train_dl = make_dataloaders(paths=train_paths, split='train')
    val_dl = make_dataloaders(paths=val_paths, split='val')

    # Verify the data format
    data = next(iter(train_dl))
    Ls, abs_ = data['L'], data['ab']
    print(f"L channel shape: {Ls.shape}, ab channels shape: {abs_.shape}")
    print(f"Training batches: {len(train_dl)}, Validation batches: {len(val_dl)}")

    # Step 1: Pretrain the generator with L1 loss
    print("Pretraining the generator...")
    net_G = build_res_unet(n_input=1, n_output=2, size=SIZE)
    opt = optim.Adam(net_G.parameters(), lr=1e-4)
    criterion = nn.L1Loss()
    pretrain_generator(net_G, train_dl, opt, criterion, epochs=10)

    # Save the pretrained generator
    torch.save(net_G.state_dict(), "res34-unet.pt")

    # Step 2: Load the pretrained generator and start adversarial training
    print("Starting adversarial training...")
    net_G = build_res_unet(n_input=1, n_output=2, size=SIZE)
    net_G.load_state_dict(torch.load("res34-unet.pt", map_location=device))
    model = MainModel(net_G=net_G)

    # Train the full model with adversarial loss
    train_model(model, train_dl, epochs=100, val_dl=val_dl)

    # Visualize the final outputs
    visualize(model, next(iter(val_dl)))
