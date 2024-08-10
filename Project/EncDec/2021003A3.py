import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision
import os
from torch.optim import Adam
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np
import random
import torch
import torch.nn.functional as F
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
import os
from skimage.metrics import structural_similarity as ssim,peak_signal_noise_ratio as psnr

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#----------------------------------------------------------DATASET------------------------------------------------------------

class AlteredMNIST:
    """
    dataset description:
    
    X_I_L.png
    X: {aug=[augmented], clean=[clean]}
    I: {Index range(0,60000)}
    L: {Labels range(10)}
    
    Write code to load Dataset
    """
    def __init__(self):
        self.augmented_dir = './Data/aug'
        self.clean_dir = './Data/clean'
        self.transform = transforms.Compose([
            transforms.Normalize((0.5,), (0.5,))
        ])
        # self.aug_to_clean_mapping = aug_to_clean_mapping
        
        self.augmented_images = os.listdir(self.augmented_dir)
        self.clean_images = os.listdir(self.clean_dir)

        self.aug_to_clean_mapping, self.unique_mappings = prepare_mappings()

    def __len__(self):
        return len(self.augmented_images)

    def __getitem__(self, idx):
        aug_img_name = self.augmented_images[idx]
        
        # Use the mapping to get the corresponding clean image name
        clean_img_name = self.aug_to_clean_mapping.get(aug_img_name)
        if not clean_img_name:
            raise ValueError(f"No mapping found for augmented image {aug_img_name}")

        aug_img_path = os.path.join(self.augmented_dir, aug_img_name)
        clean_img_path = os.path.join(self.clean_dir, clean_img_name)
        
        # Load images using torchvision.io.read_image
        aug_image = torchvision.io.read_image(aug_img_path)
        clean_image = torchvision.io.read_image(clean_img_path)
        digit = aug_img_name.split('_')[-1].split('.')[0]
        
        # Convert images to grayscale if they have more than 1 channel
        if aug_image.shape[0] > 1:
            aug_image = transforms.Grayscale(num_output_channels=1)(aug_image)
        if clean_image.shape[0] > 1:
            clean_image = transforms.Grayscale(num_output_channels=1)(clean_image)

        # Apply any additional transformations specified
        if self.transform:
            aug_image = self.transform(aug_image.float() / 255)  # Normalize to [0, 1]
            clean_image = self.transform(clean_image.float() / 255)  # Normalize to [0, 1]

            ### Convert to PIL Image
        # aug_image = transforms.ToPILImage()(aug_image)
        # clean_image = transforms.ToPILImage()(clean_image)
        aug_image = np.array(aug_image)
        clean_image = np.array(clean_image)

        ### Convert to tensor

        return aug_image, clean_image, torch.tensor(int(digit))



def load_images(image_directory):
    images_digit = {}
    images_digit_filename = {}
    for img_name in os.listdir(image_directory):
        _, _, digit = img_name.rsplit('_', 2)
        digit = digit.split('.')[0]
        if digit not in images_digit:
            images_digit[digit] = []
            images_digit_filename[digit] = []
        
        img_path = os.path.join(image_directory, img_name)
        img = torchvision.io.read_image(img_path)  # Load image as tensor
        # Convert to grayscale if not already
        if img.shape[0] > 1:
            img = transforms.Grayscale(num_output_channels=1)(img)
        
        images_digit[digit].append(img.flatten().numpy())  # Flatten and convert to numpy array
        images_digit_filename[digit].append(img_name)
    return images_digit, images_digit_filename

def apply_pca(images, pca):
        return pca.transform(np.array(images))

#----------------------------------------------------------Prepare Mappings------------------------------------------------------------


def prepare_mappings():
    ### Applying PCA
    clean_dir =  './Data/clean'
    aug_dir = './Data/aug'

    # Initialize containers for images and filenames
    clean_img_digit = {}
    aug_img_digit = {}
    clean_img_digit_filename = {}
    aug_img_digit_filename = {}

    # Load images
    clean_img_digit, clean_img_digit_filename = load_images(clean_dir)
    aug_img_digit, aug_img_digit_filename = load_images(aug_dir)

    # Combine clean and augmented images for PCA
    all_images_digit = {}
    for digit in clean_img_digit:
        all_images_digit[digit] = np.concatenate([clean_img_digit[digit], aug_img_digit[digit]])

    # Apply PCA
    pca = PCA(n_components=0.95)
    all_images_flattened = np.concatenate(list(all_images_digit.values()))
    pca.fit(all_images_flattened)

    # Transform images with PCA
    for digit in all_images_digit:
        clean_img_digit[digit] = apply_pca(clean_img_digit[digit], pca)
        # aug_img_digit[digit] = apply_pca(aug_img_digit[digit], pca)
    # print("PCA Applied")

    ### Dictionary to store the mappings
    aug_to_clean_mapping = {}

    representation = {}
    gmm_models = {}
    unique_mappings = {}
    for i in range(10):
        unique_mappings[i] = {}
        for j in range(40):
            unique_mappings[i][j] = set()

    for digit in clean_img_digit.keys():
        clean_images = clean_img_digit[digit]
        clean_filenames = clean_img_digit_filename[digit]
        aug_images = aug_img_digit[digit]
        aug_filenames = aug_img_digit_filename[digit]
        
        # Fit a GMM for each digit based on the number of clean images
        gmm = GaussianMixture(n_components=40, covariance_type='full', random_state=42)
        gmm.fit(clean_images)
        gmm_models[digit] = gmm
        # print(f"GMM fitted for digit {digit}")
        
        # Calculate probabilities of each clean image in each component
        clean_img_probabilities = gmm.predict_proba(clean_images)

        representation[digit] = {}
        ### Find the best representative clean image for each cluster
        for cluster_idx in range(40):
            cluster_probabilities = clean_img_probabilities[:, cluster_idx]
            most_representative_clean_img_idx = np.argmax(cluster_probabilities)
            most_representative_clean_img = clean_filenames[most_representative_clean_img_idx]
            representation[digit][cluster_idx] = most_representative_clean_img
            
        
        for idx, aug_img in enumerate(aug_images):
            ### Calculate the probabilties of the augmented image in each component using reshape

            ### Apply pca to the augmented image 
            aug_img = pca.transform(aug_img.reshape(1, -1))

            ### Calculate the probabilities of the augmented image in each component
            aug_img_probability = gmm.predict_proba(aug_img)

            # print(aug_img_probability)  
            # Identify the most likely component
            most_likely_component = np.argmax(aug_img_probability)
            
            # print(most_likely_component)
            
            # Find the clean image with the highest score within the most likely component

            
            # Use filenames to create the final mapping
            aug_filename = aug_filenames[idx]
            clean_filename = representation[digit][most_likely_component]
            aug_to_clean_mapping[aug_filename] = clean_filename
            unique_mappings[int(digit)][most_likely_component].add(clean_filename)
            
            # Optionally print the mapping
            # print(f"Augmented Image {aug_filename} mapped to Clean Image {clean_filename}")
            
            # if idx % 100 == 0:
            #     print(f"Augmented Image {idx} processed")
        
        # print(f"Digit {digit} processing complete")

    return aug_to_clean_mapping, unique_mappings

#----------------------------------------------------------RESBLOCKS------------------------------------------------------------


class ResBlock_AE(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResBlock_AE, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out
    
class ResBlock_VAE(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResBlock_VAE, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        out = self.relu(out)
        return out

class ResBlock_CVAE(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResBlock_CVAE, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # if downsample is None and (stride != 1 or in_channels != out_channels):
        #     self.downsample = nn.Sequential(
        #         nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
        #         nn.BatchNorm2d(out_channels)
        #     )
        # else:
        #     self.downsample = downsample
        self.downsample = downsample

    def forward(self, x):
        identity = self.downsample(x) if self.downsample else x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        ### Reduce out to half the size
        if(self.downsample):
            out = F.interpolate(out, scale_factor=0.5, mode='bilinear', align_corners=False)
        out += identity
        out = self.relu(out)
        return out

#----------------------------------------------------------ENCODER------------------------------------------------------------

    
class Encoder_AE(nn.Module):
    def __init__(self):
        super(Encoder_AE, self).__init__()
        self.initial = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.resblock1 = ResBlock_AE(16, 32, stride=2)
        self.resblock2 = ResBlock_AE(32, 64, stride=2)
        self.resblock3 = ResBlock_AE(64, 128, stride=2)
        # self._initialize_weights()

    def forward(self, x):
        out = F.relu(self.initial(x))
        # print(out.shape)
        out = self.resblock1(out)
        # print(out.shape)
        out = self.resblock2(out)
        # print(out.shape)
        out = self.resblock3(out)
        # print(out.shape)
        return out
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)

class Encoder_VAE(nn.Module):
    def __init__(self):
        super(Encoder_VAE, self).__init__()
        self.initial = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.resblock1 = ResBlock_VAE(16, 32, stride=2)
        self.resblock2 = ResBlock_VAE(32, 64, stride=2)
        self.resblock3 = ResBlock_VAE(64, 128, stride=2)
        self.fc_mu = nn.Linear(2048, 128)  # Assuming the output dimension is 3x3 at this point
        self.fc_logvar = nn.Linear(2048, 128)

    def forward(self, x):
        out = F.relu(self.initial(x))
        out = self.resblock1(out)
        out = self.resblock2(out)
        out = self.resblock3(out)
        out = out.view(out.size(0), -1)
        mu = self.fc_mu(out)
        logvar = self.fc_logvar(out)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
class Encoder_CVAE(nn.Module):
    def __init__(self):
        super(Encoder_CVAE, self).__init__()
        self.input_channels = 1
        self.latent_dim = 128
        self.class_dim = 10

        self.initial = nn.Sequential(
            nn.Conv2d(self.input_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

        self.downsample1 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm2d(64)
        )
        self.downsample2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm2d(128)
        )
        
        self.resblock1 = ResBlock_CVAE(32, 64, downsample=self.downsample1)
        self.resblock2 = ResBlock_CVAE(64, 128, downsample=self.downsample2)

        self.fc_mu = nn.Linear(6272, self.latent_dim)
        self.fc_logvar = nn.Linear(6272, self.latent_dim)

    def forward(self, x):
        x = self.initial(x)
        # print(x.shape)
        x = self.resblock1(x)
        # print(x.shape)
        x = self.resblock2(x)
        # print(x.shape)
        x = torch.flatten(x, start_dim=1)
        # print(x.shape)
        mu = self.fc_mu(x)
        log_var = self.fc_logvar(x)
        return mu, log_var


#----------------------------------------------------------DECODER------------------------------------------------------------


class Decoder_AE(nn.Module):
    def __init__(self):
        super(Decoder_AE, self).__init__()
        # Correctly Upsample from 4x4 to 8x8
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2, padding=0)
        self.resblock1 = ResBlock_AE(64, 64)
        # Upsample from 8x8 to 16x16
        self.up2 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2, padding=0)
        self.resblock2 = ResBlock_AE(32, 32)
        # Upsample from 16x16 to 32x32, then use a conv to adjust down to 28x28
        self.up3 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2, padding=0)
        self.resblock3 = ResBlock_AE(16, 16)
        # Adjust to 28x28: kernel size and padding chosen to reduce 32x32 to 28x28
        self.adjust = nn.Conv2d(16, 16, kernel_size=5, stride=1, padding=0)
        # Final convolution to get the desired channel size
        self.final = nn.Conv2d(16, 1, kernel_size=3, stride=1, padding=1)
        # self._initialize_weights()

    def forward(self, x):
        out = self.up1(x)  # Upsample to 8x8
        out = self.resblock1(out)
        out = self.up2(out)  # Upsample to 16x16
        out = self.resblock2(out)
        out = self.up3(out)  # Upsample to 32x32
        out = self.resblock3(out)
        out = self.adjust(out)  # Adjust size from 32x32 to 28x28
        out = self.final(out)  # Final conv to maintain spatial dimensions and adjust channels
        return out
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                torch.nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)

class Decoder_VAE(nn.Module):
    def __init__(self):
        super(Decoder_VAE, self).__init__()
        # Project latent space back to feature map
        self.fc = nn.Linear(128, 128 * 3 * 3)  # Match encoder's flattening

        # ResBlock structure is maintained, but now for upsampling
        self.resblock1 = ResBlock_VAE(128, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2, padding=0)

        self.resblock2 = ResBlock_VAE(64, 64)
        self.up2 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2, padding=0)

        self.resblock3 = ResBlock_VAE(32, 32)
        self.up3 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2, padding=0)

        # Ensure the final layer upsamples to the correct size, adding padding as necessary
        self.final_resblock = ResBlock_VAE(16, 16)
        self.final = nn.Conv2d(16, 1, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        out = self.fc(x)
        out = out.view(-1, 128, 3, 3)  # Reshape to spatial size

        # Sequentially apply ResBlocks and upsample
        out = self.resblock1(out)
        out = self.up1(out)  # 6x6
        
        out = self.resblock2(out)
        out = self.up2(out)  # 12x12

        out = self.resblock3(out)
        out = self.up3(out)  # 24x24

        # Adjust to 28x28: We use a final resblock and then a conv layer with padding
        # to ensure the output size matches the expected 28x28
        out = self.final_resblock(out)
        out = F.interpolate(out, size=(28, 28), mode='bilinear', align_corners=False)
        out = self.final(out)
        
        return out



class Decoder_CVAE(nn.Module):
    def __init__(self):
        super(Decoder_CVAE, self).__init__()
        self.output_channels = 1
        self.latent_dim = 128
        self.class_dim = 10

        self.decoder_input = nn.Linear(self.latent_dim + self.latent_dim, 128 * 7 * 7)  # Adjusting this line

        
        self.resblock1_up = ResBlock_CVAE(128, 128)
        self.upsample1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        
        self.resblock2_up = ResBlock_CVAE(64, 64)
        self.upsample2 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        
        self.final = nn.Sequential(
            nn.ConvTranspose2d(32, self.output_channels, kernel_size=3, stride=1, padding=1),
            nn.Tanh()
        )

        self.condition_embedding = nn.Embedding(self.class_dim, self.latent_dim)

    def forward(self, z, labels):
        # print(z.shape)  # Expected to be something like [batch_size, 128]
        # print(labels.shape)  # Expected to be something like [batch_size]
        
        labels_embedded = self.condition_embedding(labels.to(device))  # Embedding labels
        # print(labels_embedded.shape)  # Should now be [batch_size, 128]
        
        z_and_labels = torch.cat([z, labels_embedded], dim=1)  # Concatenating z and embedded labels
        # print(z_and_labels.shape)  # Ensure this matches the input expectation of decoder_input layer
        
        x = self.decoder_input(z_and_labels)
        x = x.view(-1, 128, 7, 7)  # Ensure this matches the expected shape after decoder_input
        
        # Follow through with the decoding process as before
        x = self.resblock1_up(x)
        x = self.upsample1(x)
        x = self.resblock2_up(x)
        x = self.upsample2(x)
        x = self.final(x)
        return x



#----------------------------------------------------------PARENT ENCODER------------------------------------------------------------


class Encoder:
    """
    Write code for Encoder ( Logits/embeddings shape must be [batch_size,channel,height,width] )
    """
    def __init__(self):
        self.encoder = Encoder_AE().to(device)

    def assign_encoder(self, loss_fn):
        ### Check instance of loss_fn
        if isinstance(loss_fn, AELossFn):
            self.encoder = Encoder_AE().to(device)
        elif isinstance(loss_fn, VAELossFn):
            self.encoder = Encoder_VAE().to(device)
        elif isinstance(loss_fn, CVAELossFn):
            self.encoder = Encoder_CVAE().to(device)

#----------------------------------------------------------PARENT DECODER------------------------------------------------------------


class Decoder:
    """
    Write code for decoder here ( Output image shape must be same as Input image shape i.e. [batch_size,1,28,28] )
    """
    def __init__(self):
        self.decoder = Decoder_AE().to(device)

    def assign_decoder(self, loss_fn):
        ### Check instance of loss_fn
        if isinstance(loss_fn, AELossFn):
            self.decoder = Decoder_AE().to(device)
        elif isinstance(loss_fn, VAELossFn):
            self.decoder = Decoder_VAE().to(device)
        elif isinstance(loss_fn, CVAELossFn):
            self.decoder = Decoder_CVAE().to(device)

#---------------------------------------------------------LOSS FUNCTIONS------------------------------------------------------------

class AELossFn:
    """
    Loss function for AutoEncoder Training Paradigm
    """
    def __init__(self):
        super(AELossFn, self).__init__()

    def forward(self, recon_x, x):
        MSE_loss = F.mse_loss(recon_x, x)
        return MSE_loss
    

class VAELossFn:
    """
    Loss function for Variational AutoEncoder Training Paradigm
    """
    
    def __init__(self):
        super(VAELossFn, self).__init__()

    def forward(self, recon_x, x, mu, log_var):
        # Reconstruction loss (MSE for simplicity, could also be BCE)
        recon_loss = F.mse_loss(recon_x, x, reduction='sum')
        
        # KL Divergence loss
        kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        
        return recon_loss + kl_loss
    
class CVAELossFn(nn.Module):
    def forward(self, recon_x, x, mu, logvar):
        # Reconstruction loss - comparing the reconstructed image to the original image
        # It is important to match the shape of recon_x and x. Ensure they are both [batch_size, channels, height, width]
        # Assuming x and recon_x are already in the correct shape [batch_size, 1, 28, 28]
        recon_loss = F.mse_loss(recon_x, x, reduction='sum')

        # KL divergence loss
        # The KL divergence measures how one probability distribution diverges from a second, expected probability distribution
        # For numerical stability, it's crucial to ensure logvar.exp() doesn't go to infinity. Adding a small epsilon can help if issues arise.
        kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        # Ensure KL divergence is non-negative
        # kl_div = torch.abs(kl_div)

        # Total loss
        loss = recon_loss + kl_div
        return loss

#----------------------------------------------------------PARAMETER SELECTOR------------------------------------------------------------


def ParameterSelector(E, D):
    """
    Write code for selecting parameters to train
    """
    encoder_params = list(E.encoder.parameters())
    decoder_params = list(D.decoder.parameters())
    
    # Combine the parameter lists
    combined_params = encoder_params + decoder_params
    
    return combined_params

# def calculate_image_metrics(reconstructed, original):
#     # Convert PyTorch tensors to NumPy arrays
#     # clean_img_np = clean_img.detach().cpu().numpy()
#     # decoded_img_np = decoded_img.detach().cpu().numpy()

#     # # skimage expects the image in the format (height, width) for grayscale images
#     # # or (height, width, channels) for RGB images. Ensure your images are correctly shaped.
#     # # If your images are single-channel (grayscale), you might need to squeeze the channel dimension:
#     # if clean_img_np.ndim == 3 and clean_img_np.shape[0] == 1:  # Assuming (channel, height, width)
#     #     clean_img_np = np.squeeze(clean_img_np, axis=0)
#     #     decoded_img_np = np.squeeze(decoded_img_np, axis=0)

#     # # Calculate SSIM and PSNR. Ensure to specify the data_range if your images are normalized
#     # ssim_value = ssim(clean_img_np, decoded_img_np, data_range=decoded_img_np.max() - decoded_img_np.min())
#     # psnr_value = psnr(clean_img_np, decoded_img_np, data_range=decoded_img_np.max() - decoded_img_np.min())
#     if isinstance(original, torch.Tensor):
#         clean_img_np = original.detach().cpu().numpy()
#     else:
#         clean_img_np = original  # Assuming it's already a numpy array

#     if isinstance(reconstructed, torch.Tensor):
#         decoded_img_np = reconstructed.detach().cpu().numpy()
#     else:
#         decoded_img_np = reconstructed

#     # Now you can safely call functions that expect numpy arrays
#     ssim_value = ssim(clean_img_np, decoded_img_np, data_range=decoded_img_np.max() - decoded_img_np.min())
#     psnr_value = psnr(clean_img_np, decoded_img_np, data_range=decoded_img_np.max() - decoded_img_np.min())
#     return ssim_value, psnr_value

def calculate_image_metrics(reconstructed, original):
        
        # if reconstructed is a tensor that requires grad, detach it
        if reconstructed.requires_grad:
            reconstructed = reconstructed.detach()
        if original.requires_grad:
            original = original.detach()
        

        # Assuming images are single-channel and on CPU for skimage compatibility
        # Ensure image tensor is on CPU and convert to numpy for SSIM
        reconstructed_np = reconstructed.squeeze().cpu().numpy()
        original_np = original.squeeze().cpu().numpy()
        
        psnr_value = peak_signal_to_noise_ratio_torch(reconstructed, original)
        ssim_value = ssim(original_np, reconstructed_np, data_range = reconstructed_np.max() - reconstructed_np.min())

        return psnr_value, ssim_value

def peak_signal_to_noise_ratio_torch(img1, img2):
    mse = F.mse_loss(img1, img2)
    max_pixel = 1.0
    psnr = 20 * torch.log10(max_pixel / torch.sqrt(mse))
    return psnr.item()

def reconstruct(img_tensor, encoder, decoder):
    with torch.no_grad():
        encoded = encoder(img_tensor.to(device))
        decoded = decoder(encoded)
    return decoded

def reparameterize(mu, log_var):
    """
    Applies the reparameterization trick: z = mu + eps * sigma.
    :param mu: Mean from the encoder's latent space.
    :param log_var: Log variance from the encoder's latent space.
    :return: Sampled z from the latent space distribution.
    """
    std = torch.exp(0.5 * log_var)
    eps = torch.randn_like(std)
    return mu + eps * std

def reconstruct_vae(input_image, encoder, decoder):
    """
    Reconstructs the input image through the encoder-decoder pathway.
    :param input_image: A tensor of shape (1, 1, 28, 28) representing the input image.
    :return: A tensor of the reconstructed image.
    """
    with torch.no_grad():
        mu, log_var = encoder(input_image.to(device))
        z = reparameterize(mu, log_var)
        recon_image = decoder(z)
    return recon_image


def calculate_similarity(train_dataloader, encoder, decoder):
    total_psnr = 0.0
    total_ssim = 0.0
    num_images = 0

    # print(augmented_images.size(0))
    ### Validation Metrics
    for augmented_images, clean_images in train_dataloader:
        augmented_images, clean_images = augmented_images.to(device), clean_images.to(device)
        
        for idx in range(augmented_images.size(0)):
            reconstructed = reconstruct(augmented_images[idx].unsqueeze(0), encoder, decoder)
            ssim_score, psnr = calculate_image_metrics(reconstructed, clean_images[idx].unsqueeze(0))
            
            total_psnr += psnr
            total_ssim += ssim_score
            num_images += 1

    average_psnr = total_psnr / num_images
    average_ssim = total_ssim / num_images
    return average_psnr, average_ssim

def calculate_sim(train_dataloader, encoder, decoder):
    total_psnr = 0.0
    total_ssim = 0.0
    num_images = 0

    for augmented_images, clean_images in train_dataloader:
        augmented_images, clean_images = augmented_images.to(device), clean_images.to(device)
        
        for idx in range(augmented_images.size(0)):
            reconstructed = reconstruct_vae(augmented_images[idx].unsqueeze(0), encoder, decoder)
            reconstructed = reconstructed.squeeze(0)
            # print(reconstructed.shape)
            # print(clean_images[idx].unsqueeze(0).shape)
            ssim_score, psnr = calculate_image_metrics(reconstructed, clean_images[idx])
            
            total_psnr += psnr
            total_ssim += ssim_score
            num_images += 1

    average_psnr = total_psnr / num_images
    average_ssim = total_ssim / num_images

#----------------------------------------------------------TRAINER CLASSES------------------------------------------------------------


class AETrainer:
    """
    Write code for training AutoEncoder here.
    
    for each 10th minibatch use only this print statement
    print(">>>>> Epoch:{}, Minibatch:{}, Loss:{}, Similarity:{}".format(epoch,minibatch,loss,similarity))
    
    for each epoch use only this print statement
    print("----- Epoch:{}, Loss:{}, Similarity:{}")
    
    After every 5 epochs make 3D TSNE plot of logits of whole data and save the image as AE_epoch_{}.png
    """
    def __init__(self, dataloader, encoder, decoder, loss_fn, optimizer, gpu):
        if gpu:
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        encoder.assign_encoder(loss_fn)
        decoder.assign_decoder(loss_fn)
        self.encoder = encoder.encoder.to(device)
        self.decoder = decoder.decoder.to(device)
        self.dataloader = dataloader
        self.epochs = 50
        self.optimizer = torch.optim.Adam(ParameterSelector(encoder, decoder), lr=0.001)
        self.loss_fn = loss_fn
        self.loss_fn = AELossFn()
        
        self.train()

    def train(self):
        for epoch in range(1, self.epochs+1):
            self.encoder.train()
            self.decoder.train()
            epoch_loss = 0.0
            total_psnr = 0.0
            total_ssim = 0.0
            total_images = 0
            for minibatch, (augmented_images, clean_images, digits) in enumerate(self.dataloader):  # Adjusted to receive pairs
                self.optimizer.zero_grad()
                batch_psnr = 0.0
                batch_ssim = 0.0
                batch_images = 0


                augmented_images = augmented_images.to(device)
                clean_images = clean_images.to(device)

                # Encode and Decode the augmented images
                encoded = self.encoder(augmented_images)
                decoded = self.decoder(encoded)

                ### Display the shapes
                # print(f"Augmented images shape: {augmented_images.shape}")
                # print(f"Encoded shape: {encoded.shape}")
                # print(f"Decoded shape: {decoded.shape}")
                # print(f"Clean images shape: {clean_images.shape}")
                
                # Calculate the loss between the decoded (reconstructed) images and the clean images
                loss = self.loss_fn.forward(decoded, clean_images)
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()
                ### Calculate similarity

                if minibatch % 10 == 0:
                    ### Calculate ssim for minibatch
                    # Calculate metrics for the batch
                    for original, reconstructed in zip(clean_images, decoded):
                        psnr, ssim = calculate_image_metrics(reconstructed, original)
                        batch_psnr += psnr
                        batch_ssim += ssim
                        batch_images += 1
                    print(">>>>> Epoch:{}, Minibatch:{}, Loss:{}, Similarity:{}".format(epoch,minibatch,loss,batch_ssim/batch_images))

            epoch_loss /= len(self.dataloader)
            for original, reconstructed in zip(clean_images, decoded):
                psnr, ssim = calculate_image_metrics(reconstructed, original)
                total_psnr += psnr
                total_ssim += ssim
                total_images += 1

            total_psnr /= total_images
            total_ssim /= total_images
            # similarity, psnr = calculate_similarity(self.dataloader)
            print("----- Epoch:{}, Loss:{}, Similarity:{}".format(epoch, epoch_loss, total_ssim))

            if (epoch) % 10 == 0:
                self.plot_embeddings(epoch)  # Modify plot_embeddings to use clean images
        ### Save the model
        torch.save(self.encoder, "encoder_AE.pt")
        torch.save(self.decoder, "decoder_AE.pt")

    
    def plot_embeddings(self, epoch):
        embeddings = []
        self.encoder.eval()
        # Assuming you have a way to fetch all clean_images for embedding plotting
        # This part needs adjustment to correctly fetch images from the DataLoader or another dataset
        for _, (augmented_images, _, _) in enumerate(self.dataloader):
            augmented_images = augmented_images.to(device)
            with torch.no_grad():
                encoded = self.encoder(augmented_images).view(augmented_images.size(0), -1)  # Adjust for actual batch size
                embeddings.append(encoded.cpu().numpy())
        embeddings = np.concatenate(embeddings, axis=0)

        tsne = TSNE(n_components=3, verbose=0, perplexity=40, n_iter=300)
        tsne_results = tsne.fit_transform(embeddings)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(tsne_results[:, 0], tsne_results[:, 1], tsne_results[:, 2], cmap='viridis')
        plt.title(f"3D TSNE - Epoch {epoch}")
        plt.savefig(f"AE_epoch_{epoch}.png")
        plt.close()


class VAETrainer:
    """
    Write code for training Variational AutoEncoder here.
    
    for each 10th minibatch use only this print statement
    print(">>>>> Epoch:{}, Minibatch:{}, Loss:{}, Similarity:{}".format(epoch,minibatch,loss,similarity))
    
    for each epoch use only this print statement
    print("----- Epoch:{}, Loss:{}, Similarity:{}")
    
    After every 5 epochs make 3D TSNE plot of logits of whole data and save the image as VAE_epoch_{}.png
    """
    
    def __init__(self, dataloader, encoder, decoder, loss_fn, optimizer, gpu):
        if gpu:
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        encoder.assign_encoder(loss_fn)
        decoder.assign_decoder(loss_fn)
        self.encoder = encoder.encoder
        self.decoder = decoder.decoder
        self.dataset_loader = dataloader
        self.epochs = 50
        self.loss_fn = loss_fn
        self.loss_fn = VAELossFn()
        self.optimizer = torch.optim.Adam(ParameterSelector(encoder, decoder), lr=0.001)
        self.train()

    def train(self):
        self.encoder.train()
        self.decoder.train()
        
        for epoch in range(1, self.epochs+1):
            total_loss = 0
            total_psnr = 0.0
            total_ssim = 0.0
            total_images = 0
            for i, (aug_images, clean_images, digits) in enumerate(self.dataset_loader):
                aug_images, clean_images = aug_images.to(device), clean_images.to(device)
                
                # Forward pass through encoder, decoder
                mu, log_var = self.encoder(aug_images)
                z = self.reparameterize(mu, log_var)
                recon_images = self.decoder(z)
                # print(f"aug_images: {aug_images.shape}")
                # print(f"clean_images: {clean_images.shape}")
                # print(f"Sampled_image: {z.shape}")
                # print(f"recon_images: {recon_images.shape}")
                # Compute loss
                loss = self.loss_fn.forward(recon_images, clean_images, mu, log_var)
                total_loss += loss.item()
                
                # Backpropagation
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                batch_psnr = 0.0
                batch_ssim = 0.0
                batch_images = 0
                
                if i % 10 == 0:
                    ### Calculate ssim for minibatch
                    # Calculate metrics for the batch
                    for original, reconstructed in zip(clean_images, recon_images):
                        psnr, ssim = calculate_image_metrics(reconstructed, original)
                        batch_psnr += psnr
                        batch_ssim += ssim
                        batch_images += 1
                    batch_psnr /= batch_images
                    batch_ssim /= batch_images
                #     similarity, psnr = calculate_sim(self.dataset_loader, self.encoder, self.decoder)
                    print(">>>>> Epoch:{}, Minibatch:{}, Loss:{}, Similarity:{}".format(epoch,i,loss,batch_ssim))
            
            avg_loss = total_loss / len(self.dataset_loader)
            for original, reconstructed in zip(clean_images, recon_images):
                psnr, ssim = calculate_image_metrics(reconstructed, original)
                total_psnr += psnr
                total_ssim += ssim
                total_images += 1

            total_psnr /= total_images
            total_ssim /= total_images
            print("----- Epoch:{}, Loss:{}, Similarity:{}".format(epoch, avg_loss, total_ssim))
            
            if epoch % 10 == 0:
                self.plot_tsne(epoch)

        ### Save the Models
        torch.save(self.encoder, "encoder_VAE.pt")
        torch.save(self.decoder, "decoder_VAE.pt")
                

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def plot_tsne(self, epoch):
        embeddings = []
        self.encoder.eval()
        # Loop through the dataset to collect embeddings
        for _, (aug_images, _, _) in enumerate(self.dataset_loader):  # Adjusted for your data structure
            aug_images = aug_images.to(device)
            with torch.no_grad():
                mu, _ = self.encoder(aug_images)  # Only use mu for embeddings
                embeddings.append(mu.view(aug_images.size(0), -1).cpu().numpy())  # Adjust view for actual batch size
        embeddings = np.concatenate(embeddings, axis=0)

        tsne = TSNE(n_components=3, verbose=0, perplexity=40, n_iter=300)
        tsne_results = tsne.fit_transform(embeddings)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(tsne_results[:, 0], tsne_results[:, 1], tsne_results[:, 2], cmap='viridis')
        plt.title(f"3D TSNE - Epoch {epoch}")
        plt.savefig(f"VAE_epoch_{epoch}.png")
        plt.close()
    

class CVAE_Trainer:
    def __init__(self, dataloader, encoder, decoder, loss_fn, optimizer):
        encoder.assign_encoder(loss_fn)
        decoder.assign_decoder(loss_fn)
        self.encoder = encoder.encoder
        self.decoder = decoder.decoder
        self.loss_fn = loss_fn
        self.loss_fn = CVAELossFn()
        self.dataloader = dataloader
        self.optimizer = optimizer
        self.optimizer = torch.optim.Adam(ParameterSelector(encoder, decoder), lr=0.001)
        self.epochs = 50
        self.train()

    def train(self):
        prev_val_ssim=0
        for epoch in range(1, self.epochs + 1):
            self.encoder.train()
            self.decoder.train()
            total_loss = 0.0
            total_train_ssim=0.0
            embeddings=[]
            labels=[]
            total_ssim = 0
            total_psnr = 0
            total_images = 0
            for i,(noisy_images, clean_images, digits) in enumerate(self.dataloader):
                noisy_images, clean_images, digits= noisy_images.to(device), clean_images.to(device), digits.to(device)
                
                self.optimizer.zero_grad()
                
                mu, logvar = self.encoder(noisy_images)
                
                z = self.reparameterize(mu, logvar)
                
                decoded_images = self.decoder(z, digits)
                if epoch % 10 == 0:
                    embeddings.extend(mu.view(mu.size(0), -1).cpu().detach().numpy())
                    labels.extend(digits.cpu().numpy())
                loss = self.loss_fn(decoded_images, clean_images, mu, logvar)
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                # batch_train_scores = [calculate_image_metrics(decoded_images[j].detach().cpu(), clean_images[j].cpu()) for j in range(clean_images.size(0))]
                # batch_train_psnr, batch_train_ssim = zip(*batch_train_scores)

                # total_train_ssim += np.mean(batch_train_ssim)
                batch_psnr = 0.0
                batch_ssim = 0.0
                batch_images = 0
                
                if i % 10 == 0:
                    ### Calculate ssim for minibatch
                    # Calculate metrics for the batch
                    for original, reconstructed in zip(clean_images, decoded_images):
                        psnr, ssim = calculate_image_metrics(reconstructed, original)
                        batch_psnr += psnr
                        batch_ssim += ssim
                        batch_images += 1
                    batch_psnr /= batch_images
                    batch_ssim /= batch_images
                #     similarity, psnr = calculate_sim(self.dataset_loader, self.encoder, self.decoder)
                    print(">>>>> Epoch:{}, Minibatch:{}, Loss:{}, Similarity:{}".format(epoch,i,loss,batch_ssim))

            avg_loss = total_loss / len(self.dataloader)
            avg_train_ssim = total_train_ssim / len(self.dataloader)

            for original, reconstructed in zip(clean_images, decoded_images):
                psnr, ssim = calculate_image_metrics(reconstructed, original)
                total_psnr += psnr
                total_ssim += ssim
                total_images += 1

            total_psnr /= total_images
            total_ssim /= total_images

            print("----- Epoch:{}, Loss:{}, Similarity:{}".format(epoch, avg_loss, total_ssim))
            if epoch % 10 == 0:
                self.plot_tsne(np.array(embeddings), np.array(labels),epoch)
        checkpoint_path = 'encoder_CVAE.pt'
        torch.save(self.encoder, checkpoint_path)
        checkpoint_path = 'decoder_CVAE.pt'
        torch.save(self.decoder, checkpoint_path)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std
    
    def plot_tsne(self, embeddings, labels, epoch):
        # Assuming embeddings and labels are already converted to numpy arrays before being passed in
        tsne = TSNE(n_components=3, verbose=0, perplexity=40, n_iter=300)
        tsne_results = tsne.fit_transform(embeddings)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Optionally, use labels to color-code the points
        scatter = ax.scatter(tsne_results[:, 0], tsne_results[:, 1], tsne_results[:, 2], c=labels, cmap='viridis')

        legend1 = ax.legend(*scatter.legend_elements(), title="Digits")
        ax.add_artist(legend1)

        plt.title(f"3D t-SNE - Epoch {epoch}")
        plt.savefig(f"CVAE_epoch_{epoch}.png")
        plt.close()

#----------------------------------------------------------TRAINED CLASSES------------------------------------------------------------

class AE_TRAINED:
    """
    Write code for loading trained Encoder-Decoder from saved checkpoints for Autoencoder paradigm here.
    use forward pass of both encoder-decoder to get output image.
    """
    def _init_(self,gpu=False):
        if(gpu):
            self.encoder=torch.load('encoder_AE.pt',map_location='cuda')
            self.decoder=torch.load('decoder_AE.pt',map_location='cuda')
            self.device='cuda'
        else:
            self.encoder=torch.load('encoder_AE',map_location='cpu')
            self.decoder=torch.load('decoder_AE.pt',map_location='cpu')
            self.device='cpu'
        self.encoder.eval()
        self.decoder.eval()

        self.transform_data = transforms.Compose([
            transforms.Grayscale(),
            transforms.Normalize((0.5,), (0.5,))
            ])


    def reconstruct(self, img_tensor):
        with torch.no_grad():
            encoded = self.encoder(img_tensor.to(device))
            decoded = self.decoder(encoded)
        return decoded

    def calculate_metrics(self, reconstructed, original):
        # Convert tensors to numpy arrays
        psnr_value = peak_signal_to_noise_ratio(reconstructed, original)
        ssim_value = structure_similarity_index(reconstructed, original)

        return psnr_value, ssim_value
    
    def calculate_image_metrics(reconstructed, original):
        
        # if reconstructed is a tensor that requires grad, detach it
        if reconstructed.requires_grad:
            reconstructed = reconstructed.detach()
        if original.requires_grad:
            original = original.detach()
        

        # Assuming images are single-channel and on CPU for skimage compatibility
        # Ensure image tensor is on CPU and convert to numpy for SSIM
        reconstructed_np = reconstructed.squeeze().cpu().numpy()
        original_np = original.squeeze().cpu().numpy()
        
        psnr_value = peak_signal_to_noise_ratio_torch(reconstructed, original)
        ssim_value = ssim(original_np, reconstructed_np, data_range = reconstructed_np.max() - reconstructed_np.min())

        return psnr_value, ssim_value

    def peak_signal_to_noise_ratio_torch(img1, img2):
        mse = F.mse_loss(img1, img2)
        max_pixel = 1.0
        psnr = 20 * torch.log10(max_pixel / torch.sqrt(mse))
        return psnr.item()

    
    

    def from_path(self, sample, original, type):
        "Compute similarity score of both 'sample' and 'original' and return in float"
        aug_image = torchvision.io.read_image(sample)
        clean_image = torchvision.io.read_image(original)

        if aug_image.shape[0] > 1:
            aug_image = transforms.Grayscale(num_output_channels=1)(aug_image)
        if clean_image.shape[0] > 1:
            clean_image = transforms.Grayscale(num_output_channels=1)(clean_image)

        if self.transform:
            aug_image = self.transform(aug_image.float() / 255)  # Normalize to [0, 1]
            clean_image = self.transform(clean_image.float() / 255)
        
        aug_image = aug_image.to(device)
        clean_image = clean_image.to(device)

        with torch.no_grad():
            # increase the dimensions of the image tensor to match the expected input shape
            decoded = self.reconstruct(aug_image.unsqueeze(0))

        psnr, ssim = self.calculate_image_metrics(decoded, clean_image)

        if type == 'psnr':
            return psnr
        elif type == 'ssim':
            return ssim
        

class VAE_TRAINED:
    """
    Write code for loading trained Encoder-Decoder from saved checkpoints for Autoencoder paradigm here.
    use forward pass of both encoder-decoder to get output image.
    """
    def _init_(self,gpu=False):
        if(gpu):
            self.encoder=torch.load('encoder_VAE.pt',map_location='cuda')
            self.decoder=torch.load('decoder_VAE.pt',map_location='cuda')
            self.device='cuda'
        else:
            self.encoder=torch.load('encoder_VAE.pt',map_location='cpu')
            self.decoder=torch.load('decoder_VAE.pt',map_location='cpu')
            self.device='cpu'
        self.encoder.eval()
        self.decoder.eval()

        self.transform_data = transforms.Compose([
            transforms.Grayscale(),
            transforms.Normalize((0.5,), (0.5,))
            ])

    def reconstruct(self, input_image):
        """
        Reconstructs the input image through the encoder-decoder pathway.
        :param input_image: A tensor of shape (1, 1, 28, 28) representing the input image.
        :return: A tensor of the reconstructed image.
        """
        with torch.no_grad():
            mu, log_var = self.encoder(input_image.to(device))
            z = self.reparameterize(mu, log_var)
            recon_image = self.decoder(z)
        return recon_image

    def reparameterize(self, mu, log_var):
        """
        Applies the reparameterization trick: z = mu + eps * sigma.
        :param mu: Mean from the encoder's latent space.
        :param log_var: Log variance from the encoder's latent space.
        :return: Sampled z from the latent space distribution.
        """
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def evaluate(self, input_image, recon_image):
        """
        Evaluates the reconstructed image using PSNR and SSIM metrics.
        :param input_image: Original input image tensor of shape (1, H, W).
        :param recon_image: Reconstructed image tensor of shape (1, H, W).
        :return: PSNR and SSIM scores.
        """
        psnr_score = peak_signal_to_noise_ratio(input_image, recon_image)
        ssim_score = structure_similarity_index(input_image, recon_image)
        return psnr_score, ssim_score
    
    def calculate_image_metrics(reconstructed, original):
        
        # if reconstructed is a tensor that requires grad, detach it
        if reconstructed.requires_grad:
            reconstructed = reconstructed.detach()
        if original.requires_grad:
            original = original.detach()
        

        # Assuming images are single-channel and on CPU for skimage compatibility
        # Ensure image tensor is on CPU and convert to numpy for SSIM
        reconstructed_np = reconstructed.squeeze().cpu().numpy()
        original_np = original.squeeze().cpu().numpy()
        
        psnr_value = peak_signal_to_noise_ratio_torch(reconstructed, original)
        ssim_value = ssim(original_np, reconstructed_np, data_range = reconstructed_np.max() - reconstructed_np.min())

        return psnr_value, ssim_value
    

    def from_path(self, sample_path, original_path, type):
        "Compute similarity score of both 'sample' and 'original' and return in float"
        aug_image = torchvision.io.read_image(sample_path)
        clean_image = torchvision.io.read_image(original_path)

        if aug_image.shape[0] > 1:
            aug_image = transforms.Grayscale(num_output_channels=1)(aug_image)
        if clean_image.shape[0] > 1:
            clean_image = transforms.Grayscale(num_output_channels=1)(clean_image)

        if self.transform:
            aug_image = self.transform(aug_image.float() / 255)  # Normalize to [0, 1]
            clean_image = self.transform(clean_image.float() / 255)
        
        aug_image = aug_image.to(device)
        clean_image = clean_image.to(device)

        with torch.no_grad():
            # increase the dimensions of the image tensor to match the expected input shape
            decoded = self.reconstruct(aug_image.unsqueeze(0))

        psnr, ssim = self.calculate_image_metrics(decoded, clean_image)

        if type == 'psnr':
            return psnr
        elif type == 'ssim':
            return ssim

#----------------------------------------------------------CVAE GENERATOR------------------------------------------------------------


class CVAE_Generator:
    def __init__(self,gpu=False):
        if(gpu):
            self.encoder=torch.load('encoder_CVAE',map_location='cuda')
            self.decoder=torch.load('decoder_CVAE',map_location='cuda')
            self.device='cuda'
        else:
            self.encoder=torch.load('encoder_CVAE',map_location='cpu')
            self.decoder=torch.load('decoder_CVAE',map_location='cpu')
            self.device='cpu'
        self.encoder.eval()
        self.decoder.eval()
        self.transform = transforms.Compose([
            transforms.Normalize((0.5,), (0.5,))  # Adjust these values based on your dataset normalization
        ])
    
    def save_image(self,digit, save_path):
        digit_tensor = torch.tensor(int(digit), dtype=torch.long)
        z = torch.randn(1,128)
        digit_tensor = digit_tensor.unsqueeze(0)

        decoded_image = self.decoder(z, digit_tensor)

        ### Display the generated image
        # plt.imshow(decoded_image.squeeze().detach().cpu().numpy(), cmap='gray')

        ### Save the image
        # plt.savefig(save_path)

        ### Save the Image
        torchvision.utils.save_image(decoded_image, save_path)

# class CVAELossFn():
#     """
#     Write code for loss function for training Conditional Variational AutoEncoder
#     """
#     pass

# class CVAE_Trainer:
#     """
#     Write code for training Conditional Variational AutoEncoder here.
    
#     for each 10th minibatch use only this print statement
#     print(">>>>> Epoch:{}, Minibatch:{}, Loss:{}, Similarity:{}".format(epoch,minibatch,loss,similarity))
    
#     for each epoch use only this print statement
#     print("----- Epoch:{}, Loss:{}, Similarity:{}")
    
#     After every 5 epochs make 3D TSNE plot of logits of whole data and save the image as CVAE_epoch_{}.png
#     """
#     pass

# class CVAE_Generator:
#     """
#     Write code for loading trained Encoder-Decoder from saved checkpoints for Conditional Variational Autoencoder paradigm here.
#     use forward pass of both encoder-decoder to get output image conditioned to the class.
#     """
    
#     def save_image(digit, save_path):
#         pass

def peak_signal_to_noise_ratio(img1, img2):
    if img1.shape[0] != 1: raise Exception("Image of shape [1,H,W] required.")
    img1, img2 = img1.to(torch.float64), img2.to(torch.float64)
    mse = img1.sub(img2).pow(2).mean()
    if mse == 0: return float("inf")
    else: return 20 * torch.log10(255.0/torch.sqrt(mse)).item()

def structure_similarity_index(img1, img2):
    if img1.shape[0] != 1: raise Exception("Image of shape [1,H,W] required.")
    # Constants
    window_size, channels = 11, 1
    K1, K2, DR = 0.01, 0.03, 255
    C1, C2 = (K1*DR)**2, (K2*DR)**2

    window = torch.randn(11)
    window = window.div(window.sum())
    window = window.unsqueeze(1).mul(window.unsqueeze(0)).unsqueeze(0).unsqueeze(0)
    
    mu1 = F.conv2d(img1, window, padding=window_size//2, groups=channels)
    mu2 = F.conv2d(img2, window, padding=window_size//2, groups=channels)
    mu12 = mu1.pow(2).mul(mu2.pow(2))

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size//2, groups=channels) - mu1.pow(2)
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size//2, groups=channels) - mu2.pow(2)
    sigma12 =  F.conv2d(img1 * img2, window, padding=window_size//2, groups=channels) - mu12


    SSIM_n = (2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)
    denom = ((mu1**2 + mu2**2 + C1) * (sigma1_sq + sigma2_sq + C2))
    return torch.clamp((1 - SSIM_n / (denom + 1e-8)), min=0.0, max=1.0).mean().item()