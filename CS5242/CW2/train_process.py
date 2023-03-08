import torch
import torch.nn as nn
import torch.optim as optim
import os
from torch.utils import data
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np
from torchvision.transforms import transforms
from gan_models import discriminator, generator
import wandb
import torchvision.utils
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

epochs = 1000
lr = 1e-4
batch_size = 16
loss = nn.BCELoss()

# Model
G = generator().to(device)
D = discriminator().to(device)

G_optimizer = optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
D_optimizer = optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))

# Transform
transform = transforms.Compose([transforms.ToTensor()])


# Load data
class EmojiSet(data.Dataset):
    def __init__(self, root):
        imgs = os.listdir(root)
        self.imgs = [os.path.join(root, k) for k in imgs]
        self.transforms = transform

    def __getitem__(self, index):
        img_path = self.imgs[index]
        pil_img = Image.open(img_path)
        if self.transforms:
            img_data = self.transforms(pil_img)
        else:
            pil_img = np.asarray(pil_img)
            img_data = torch.from_numpy(pil_img)
        return img_data

    def __len__(self):
        return len(self.imgs)


train_set = EmojiSet("emojis_am")
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

wandb.init(
    project="emoji_gan"
)

for epoch in range(epochs):
    avg_D_loss = 0
    avg_G_loss = 0
    for idx, imgs in tqdm(enumerate(train_loader)):
        idx += 1

        # grid = torchvision.utils.make_grid(imgs, nrow=4, normalize=True, scale_each=True)
        # grid_img = transforms.ToPILImage()(grid)
        # grid_img.show()
        # Training the discriminator
        # Real inputs are actual images of the MNIST dataset
        # Fake inputs are from the generator
        # Real inputs should be classified as 1 and fake as 0
        real_inputs = imgs.to(device)
        real_outputs = D(real_inputs)
        real_label = torch.ones(real_inputs.shape[0], 1).to(device)

        noise = torch.rand(real_inputs.shape[0], 1, 128, 128)
        noise = noise.to(device)
        fake_inputs = G(noise)
        fake_outputs = D(fake_inputs)
        fake_label = torch.zeros(fake_inputs.shape[0], 1).to(device)

        outputs = torch.cat((real_outputs, fake_outputs), 0)
        targets = torch.cat((real_label, fake_label), 0)

        D_loss = loss(outputs, targets)
        D_optimizer.zero_grad()
        D_loss.backward()
        D_optimizer.step()

        # Training the generator
        # For generator, goal is to make the discriminator believe everything is 1
        noise = torch.rand(real_inputs.shape[0], 1, 128, 128)
        noise = noise.to(device)

        fake_inputs = G(noise)
        fake_outputs = D(fake_inputs)
        fake_targets = torch.ones([fake_inputs.shape[0], 1]).to(device)
        G_loss = loss(fake_outputs, fake_targets)
        G_optimizer.zero_grad()
        G_loss.backward()
        G_optimizer.step()

        avg_D_loss += D_loss.item() * imgs.shape[0]
        avg_G_loss += G_loss.item() * imgs.shape[0]

    wandb.log({"discriminator_loss": avg_D_loss / len(train_loader),
               "generator_loss": avg_G_loss / len(train_loader)}, step=epoch)
    if epoch % 20 == 0 or epoch == epochs - 1:
        with torch.no_grad():
            noise = torch.rand(64, 1, 128, 128)
            noise = noise.to(device)
            generate_images = G(noise)
            grid = torchvision.utils.make_grid(generate_images, nrow=8, normalize=True, scale_each=True)
            g_img = transforms.ToPILImage()(grid)
            g_img.show()
            wandb.log({"generate_images": wandb.Image(torch.nan_to_num(grid.detach().cpu()))}, step=epoch)


