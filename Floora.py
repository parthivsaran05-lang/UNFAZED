import argparse
import os
import sys
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.utils import save_image, make_grid
from PIL import Image as PILImage
import matplotlib.pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_epochs", type=int, default=20, help="number of epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size")
    parser.add_argument("--g_lr", type=float, default=0.0001, help="generator learning rate")
    parser.add_argument("--d_lr", type=float, default=0.0001, help="discriminator learning rate")
    parser.add_argument("--b1", type=float, default=0.5, help="adam beta1")
    parser.add_argument("--b2", type=float, default=0.999, help="adam beta2")
    parser.add_argument("--n_cpu", type=int, default=2, help="workers")
    parser.add_argument("--latent_dim", type=int, default=128, help="latent dimension")
    parser.add_argument("--img_size", type=int, default=64, help="image size")
    parser.add_argument("--channels", type=int, default=3, help="num image channels")
    parser.add_argument("--sample_interval", type=int, default=200, help="save every N steps")
    parser.add_argument("--exp_folder", type=str, default="exp", help="save folder")
    parser.add_argument("--data_dir", type=str, default="./floorplan_images", help="folder with floorplan images")
    if 'google.colab' in sys.modules:
        sys.argv = [a for a in sys.argv if not a.startswith('-f')]
        return parser.parse_args([])
    return parser.parse_args()

opt = parse_args()
cuda = torch.cuda.is_available()

os.makedirs(f"./exps/{opt.exp_folder}", exist_ok=True)

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
lambda_gp = 10

class FloorplanImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(root_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
        if not self.image_files:
            raise RuntimeError(f"No images found in {root_dir}. Please put your floorplan images there.")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.image_files[idx])
        img = PILImage.open(img_path).convert("RGB").resize((opt.img_size, opt.img_size))
        if self.transform:
            img = self.transform(img)
        return img

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.fc = nn.Linear(opt.latent_dim, 512*4*4)
        self.conv = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, 2, 1), nn.BatchNorm2d(256), nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.BatchNorm2d(64), nn.ReLU(True),
            nn.ConvTranspose2d(64, opt.channels, 4, 2, 1), nn.Tanh()
        )

    def forward(self, z):
        x = self.fc(z).view(-1, 512, 4, 4)
        return self.conv(x)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(opt.channels, 64, 4, 2, 1), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1), nn.BatchNorm2d(256), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 4, 2, 1), nn.BatchNorm2d(512), nn.LeakyReLU(0.2, inplace=True),
        )
        self.fc = nn.Linear(512*4*4, 1)

    def forward(self, img):
        x = self.conv(img).view(-1, 512*4*4)
        return self.fc(x)

def compute_gradient_penalty(D, real_samples, fake_samples):
    alpha = torch.rand(real_samples.size(0), 1, 1, 1, device=real_samples.device)
    interpolates = (alpha * real_samples + (1 - alpha) * fake_samples).requires_grad_(True)
    d_interpolates = D(interpolates)
    fake = torch.ones(real_samples.size(0), 1, device=real_samples.device)
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    return ((gradients.norm(2, dim=1) - 1) ** 2).mean()

generator = Generator()
discriminator = Discriminator()
if cuda:
    generator.cuda()
    discriminator.cuda()

optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.g_lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.d_lr, betas=(opt.b1, opt.b2))

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])
dataset = FloorplanImageDataset(opt.data_dir, transform)
dataloader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.n_cpu)

batches_done = 0
for epoch in range(opt.n_epochs):
    for i, real_imgs in enumerate(dataloader):
        real_imgs = real_imgs.type(Tensor)

        # Train D
        optimizer_D.zero_grad()
        z = torch.randn(real_imgs.size(0), opt.latent_dim).type(Tensor)
        fake_imgs = generator(z).detach()
        real_validity = discriminator(real_imgs)
        fake_validity = discriminator(fake_imgs)
        gradient_penalty = compute_gradient_penalty(discriminator, real_imgs.data, fake_imgs.data)
        d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + lambda_gp * gradient_penalty
        d_loss.backward()
        optimizer_D.step()

        # Train G every n_critic steps
        if i % 5 == 0:
            optimizer_G.zero_grad()
            z = torch.randn(real_imgs.size(0), opt.latent_dim).type(Tensor)
            gen_imgs = generator(z)
            g_loss = -torch.mean(discriminator(gen_imgs))
            g_loss.backward()
            optimizer_G.step()

            print(f"[Epoch {epoch}/{opt.n_epochs}] [Batch {i}/{len(dataloader)}] "
                  f"[D loss: {d_loss.item():.4f}] [G loss: {g_loss.item():.4f}]")

            if batches_done % opt.sample_interval == 0:
                # Save
                save_image(gen_imgs[:25], f"./exps/{opt.exp_folder}/{batches_done}.png",
                           nrow=5, normalize=True)

                # Show in matplotlib
                grid = make_grid(gen_imgs[:25], nrow=5, normalize=True)
                npimg = grid.cpu().detach().numpy()
                plt.figure(figsize=(8,8))
                plt.imshow(np.transpose(npimg, (1,2,0)))
                plt.axis("off")
                plt.show()

        batches_done += 1
