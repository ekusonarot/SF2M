import torch
import torchvision
import torchvision.transforms as transforms
from torch.optim.adam import Adam
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
import numpy as np
from model import CNF
from tqdm import tqdm
import pickle
from torchvision.utils import save_image
from torch.utils.tensorboard import SummaryWriter

from sinkhorn import OTLayer

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def lr_scheduler(step):
    if step < 50:
        return 1.
    elif step < 100:
        return 0.1
    elif step < 200:
        return 0.01
    else:
        return 0.001
    
image_size = 32
channels = 3
beta = 1e-3
step = 1000
epoch = 100
model = CNF(image_size, channels).to(device)
optim = Adam(model.parameters(), lr=1e-3)
scheduler = LambdaLR(optim, lr_lambda=lr_scheduler)

if __name__ == "__main__":
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5),(0.5,0.5,0.5))])
    dataset = torchvision.datasets.CIFAR10(root='./data', 
                                            train=True,
                                            download=True,
                                            transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset,
                                                batch_size=200,
                                                shuffle=True,
                                                num_workers=2)
    writer = SummaryWriter()
    ot = OTLayer(1)
    losses = [None]*epoch
    for j in range(1,epoch+1):
        model.train()
        sum_loss = 0
        for i, (x, _) in enumerate(tqdm(dataloader)):
            optim.zero_grad()
            x_1 = x.to(device)
            x_0 = torch.randn_like(x).to(device)
            t = torch.rand(x.size(0)).to(device)
            B = x_1.size(0)
            with torch.no_grad():
                x_0 = ot(x_1.reshape(B, -1), x_0.reshape(B, -1), 100)
                x_0 = torch.nan_to_num(x_0)
                x_0 = x_0.reshape(x_1.shape)
            loss = model(x_0, x_1, t)
            loss.backward()
            optim.step()
            writer.add_scalar(f"loss/{j}", float(loss), i)
            sum_loss += float(loss)
            del loss
        losses[j-1] = sum_loss/len(dataloader)
        scheduler.step()
        #if j % 10 != 0:
        #    continue
        model.eval()
        x_0 = torch.randn(size=(16, channels, image_size, image_size)).to(device)
        x = model.sample(x_0)
        x = (x.clamp(-1, 1) + 1)/2
        grid = torchvision.utils.make_grid(x, 4)
        writer.add_image("SF2M", grid, j)
        with open("losses.pkl", "wb") as f:
            pickle.dump(losses[:j-1], f)
        if j%10==0:
            save_image(grid, f"checkpoint/epoch{j}.png")