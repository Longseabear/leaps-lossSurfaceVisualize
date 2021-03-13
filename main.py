import torch
import torchvision.models as models
import torchvision
import skimage.io as io
from mpl_toolkits.mplot3d import Axes3D
from torchvision import datasets, transforms
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import argparse

# Visualizing the Loss Landscape of Neural Nets

def make_loss_surface(img, class_idx, model_name, model, step=10):
    model.eval()
    parameters = {}
    v1 = {}
    v2 = {}
    for name, params in model.named_parameters():
        parameters[name] = params.clone()
        v1[name] = torch.randn_like(params) * parameters[name]
        v2[name] = torch.randn_like(params) * parameters[name]
    X = np.linspace(-1, 1, step)
    alpha, beta = np.meshgrid(X, X)
    h, w = alpha.shape
    z = np.zeros_like(alpha)

    for y in range(h):
        for x in range(w):
            for name, params in model.named_parameters():
                if params.dim() > 1:
                    direction = alpha[y,x] * v1[name] + beta[y,x] * v2[name]
                    params.data.copy_(parameters[name] + direction)
            out = model(img)
            z[y, x] = -torch.log_softmax(out, 1)[0, class_idx]

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.set_title(model_name)
    surf = ax.plot_surface(alpha, beta, z, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()

parser = argparse.ArgumentParser(description='loss surface project')
parser.add_argument('--img_path', default='black_swan.jpeg', help='img path')
parser.add_argument('--label', default=100, help='imagenet class label')
parser.add_argument('--step',  default=15, help='-1:step:1 x,y coordinate')

args = parser.parse_args()

models = [('resnet18', models.resnet18(pretrained=True)),
          ('resnet34', models.resnet34(pretrained=True)),
          ('resnet50', models.resnet50(pretrained=True)),
          ('resnet101', models.resnet101(pretrained=True))]

img = io.imread(args.img_path)/255.
plt.imshow(img)
plt.show()
img = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((244,244))])(img)

img = img[None, :].float()

for name, model in models:
    make_loss_surface(img, args.label, name, model, args.step)

