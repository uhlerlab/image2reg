from torchvision.transforms import functional as F
import numpy as np
import torch
from torchvision import transforms


class RandomGamma(object):

    def __init__(self, limit=[0.5, 1.5]):
        self.limit = limit

    def __call__(self, img):
        gamma = np.random.uniform(self.limit[0], self.limit[1])
        return F.adjust_gamma(img=img, gamma=gamma)

    def __repr__(self):
        return self.__class__.__name__ + "(gamma={})".format(self.gamma)


class ToRGBTensor(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, img):
        return transforms.ToTensor()(img).repeat(3, 1, 1)
