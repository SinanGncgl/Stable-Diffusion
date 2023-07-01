import torch
import torchvision
import pickle
import os
import numpy as np
from PIL import Image


class MiniImageNetTrain(torch.utils.data.Dataset):
    """
    ### mini imagenet dataset
    """

    def __init__(self, root, image_size=84):
        super(MiniImageNetTrain, self).__init__()
        self.mean = [120.39586422 / 255.0, 115.59361427 / 255.0, 104.54012653 / 255.0]
        self.std = [70.68188272 / 255.0, 68.27635443 / 255.0, 72.54505529 / 255.0]
        self.normalize = torchvision.transforms.Normalize(mean=self.mean, std=self.std)

        self.transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize(image_size),
                torchvision.transforms.ToTensor(),
                self.normalize,
            ]
        )

        self.data = {}
        split_path = "miniimagenet_train.pickle"
        with open(os.path.join(root, split_path), "rb") as f:
            data = pickle.load(f, encoding="latin1")
            self.imgs = data["data"]

    def __getitem__(self, item):
        img = np.array(self.imgs[item]).astype("uint8")
        img = Image.fromarray(img)
        img = self.transform(img)

        example = {}
        example["image"] = img
        # condition is also the same img because we will send it to the image embedder
        example["condition"] = img
        return example

    def __len__(self):
        return len(self.imgs)


class MiniImageNetVal(torch.utils.data.Dataset):
    """
    ### mini imagenet dataset
    """

    def __init__(self, root, image_size=84):
        super(MiniImageNetVal, self).__init__()
        self.mean = [120.39586422 / 255.0, 115.59361427 / 255.0, 104.54012653 / 255.0]
        self.std = [70.68188272 / 255.0, 68.27635443 / 255.0, 72.54505529 / 255.0]
        self.normalize = torchvision.transforms.Normalize(mean=self.mean, std=self.std)

        self.transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize(image_size),
                torchvision.transforms.ToTensor(),
                self.normalize,
            ]
        )

        self.data = {}
        split_path = "miniimagenet_val.pickle"
        with open(os.path.join(root, split_path), "rb") as f:
            data = pickle.load(f, encoding="latin1")
            self.imgs = data["data"]

    def __getitem__(self, item):
        img = np.array(self.imgs[item]).astype("uint8")
        img = Image.fromarray(img)
        img = self.transform(img)

        example = {}
        example["image"] = img
        # condition is also the same img because we will send it to the image embedder
        example["condition"] = img
        return example

    def __len__(self):
        return len(self.imgs)
