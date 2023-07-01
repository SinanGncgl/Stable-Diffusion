import torch
import torchvision
import os
import numpy as np
from PIL import Image


class MiniImageNetTrain(torch.utils.data.Dataset):
    """
    ### mini imagenet dataset
    """

    def __init__(self, root, image_size=84):
        super(MiniImageNetTrain, self).__init__()
        self.image_size = image_size
        self.data = {}
        split_path = "miniimagenet_train.npy"
        with open(os.path.join(root, split_path), "rb") as f:
            self.imgs = np.load(f)

    def __getitem__(self, item):
        img = self.imgs[item]
        img = Image.fromarray(img)
        if not img.mode == "RGB":
            img = img.convert("RGB")

        if self.image_size != 84:
            img = img.resize((self.image_size, self.image_size), resample=Image.BICUBIC)

        image = np.array(img).astype(np.uint8)
        # normalize to [0, 1] then to [-1, 1]
        image = (image / 127.5 - 1.0).astype(np.float32)

        example = {}
        example["image"] = image
        # condition is also the same img because we will send it to the image embedder
        example["condition"] = image
        return example

    def __len__(self):
        return len(self.imgs)


class MiniImageNetVal(torch.utils.data.Dataset):
    """
    ### mini imagenet dataset
    """

    def __init__(self, root, image_size=84):
        super(MiniImageNetVal, self).__init__()
        self.image_size = image_size
        self.data = {}
        split_path = "miniimagenet_val.npy"
        with open(os.path.join(root, split_path), "rb") as f:
            self.imgs = np.load(f)

    def __getitem__(self, item):
        img = self.imgs[item]
        img = Image.fromarray(img)
        if not img.mode == "RGB":
            img = img.convert("RGB")

        if self.image_size != 84:
            img = img.resize((self.image_size, self.image_size), resample=Image.BICUBIC)

        image = np.array(img).astype(np.uint8)
        # normalize to [0, 1] then to [-1, 1]
        image = (image / 127.5 - 1.0).astype(np.float32)

        example = {}
        example["image"] = image
        # condition is also the same img because we will send it to the image embedder
        example["condition"] = image
        return example

    def __len__(self):
        return len(self.imgs)


if __name__ == "__main__":
    root = "./data/miniimagenet"
    dataset = MiniImageNetVal(root, image_size=64)
    print(len(dataset))

    # data loader
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=8, shuffle=True, num_workers=0, pin_memory=True
    )

    data = next(iter(data_loader))
    print("image:", data["image"].shape)
    print("condition", data["condition"].shape)

    # image max min values
    print("image max:", data["image"].max())
    print("image min:", data["image"].min())

    # save image to check
    torchvision.utils.save_image(
        data["image"].permute(0, 3, 1, 2).float(),
        "img_debug.png",
        normalize=True,
        nrow=8,
        value_range=(-1, 1),
    )
