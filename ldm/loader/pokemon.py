from datasets import load_dataset
from torchvision import transforms
from einops import rearrange
from omegaconf import ListConfig
from ldm.util import instantiate_from_config
from pathlib import Path
import torch
from torch.utils.data import Dataset


def hf_dataset(
    name,
    image_transforms=[],
    image_column="image",
    text_column="text",
    split="train",
    image_key="image",
    caption_key="txt",
):
    """Make huggingface dataset with appropriate list of transforms applied"""
    ds = load_dataset(name, split=split)
    tform = make_tranforms(image_transforms)

    assert (
        image_column in ds.column_names
    ), f"Didn't find column {image_column} in {ds.column_names}"
    assert (
        text_column in ds.column_names
    ), f"Didn't find column {text_column} in {ds.column_names}"

    def pre_process(examples):
        processed = {}
        processed[image_key] = [tform(im) for im in examples[image_column]]
        processed[caption_key] = examples[text_column]
        return processed

    ds.set_transform(pre_process)
    return ds


def make_tranforms(image_transforms):
    if isinstance(image_transforms, ListConfig):
        image_transforms = [instantiate_from_config(tt) for tt in image_transforms]
    image_transforms.extend(
        [
            transforms.ToTensor(),
            transforms.Lambda(lambda x: rearrange(x * 2.0 - 1.0, "c h w -> h w c")),
        ]
    )
    image_transforms = transforms.Compose(image_transforms)
    return image_transforms


class TextOnly(Dataset):
    def __init__(
        self, captions, output_size, image_key="image", caption_key="txt", n_gpus=1
    ):
        """Returns only captions with dummy images"""
        self.output_size = output_size
        self.image_key = image_key
        self.caption_key = caption_key
        if isinstance(captions, Path):
            self.captions = self._load_caption_file(captions)
        else:
            self.captions = captions

        if n_gpus > 1:
            # hack to make sure that all the captions appear on each gpu
            repeated = [n_gpus * [x] for x in self.captions]
            self.captions = []
            [self.captions.extend(x) for x in repeated]

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, index):
        dummy_im = torch.zeros(3, self.output_size, self.output_size)
        dummy_im = rearrange(dummy_im * 2.0 - 1.0, "c h w -> h w c")
        return {self.image_key: dummy_im, self.caption_key: self.captions[index]}

    def _load_caption_file(self, filename):
        with open(filename, "rt") as f:
            captions = f.readlines()
        return [x.strip("\n") for x in captions]
