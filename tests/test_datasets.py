import pytest

import torch

from torchvision import transforms
from PIL import Image

from compressai.datasets import ImageFolder


def save_fake_image(filepath, size=(512, 512)):
    img = Image.new('RGB', size=size)
    img.save(filepath)


class TestImageFolder():
    def test_init_ok(self, tmpdir):
        tmpdir.mkdir("train")
        tmpdir.mkdir("test")

        train_dataset = ImageFolder(tmpdir, split='train')
        test_dataset = ImageFolder(tmpdir, split='test')

        assert len(train_dataset) == 0
        assert len(test_dataset) == 0

    def test_count_ok(self, tmpdir):
        tmpdir.mkdir('train')
        (tmpdir / "train" / 'img1.jpg').write('')
        (tmpdir / "train" / 'img2.jpg').write('')
        (tmpdir / "train" / 'img3.jpg').write('')

        train_dataset = ImageFolder(tmpdir, split='train')

        assert len(train_dataset) == 3

    def test_invalid_dir(self, tmpdir):
        with pytest.raises(RuntimeError):
            ImageFolder(tmpdir)

    def test_load(self, tmpdir):
        tmpdir.mkdir('train')
        save_fake_image((tmpdir / 'train' / 'img0.jpeg').strpath)

        train_dataset = ImageFolder(tmpdir, split='train')
        assert isinstance(train_dataset[0], Image.Image)

    def test_load_transforms(self, tmpdir):
        tmpdir.mkdir('train')
        save_fake_image((tmpdir / 'train' / 'img0.jpeg').strpath)

        transform = transforms.Compose([
            transforms.CenterCrop((128, 128)),
            transforms.ToTensor(),
        ])
        train_dataset = ImageFolder(tmpdir, split='train', transform=transform)
        assert isinstance(train_dataset[0], torch.Tensor)
        assert train_dataset[0].size() == (3, 128, 128)
