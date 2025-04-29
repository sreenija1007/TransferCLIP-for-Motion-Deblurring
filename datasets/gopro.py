import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T

class GoProDataset(Dataset):
    def __init__(self, root, mode='train', crop_size=256, crops_per=5):
        """
        Expects on disk:
          root/GoPro_sampled/train/blur/*.png
          root/GoPro_sampled/train/sharp/*.png
          root/GoPro_sampled/test/blur/*.png
          root/GoPro_sampled/test/sharp/*.png
        - train: produces `crops_per` random crop_size×crop_size patches per image
        - test : resizes each image once to crop_size×crop_size
        """
        self.mode = mode
        self.crop_size = crop_size
        self.crops_per = crops_per if mode == 'train' else 1

        split_dir = os.path.join(root, mode)
        blur_dir  = os.path.join(split_dir, 'blur')
        sharp_dir = os.path.join(split_dir, 'sharp')

        # Gather matching blur/sharp file pairs
        self.pairs = []
        for fname in sorted(os.listdir(blur_dir)):
            b = os.path.join(blur_dir, fname)
            s = os.path.join(sharp_dir, fname)
            if os.path.exists(s):
                self.pairs.append((b, s))

        # normalization: [0,255]→[0,1]→[-1,1]
        self.to_tensor = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.5]*3, std=[0.5]*3),
        ])
        # For test: resize then normalize
        self.test_transform = T.Compose([
            T.Resize((crop_size, crop_size), interpolation=Image.BILINEAR),
            self.to_tensor
        ])

    def __len__(self):
        return len(self.pairs) * self.crops_per

    def __getitem__(self, idx):
        pair_idx = idx // self.crops_per
        b_path, s_path = self.pairs[pair_idx]

        # Load PIL images
        blur  = Image.open(b_path).convert('RGB')
        sharp = Image.open(s_path).convert('RGB')

        if self.mode == 'train':
            i, j, h, w = T.RandomCrop.get_params(
                blur, (self.crop_size, self.crop_size)
            )
            blur  = T.functional.crop(blur,  i, j, h, w)
            sharp = T.functional.crop(sharp, i, j, h, w)
            blur_t  = self.to_tensor(blur)
            sharp_t = self.to_tensor(sharp)
        else:
            blur_t  = self.test_transform(blur)
            sharp_t = self.test_transform(sharp)

        fname = os.path.basename(b_path)
        return blur_t, sharp_t, fname
