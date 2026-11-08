import os
import cv2
import torch
from torch.utils.data import Dataset
from .transforms import normalize

class IVIFDataset(Dataset):

    def __init__(self, root, img_size=None):
        self.vis_dir = os.path.join(root, 'vi')
        self.ir_dir = os.path.join(root, 'ir')
        self.names = sorted(os.listdir(self.vis_dir))
        self.img_size = img_size

    def __len__(self):
        return len(self.names)

    def _load_img(self, path):
        img = cv2.imread(path, 0)
        img = cv2.resize(img, (self.img_size, self.img_size))
        img = img.astype('float32') / 255.0
        img = normalize(img)
        img = torch.from_numpy(img).unsqueeze(0)
        return img

    def __getitem__(self, idx):
        name = self.names[idx]
        vis = self._load_img(os.path.join(self.vis_dir, name))
        ir = self._load_img(os.path.join(self.ir_dir, name))
        return {'vis': vis, 'ir': ir, 'name': name}
