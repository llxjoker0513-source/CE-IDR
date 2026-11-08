from public_config import load_config

import os
import argparse
import cv2
import torch
import yaml
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from utils.metrics_gpu import evaluate_main_batch
METRIC_KEYS = ('EN', 'AG', 'SD', 'SCD', 'Qabf', 'VIF', 'SF')

class FusionDataset(Dataset):

    def __init__(self, vis_dir, ir_dir, fused_dir, target_size=None):
        self.vis_dir = vis_dir
        self.ir_dir = ir_dir
        self.fused_dir = fused_dir
        self.target_size = target_size
        exts = ('.png', '.jpg', '.bmp', '.tif')
        vis_map = {os.path.splitext(f)[0]: f for f in os.listdir(vis_dir) if f.lower().endswith(exts)}
        ir_map = {os.path.splitext(f)[0]: f for f in os.listdir(ir_dir) if f.lower().endswith(exts)}
        fused_map = {}
        for f in sorted(os.listdir(fused_dir)):
            if not f.lower().endswith(exts):
                continue
            (stem, ext) = os.path.splitext(f)
            if stem not in fused_map or ext.lower() == '.png':
                fused_map[stem] = f
        stems = sorted(set(vis_map) & set(ir_map) & set(fused_map))
        assert stems, 'No matched vis/ir/fused image triples found!'
        self.samples = [(vis_map[stem], ir_map[stem], fused_map[stem]) for stem in stems]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        (vis_name, ir_name, fused_name) = self.samples[idx]
        vis = cv2.imread(os.path.join(self.vis_dir, vis_name), cv2.IMREAD_GRAYSCALE)
        ir = cv2.imread(os.path.join(self.ir_dir, ir_name), cv2.IMREAD_GRAYSCALE)
        fused = cv2.imread(os.path.join(self.fused_dir, fused_name), cv2.IMREAD_GRAYSCALE)
        assert vis is not None and ir is not None and (fused is not None), f'Load failed: {fused_name}'
        if vis.shape != (self.target_size, self.target_size):
            vis = cv2.resize(vis, (self.target_size, self.target_size), interpolation=cv2.INTER_LINEAR)
        if ir.shape != (self.target_size, self.target_size):
            ir = cv2.resize(ir, (self.target_size, self.target_size), interpolation=cv2.INTER_LINEAR)
        if fused.shape != (self.target_size, self.target_size):
            fused = cv2.resize(fused, (self.target_size, self.target_size), interpolation=cv2.INTER_LINEAR)
        vis = torch.from_numpy(vis).float().unsqueeze(0) / 255.0
        ir = torch.from_numpy(ir).float().unsqueeze(0) / 255.0
        fused = torch.from_numpy(fused).float().unsqueeze(0) / 255.0
        return (fused, vis, ir)

def main():
    parser = argparse.ArgumentParser(description="CE-IDR saved-image evaluation.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    settings = config["eval"]
    device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")
    dataset = FusionDataset(
        settings["vis_dir"], settings["ir_dir"], settings["fused_dir"],
        target_size=settings["target_size"],
    )
    loader = DataLoader(
        dataset, batch_size=settings["batch_size"], shuffle=False,
        num_workers=settings["num_workers"], pin_memory=device.type == "cuda",
    )
    sums = {key: 0.0 for key in METRIC_KEYS}
    count = 0
    with torch.no_grad():
        for fused, vis, ir in tqdm(loader):
            metrics = evaluate_main_batch(fused.to(device), vis.to(device), ir.to(device))
            for key in METRIC_KEYS:
                sums[key] += metrics[key].sum().item()
            count += fused.size(0)
    for key in METRIC_KEYS:
        print(f"{key}: {sums[key] / count:.4f}")


if __name__ == "__main__":
    main()
