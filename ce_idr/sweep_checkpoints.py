from public_config import build_model as _build_model, load_config

def build_model(model_config):
    return _build_model(model_config).to(DEVICE)

import argparse
import csv
import os
import re
import cv2
import torch
import yaml
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from utils.metrics_gpu import evaluate_main_batch
METRIC_KEYS = ('EN', 'AG', 'SD', 'SCD', 'Qabf', 'VIF', 'SF')

class PairDataset(Dataset):

    def __init__(self, vis_dir, ir_dir, target_size=None):
        self.vis_dir = vis_dir
        self.ir_dir = ir_dir
        self.target_size = target_size
        exts = ('.png', '.jpg', '.bmp', '.tif')
        vis_names = {f for f in os.listdir(vis_dir) if f.lower().endswith(exts)}
        ir_names = {f for f in os.listdir(ir_dir) if f.lower().endswith(exts)}
        self.names = sorted(vis_names & ir_names)
        assert self.names, 'No matched vis/ir image pairs found.'

    def __len__(self):
        return len(self.names)

    def _read(self, path):
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        assert img is not None, f'Failed to read image: {path}'
        if img.shape != (self.target_size, self.target_size):
            img = cv2.resize(img, (self.target_size, self.target_size), interpolation=cv2.INTER_LINEAR)
        img = torch.from_numpy(img).float().unsqueeze(0) / 255.0
        return img

    def __getitem__(self, idx):
        name = self.names[idx]
        vis = self._read(os.path.join(self.vis_dir, name))
        ir = self._read(os.path.join(self.ir_dir, name))
        return (name, vis, ir)

def epoch_from_path(path):
    match = re.search('epoch_(\\d+)\\.pth$', os.path.basename(path))
    return int(match.group(1)) if match else -1

def collect_checkpoints(ckpt_dir, epochs=None):
    ckpts = [os.path.join(ckpt_dir, f) for f in os.listdir(ckpt_dir) if re.match('epoch_\\d+\\.pth$', f)]
    ckpts = sorted(ckpts, key=epoch_from_path)
    if epochs is not None:
        keep = set(epochs)
        ckpts = [p for p in ckpts if epoch_from_path(p) in keep]
    assert ckpts, f'No checkpoints found in {ckpt_dir}'
    return ckpts

def load_model(ckpt_path, model_cfg):
    model = build_model(model_cfg)
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    state = ckpt['model'] if isinstance(ckpt, dict) and 'model' in ckpt else ckpt
    model.load_state_dict(state)
    model.eval()
    return model

def fused_name(name):
    return os.path.splitext(name)[0] + '.png'

@torch.no_grad()
def evaluate_checkpoint(ckpt_path, loader, model_cfg):
    model = load_model(ckpt_path, model_cfg)
    sums = {}
    n = 0
    for (_, vis, ir) in tqdm(loader, desc=f'epoch_{epoch_from_path(ckpt_path)}', leave=False):
        vis = vis.to(DEVICE, non_blocking=True)
        ir = ir.to(DEVICE, non_blocking=True)
        (fused, _, _) = model(vis, ir)
        metrics = evaluate_main_batch(fused, vis, ir)
        batch_size = fused.size(0)
        for key in METRIC_KEYS:
            value = metrics[key]
            sums[key] = sums.get(key, 0.0) + value.detach().cpu().sum().item()
        n += batch_size
    del model
    if DEVICE.type == 'cuda':
        torch.cuda.empty_cache()
    result = {'epoch': epoch_from_path(ckpt_path), 'ckpt': ckpt_path}
    result.update({key: sums[key] / n for key in METRIC_KEYS})
    return result

def add_scores(rows):
    for row in rows:
        row['score'] = 0.0
    for (metric, weight) in SCORE_WEIGHTS.items():
        values = [row[metric] for row in rows]
        min_value = min(values)
        max_value = max(values)
        denom = max(max_value - min_value, 1e-08)
        for row in rows:
            if metric in LOWER_IS_BETTER:
                row['score'] += weight * ((max_value - row[metric]) / denom)
            else:
                row['score'] += weight * ((row[metric] - min_value) / denom)

@torch.no_grad()
def save_fused_images(ckpt_path, loader, save_dir, model_cfg):
    os.makedirs(save_dir, exist_ok=True)
    model = load_model(ckpt_path, model_cfg)
    for (names, vis, ir) in tqdm(loader, desc='saving best fused images'):
        vis = vis.to(DEVICE, non_blocking=True)
        ir = ir.to(DEVICE, non_blocking=True)
        (fused, _, _) = model(vis, ir)
        fused = fused.squeeze(1).detach().cpu().numpy() * 255.0
        for (idx, name) in enumerate(names):
            img = fused[idx].clip(0, 255).astype('uint8')
            cv2.imwrite(os.path.join(save_dir, fused_name(name)), img)
    del model
    if DEVICE.type == 'cuda':
        torch.cuda.empty_cache()

def write_csv(rows, csv_path):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    keys = ['epoch', 'score', 'ckpt'] + [key for key in METRIC_KEYS if key in rows[0]]
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)

def print_top(rows, key, top_k=5):
    print(f'\n====== Top {top_k} by {key} ======')
    for row in sorted(rows, key=lambda item: item[key], reverse=True)[:top_k]:
        print(f"epoch_{row['epoch']:03d}  {key}: {row[key]:.4f}  " + '  '.join((f'{metric}: {row[metric]:.4f}' for metric in METRIC_KEYS)))

def main():
    global SCORE_WEIGHTS, LOWER_IS_BETTER, PRIMARY_METRIC, DEVICE
    parser = argparse.ArgumentParser(description="CE-IDR checkpoint evaluation (private module required).")
    parser.add_argument("--config", required=True)
    parser.add_argument("--epochs", nargs="+", type=int)
    parser.add_argument("--no-save-fused", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    settings = config["eval"]
    selection = config["selection"]
    SCORE_WEIGHTS = selection["score_weights"]
    LOWER_IS_BETTER = set(selection["lower_is_better"])
    PRIMARY_METRIC = selection["primary_metric"]
    DEVICE = torch.device(config["device"] if torch.cuda.is_available() else "cpu")
    dataset = PairDataset(settings["vis_dir"], settings["ir_dir"], settings["target_size"])
    loader = DataLoader(
        dataset, batch_size=settings["batch_size"], shuffle=False,
        num_workers=settings["num_workers"], pin_memory=DEVICE.type == "cuda",
    )
    checkpoints = collect_checkpoints(config["train"]["ckpt_dir"], args.epochs)
    rows = [evaluate_checkpoint(path, loader, config["model"]) for path in checkpoints]
    add_scores(rows)
    write_csv(rows, settings["csv_path"])
    best = max(rows, key=lambda row: row["score"])
    primary = min if PRIMARY_METRIC in LOWER_IS_BETTER else max
    best_primary = primary(rows, key=lambda row: row[PRIMARY_METRIC])
    print("Best by configured score:", best["ckpt"])
    print("Best by configured primary metric:", best_primary["ckpt"])
    if not args.no_save_fused:
        save_fused_images(best["ckpt"], loader, settings["save_dir"], config["model"])


if __name__ == "__main__":
    main()
