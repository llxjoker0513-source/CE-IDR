import argparse
from pathlib import Path

import cv2
import torch
from tqdm import tqdm

from datasets.ivif_dataset import IVIFDataset
from public_config import build_model, load_config


def main():
    parser = argparse.ArgumentParser(description="CE-IDR inference (private module required).")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")
    settings = config["inference"]
    model = build_model(config["model"]).to(device)
    checkpoint = torch.load(settings["checkpoint"], map_location=device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    dataset = IVIFDataset(settings["root"], settings["img_size"])
    output = Path(settings["save_dir"])
    output.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        for sample in tqdm(dataset):
            vis = sample["vis"].unsqueeze(0).to(device)
            ir = sample["ir"].unsqueeze(0).to(device)
            fused, _, _ = model(vis, ir)
            image = fused.squeeze().cpu().numpy()
            image = (image.clip(0, 1) * 255).astype("uint8")
            path = output / (Path(sample["name"]).stem + ".png")
            if not cv2.imwrite(str(path), image):
                raise OSError("Failed to save fused image: " + str(path))


if __name__ == "__main__":
    main()
