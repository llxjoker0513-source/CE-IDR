from public_config import load_config
import os
import argparse
import sys
import yaml
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from datasets.ivif_dataset import IVIFDataset
from models.fusion_model import CEIDR
from losses.fusion_loss import FusionLoss
from utils.logger import setup_logger
from utils.checkpoint import save_checkpoint
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def parse_args():
    parser = argparse.ArgumentParser(description='Train the IVIF fusion model.')
    parser.add_argument('--config', help='Path to a YAML training config.', required=True)
    return parser.parse_args()

def main():
    args = parse_args()
    cfg = load_config(args.config)
    torch.manual_seed(cfg['seed'])
    device = torch.device(cfg['device'] if torch.cuda.is_available() else 'cpu')
    train_set = IVIFDataset(cfg['dataset']['root'], cfg['dataset']['img_size'])
    train_loader = DataLoader(train_set, batch_size=cfg['dataset']['batch_size'], shuffle=True, num_workers=cfg['dataset']['num_workers'], pin_memory=device.type == 'cuda')
    model_cfg = cfg['model']
    model = CEIDR(in_ch=model_cfg['in_ch'], base_ch=model_cfg['base_ch'], num_experts=model_cfg['num_experts'], use_region_router=model_cfg['use_region_router'], use_source_router=model_cfg['use_source_router'], use_spatial_gate=model_cfg['use_spatial_gate'], expert_mode=model_cfg['expert_mode'], use_experts=model_cfg['use_experts'], router_mode=model_cfg['router_mode'], semantic_expand=model_cfg['semantic_expand']).to(device)
    optimizer = optim.Adam(model.parameters(), lr=cfg['train']['lr'], weight_decay=cfg['train']['weight_decay'])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg['train']['epochs'], eta_min=cfg['train']['min_lr'])
    criterion = FusionLoss(cfg['loss'])
    (logger, writer) = setup_logger(cfg['train']['log_dir'])
    logger.info('Training started...')
    logger.info(f'Python executable: {sys.executable}')
    logger.info(f'PyTorch version: {torch.__version__}')
    logger.info(f'torch.cuda.is_available(): {torch.cuda.is_available()}')
    logger.info(f'torch.version.cuda: {torch.version.cuda}')
    logger.info(f'Using device: {device}')
    if device.type == 'cuda':
        logger.info(f'CUDA device: {torch.cuda.get_device_name(0)}')
    logger.info(f'Config: {args.config}')
    logger.info(f'Model config: {model_cfg}')
    global_step = 0
    for epoch in range(cfg['train']['epochs']):
        model.train()
        epoch_loss = 0
        pbar = tqdm(train_loader)
        for batch in pbar:
            vis = batch['vis'].to(device)
            ir = batch['ir'].to(device)
            (fused, moe_aux, unc_aux) = model(vis, ir)
            loss = criterion(fused, vis, ir, moe_aux=moe_aux, unc=unc_aux)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            global_step += 1
            pbar.set_description(f'Epoch {epoch} Loss {loss.item():.4f}')
            writer.add_scalar('train/loss_step', loss.item(), global_step)
        avg_loss = epoch_loss / len(train_loader)
        current_lr = optimizer.param_groups[0]['lr']
        logger.info(f"Epoch [{epoch}/{cfg['train']['epochs']}], Loss: {avg_loss:.4f}")
        writer.add_scalar('train/loss_epoch', avg_loss, epoch)
        writer.add_scalar('train/lr', current_lr, epoch)
        if epoch % cfg['train']['save_freq'] == 0:
            save_checkpoint(model, optimizer, epoch, os.path.join(cfg['train']['ckpt_dir'], f'epoch_{epoch}.pth'))
        scheduler.step()
    logger.info('Training finished.')
if __name__ == '__main__':
    main()
