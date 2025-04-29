import os
import yaml
import torch
from torch.utils.data import DataLoader, Subset
from torchvision.utils import save_image
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim
import pandas as pd

from models.clip_feature_extractor import CLIPFeatureExtractor
from models.deblur_decoder import DeblurDecoder
from datasets.gopro import GoProDataset

SELECTED = set()  # {'0010.png','0025.png'} to pick specific

def evaluate(model, extractor, loader, device, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    records = []
    model.eval()
    with torch.no_grad():
        for blurry, sharp, fname in loader:
            fname = fname[0]
            blurry, sharp = blurry.to(device), sharp.to(device)
            feats = extractor(blurry)
            residual = model(blurry, feats)
            pred = blurry+residual

            # compute metrics on the single image in batch
            b = blurry.cpu().numpy()[0].transpose(1,2,0)
            s = sharp.cpu().numpy()[0].transpose(1,2,0)
            p = pred.cpu().numpy()[0].transpose(1,2,0)
            m_psnr = psnr(s, p, data_range=2)
            m_ssim = ssim(s, p, channel_axis=2, data_range=2)
            records.append({'file': fname, 'psnr': m_psnr, 'ssim': m_ssim})

            save_image(blurry, os.path.join(output_dir, f"blur_{fname}"),
                       normalize=True, value_range=(-1,1))
            save_image(pred,   os.path.join(output_dir, f"pred_{fname}"),
                       normalize=True, value_range=(-1,1))
            save_image(sharp,  os.path.join(output_dir, f"sharp_{fname}"),
                       normalize=True, value_range=(-1,1))

    df = pd.DataFrame(records)
    df.to_csv(os.path.join(output_dir, 'detailed_metrics.csv'), index=False)
    print(f"Avg PSNR: {df['psnr'].mean():.2f} dB, Avg SSIM: {df['ssim'].mean():.4f}")

if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--config',     type=str, required=True)
    p.add_argument('--checkpoint', type=str, default='runs/clip_deblur/best_decoder.pth')
    p.add_argument('--output',     type=str, default='results/deblur_test')
    args = p.parse_args()

    opt = yaml.safe_load(open(args.config))
    device = torch.device("mps" if torch.backends.mps.is_available() and opt['training']['device']=='mps' else "cpu")

    full_ds = GoProDataset(
        root=opt['data']['root'],
        mode='test',
        crop_size=opt['data']['crop_size'],
        crops_per=1
    )

    if SELECTED:
        indices = [i for i,(b,_) in enumerate(full_ds.pairs) 
                   if os.path.basename(b) in SELECTED]
        ds = Subset(full_ds, indices)
    else:
        ds = full_ds

    test_loader = DataLoader(ds, batch_size=1, shuffle=False)

    # build models
    extractor = CLIPFeatureExtractor(opt['model']['clip_backbone'], device).to(device)
    decoder   = DeblurDecoder(opt['model']['clip_channels']).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    decoder.load_state_dict(ckpt['model_state'])

    # run
    evaluate(decoder, extractor, test_loader, device, args.output)
