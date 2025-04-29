import os
import yaml
import numpy as np
import pandas as pd
from scipy.linalg import solve
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim    
from scipy.signal import wiener
from torch.utils.data import DataLoader
from torch.utils.data import Subset
import torch
from torchvision.utils import save_image

from datasets.gopro import GoProDataset

def evaluate_baseline(loader, output_dir, device):
    from skimage.filters import unsharp_mask
    os.makedirs(output_dir, exist_ok=True)
    records = []
    for blurry, sharp, fname in loader:
        fname = fname[0]
        blurry, sharp = blurry.to(device), sharp.to(device)

        # wiener filter
        b = blurry.cpu().numpy()[0].transpose(1,2,0)
        s = sharp.cpu().numpy()[0].transpose(1,2,0)

        p = wiener(b)
        m_psnr = psnr(s, p, data_range=2)
        m_ssim = ssim(s, p, channel_axis=2, data_range=2)

        records.append({'file': fname, 'psnr': m_psnr, 'ssim': m_ssim, 'method': 'wiener'})
        save_image(torch.tensor(p.transpose(2,0,1)), os.path.join(output_dir, f"wiener_{fname}"), normalize=True, value_range=(-1,1))

        # Unsharp Masking
        p = unsharp_mask(b, radius=1, amount=1.0, channel_axis=2)

        


        m_psnr = psnr(s, p, data_range=2)
        m_ssim = ssim(s, p, channel_axis=2, data_range=2)

        records.append({'file': fname, 'psnr': m_psnr, 'ssim': m_ssim, 'method': 'unsharp_mask'}) 
        save_image(torch.tensor(p.transpose(2,0,1)), os.path.join(output_dir, f"unsharp_mask_{fname}"), normalize=True, value_range=(-1,1))
    # save csv
    df = pd.DataFrame(records)
    df.to_csv(os.path.join(output_dir, 'baseline_unsharp_wiener_metrics.csv'), index=False)
    print(f"Avg PSNR: {df.loc[df['method'] == 'wiener']['psnr'].mean():.2f} dB, Avg SSIM: {df.loc[df['method'] == 'wiener']['ssim'].mean():.4f}", "(wiener)")
    print(f"Avg PSNR: {df.loc[df['method'] == 'unsharp_mask']['psnr'].mean():.2f} dB, Avg SSIM: {df.loc[df['method'] == 'unsharp_mask']['ssim'].mean():.4f}", "(unsharp_mask)")


if __name__ == '__main__':
    import argparse
    import yaml
    SELECTED = set()

    p = argparse.ArgumentParser()
    p.add_argument('--config',     type=str, default='configs/clip_deblur_small.yaml')
    p.add_argument('--output',     type=str, default='results')
    args = p.parse_args()

    # load config & device
    opt = yaml.safe_load(open(args.config))
    device = torch.device("mps" if torch.backends.mps.is_available() and opt['training']['device']=='mps' else "cpu")
    
    # build full test dataset
    full_ds = GoProDataset(
        root=opt['data']['root'],
        mode='test',
        crop_size=opt['data']['crop_size'],
        crops_per=1
    )
    # optionally filter by SELECTED filenames
    if SELECTED:
        indices = [i for i,(b,_) in enumerate(full_ds.pairs) 
                   if os.path.basename(b) in SELECTED]
        ds = Subset(full_ds, indices)
    else:
        ds = full_ds

    test_loader = DataLoader(ds, batch_size=1, shuffle=False)

    # run
    evaluate_baseline(test_loader, args.output, device)

