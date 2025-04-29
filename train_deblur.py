import os
import yaml
import torch
from torch import optim
from torch.utils.data import DataLoader, Subset
import torch.nn.functional as F

from models.clip_feature_extractor import CLIPFeatureExtractor
from models.vgg_loss import VGGLoss
from datasets.gopro import GoProDataset
from models.deblur_decoder import DeblurDecoder
from datasets.gopro import GoProDataset

def main(config_path, resume_ckpt=None):
    with open(config_path) as f:
        opt = yaml.safe_load(f)

    device = torch.device("mps" if torch.backends.mps.is_available() and opt['training']['device']=='mps' else "cpu")
    print(f"⏱ Using device: {device}")


    data_root = opt['data']['root']
    train_data_root = data_root
    
    full_ds = GoProDataset(root=train_data_root, 
                            mode='train',
                            crop_size=opt['data']['crop_size'],
                            crops_per=opt['data']['crops_per_image'])
    
    
    
    N = len(full_ds)
    val_size   = int(0.1 * (N / opt['data']['crops_per_image'])) * opt['data']['crops_per_image']
    train_size = N - val_size
    
    
    
    train_ds, val_ds = torch.utils.data.random_split(full_ds, [train_size, val_size])
    train_loader = DataLoader(train_ds, batch_size=opt['training']['batch_size'],
                              shuffle=True,  num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=opt['training']['batch_size'],
                              shuffle=False, num_workers=2, pin_memory=True)

    # Models
    clip_extractor = CLIPFeatureExtractor(backbone=opt['model']['clip_backbone'], unfreeze_layer4 = opt['model']['unfreeze_layer4'],
                                          device=device).to(device)
    if opt['model'].get('reset_clip_params', False):
        clip_extractor.reset_parameters()
    
    clip_extractor.eval()
    
    decoder = DeblurDecoder(clip_ch=opt['model']['clip_channels']).to(device)
    
    
    
    # Optimizer
    lr = float(opt['training']['lr'])
    trainable_params = decoder.parameters()
    
    if opt['model']['unfreeze_layer4']:
        trainable_params = list(trainable_params) + list(clip_extractor.get_trainable_parameters())
    
    optimizer = optim.Adam(trainable_params, lr=lr,
                           weight_decay=opt['training'].get('weight_decay', 0))
    
    
    # Resume logic
    start_epoch = 1
    best_val    = float('inf')
    wait        = 0
    patience    = opt['training'].get('early_stopping_patience', 3)
    if resume_ckpt:
        print(f"⏩ Resuming from {resume_ckpt}")
        ckpt = torch.load(resume_ckpt, map_location=device)
        if opt['model']['unfreeze_layer4']:
            clip_extractor.unfreeze_layer4 = True
            clip_extractor.reset_parameters()
            
        decoder.load_state_dict(ckpt['model_state'])
        start_epoch = ckpt['epoch'] + 1
        best_val    = ckpt.get('best_val_loss', best_val)

    # Training + Validation loops
    log_dir = opt['training']['log_dir']
    
    use_vgg_loss = opt['model'].get('use_vgg_loss', False)
    if use_vgg_loss:
        vgg_loss_fn = VGGLoss(device)
    
    
    os.makedirs(log_dir, exist_ok=True)

    for epoch in range(start_epoch, opt['training']['epochs'] + 1):
        # ——— Train ———
        decoder.train()
        running_loss = 0.0
        for blurry, sharp, fname in train_loader:
            blurry, sharp = blurry.to(device), sharp.to(device)
            with torch.no_grad():
                feats = clip_extractor(blurry)
            residual = decoder(blurry, feats)
            pred = blurry + residual
            l1_loss = F.l1_loss(pred, sharp)
            if use_vgg_loss:
                vgg_loss = vgg_loss_fn(pred, sharp)
                loss = l1_loss + vgg_loss
            else:
                loss = l1_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        train_loss = running_loss / len(train_loader)

        # ——— Validate ———
        decoder.eval()
        val_loss = 0.0
        with torch.no_grad():
            for blurry, sharp, fname in val_loader:
                blurry, sharp = blurry.to(device), sharp.to(device)
                feats = clip_extractor(blurry)
                residual = decoder(blurry, feats)
                pred = blurry + residual
                
                l1_loss = F.l1_loss(pred, sharp)
                if use_vgg_loss:
                    vgg_loss = vgg_loss_fn(pred, sharp)
                    loss = l1_loss + vgg_loss
                else:
                    loss = l1_loss
                val_loss += loss.item()
        val_loss /= len(val_loader)

        print(f"Epoch {epoch}: Train L1={train_loss:.4f} | Val L1={val_loss:.4f}")

        # ——— Early‐Stopping Check ———
        is_best = val_loss < best_val
        if is_best:
            best_val = val_loss
            wait = 0
            # Save best‐model checkpoint
            torch.save({
                'epoch': epoch,
                'model_state': decoder.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'best_val_loss': best_val
            }, os.path.join(log_dir, 'best_decoder.pth'))  
            
            print("  ⭐ New best model saved.")
        else:
            wait += 1
            if wait >= patience:
                print(f"⏸ Early stopping: no improvement in {patience} epochs.")
                break
            
        if epoch % opt['training']['save_interval'] == 0:
            ckpt = {
                'epoch': epoch,
                'model_state': decoder.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'best_val_loss': best_val
            }
            path = os.path.join(log_dir, f"decoder_epoch{epoch}.pth")
            torch.save(ckpt, path)
            print(f"  💾 Checkpoint saved: {path}")

    print("🏁 Training complete.")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str,
                        default='configs/clip_deblur_small.yaml')
    parser.add_argument('--resume', type=str, default=None,
                        help='.pth file to resume from')
    args = parser.parse_args()
    main(args.config, resume_ckpt=args.resume)