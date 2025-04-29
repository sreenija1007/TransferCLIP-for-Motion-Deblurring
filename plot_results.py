import pandas as pd
import matplotlib.pyplot as plt

# ── Load metrics ──────────────────────────────────────────────────
# CLIP-guided metrics (from test_deblur.py output)
clip_df    = pd.read_csv('results/deblur_test/detailed_metrics.csv')
# Baseline metrics (RL and Wiener)
base_df    = pd.read_csv('results/baseline_unsharp_wiener_metrics.csv')

# ── Compute average per method ────────────────────────────────────
# CLIP
clip_avg    = {'method': 'CLIP-guided',
               'psnr': clip_df['psnr'].mean(),
               'ssim': clip_df['ssim'].mean()}

# unsharp_mask & Wiener
base_avg = base_df.groupby('method')[['psnr','ssim']].mean().reset_index()

# Combine into one DataFrame
avg_df = pd.concat([
    pd.DataFrame([clip_avg]),
    base_avg
], ignore_index=True)

# ── PSNR bar chart ───────────────────────────────────────────────
plt.figure()
plt.bar(avg_df['method'], avg_df['psnr'])
plt.ylabel('Average PSNR (dB)')
plt.title('Deblurring: PSNR Comparison')
plt.tight_layout()
plt.savefig('results/psnr_comparison.png')
plt.close()

# ── SSIM bar chart ───────────────────────────────────────────────
plt.figure()
plt.bar(avg_df['method'], avg_df['ssim'])
plt.ylabel('Average SSIM')
plt.title('Deblurring: SSIM Comparison')
plt.tight_layout()
plt.savefig('results/ssim_comparison.png')
