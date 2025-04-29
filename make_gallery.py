import os
from PIL import Image
import matplotlib.pyplot as plt

files = ['000003.png', '000004.png']
n = len(files)
fig, axes = plt.subplots(n, 3, figsize=(9, 3*n))

for i, fname in enumerate(files):
    blur = Image.open(f'results/deblur_test/blur_{fname}').resize((256,256))
    pred = Image.open(f'results/deblur_test/pred_{fname}').resize((256,256))
    gt   = Image.open(f'data/GoPro_sampled/test/sharp/{fname}').resize((256,256))

    for ax, img, title in zip(axes[i], [blur, pred, gt], ['Blurred','Deblurred','Ground Truth']):
        ax.imshow(img)
        ax.axis('off')
        ax.set_title(title)
plt.tight_layout()
plt.savefig('results/qualitative_gallery.png')

