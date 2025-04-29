import glob, random, os, shutil, os.path

def sample_images(source_dir, dest_dir, num_images, seed):
    """Samples blur/sharp image pairs from a source directory and copies them to a destination directory."""

    all_blurs = glob.glob(os.path.join(source_dir, "*", "blur", "*.png"))
    pairs = [(b, b.replace("/blur/", "/sharp/")) for b in all_blurs if os.path.exists(b.replace("/blur/", "/sharp/"))]

    if len(pairs) < num_images:
        raise ValueError(f"Not enough image pairs found in {source_dir} to sample {num_images} images.")

    random.seed(seed)
    sampled_pairs = random.sample(pairs, num_images)

    out_blur_dir = os.path.join(dest_dir, "blur")
    out_sharp_dir = os.path.join(dest_dir, "sharp")

    os.makedirs(out_blur_dir, exist_ok=True)
    os.makedirs(out_sharp_dir, exist_ok=True)

    for file_name in os.listdir(out_blur_dir):
        file_path = os.path.join(out_blur_dir, file_name)
        if os.path.isfile(file_path):
            os.remove(file_path)
    for file_name in os.listdir(out_sharp_dir):
        file_path = os.path.join(out_sharp_dir, file_name)
        if os.path.isfile(file_path):
            os.remove(file_path)

    for b, s in sampled_pairs:
        filename = os.path.basename(b)
        shutil.copy(b, os.path.join(out_blur_dir, filename))
        shutil.copy(s, os.path.join(out_sharp_dir, filename))

    print(f"✅ Done - {num_images} blur/sharp pairs in {dest_dir}")


if __name__ == "__main__":
    gopro_sampled_dir = "data/GoPro_sampled"
    os.makedirs(gopro_sampled_dir, exist_ok=True)
    # Sampling for training data
    source_train_dir = "data/gopro_raw/train"
    dest_train_dir = os.path.join(gopro_sampled_dir, "train")
    num_train_images = 1500
    sample_images(source_train_dir, dest_train_dir, num_train_images, 42)

    # Sampling for testing data
    source_test_dir = "data/gopro_raw/test"
    dest_test_dir = os.path.join(gopro_sampled_dir, "test")
    num_test_images = 250
    sample_images(source_test_dir, dest_test_dir, num_test_images, 43)
    
