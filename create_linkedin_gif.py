from PIL import Image
import os
import glob
from utils import natural_sort

# Paths
DIR_STD = "tracking_vis_standard"
DIR_UNC = "tracking_vis_uncertainty"
GIF_OUT = "tissue_tracking_linkedin.gif"

def main():
    frames_std = natural_sort(glob.glob(os.path.join(DIR_STD, "*.png")))
    frames_unc = natural_sort(glob.glob(os.path.join(DIR_UNC, "*.png")))
    
    if not frames_std or not frames_unc:
        print("Frames not found. Please run visualize_dual.py first.")
        return

    # Use EVERY frame for maximum smoothness (10 FPS)
    step = 1
    f_std = frames_std[::step]
    f_unc = frames_unc[::step]
    
    imgs_std = []
    imgs_unc = []
    imgs_combined = []
    
    print(f"Processing {len(f_std)} frames at FULL resolution...")
    
    for fs, fu in zip(f_std, f_unc):
        # Open (keep original resolution 640x480)
        s = Image.open(fs)
        u = Image.open(fu)
        w, h = s.size
        
        # 1. Standard GIF
        imgs_std.append(s)
        
        # 2. Uncertainty GIF
        imgs_unc.append(u)
        
        # 3. Combined GIF (Side-by-side)
        # Create a canvas for side-by-side (1280x520 including label area)
        dst = Image.new('RGB', (w * 2, h + 40))
        dst.paste(s, (0, 40))
        dst.paste(u, (w, 40))
        import PIL.ImageDraw as ImageDraw
        draw = ImageDraw.Draw(dst)
        draw.text((10, 10), "Standard Tracking", fill=(255, 255, 255))
        draw.text((w + 10, 10), "UQ Tracking (CoTracker3 + MC-Dropout)", fill=(255, 255, 255))
        imgs_combined.append(dst)

    print("Saving FULL resolution GIFs...")
    imgs_std[0].save("tissue_tracking_standard.gif", save_all=True, append_images=imgs_std[1:], optimize=True, duration=100, loop=0)
    imgs_unc[0].save("tissue_tracking_uncertainty.gif", save_all=True, append_images=imgs_unc[1:], optimize=True, duration=100, loop=0)
    imgs_combined[0].save("tissue_tracking_comparison.gif", save_all=True, append_images=imgs_combined[1:], optimize=True, duration=100, loop=0)
    
    print("All FULL resolution GIFs saved.")

if __name__ == "__main__":
    main()
