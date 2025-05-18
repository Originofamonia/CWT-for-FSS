"""
1. make 1 csv file to have all image and depth pairs
2. make videos containing those pairs to check if files match by list order
"""

import os
import pickle
import pandas as pd
from pathlib import Path
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from moviepy import ImageSequenceClip


def find_matching_pairs(base_dir):
    base_dir = Path(base_dir)

    img_files = [f for f in sorted(base_dir.glob('**/img/*.jpg'))]
    if len(img_files) < 1:
        return None

    depth_files = [f for f in sorted(base_dir.glob('**/front_laser/*.pkl'))]

    if len(img_files) == len(depth_files):
        df = pd.DataFrame({
            'img_path': [str(f) for f in img_files],
            'depth_path': [str(f) for f in depth_files]
        })
    else:
        df = None
    
    return df


def draw_2_by_2_images():
    """
    1. read '/home/edward/data/trav/all_image_depth_pair.csv'
    2. draw a 2*2 figure: 1 img, 2 depth heatmap overlay img, 3 depth sector map
    3. save as images
    """
    # img_path, depth_path
    df = pd.read_csv('/home/edward/data/trav/all_image_depth_pair.csv')
    dpi = 200
    sector_left = -45 #-135
    sector_right = 45 # 135
    angle_min = -26
    angle_max = 36
    angle_rad_min = np.deg2rad(angle_min)
    angle_rad_max = np.deg2rad(angle_max)
    min_pct = (angle_min+45)/90  # percentile for cropping
    max_pct = (angle_max+45)/90
    for i, row in df.iterrows():
        # img = row['img_path']
        # depth = row['depth_path']

        img_basename = os.path.splitext(os.path.basename(row['img_path']))[0]
        laser_basename = os.path.splitext(os.path.basename(row['depth_path']))[0]
        rgb_image = plt.imread(row['img_path'])
        # img_width = rgb_image.shape[1]
        # img_height = rgb_image.shape[0]

        with open(row['depth_path'], 'rb') as f:
            data = pickle.load(f)

            vector = np.array(data['ranges'][::-1])[540:900]
            angles = np.linspace(np.deg2rad(sector_left), np.deg2rad(sector_right), len(vector), endpoint=False)

            fig, axes = plt.subplots(2, 2)
            axes[0,0].imshow(rgb_image)
            axes[0,0].figure.dpi = dpi
            axes[0,0].axis('off')

            axes[1,0] = plt.subplot(223, projection='polar')
            axes[1,0].plot(angles, vector)
            axes[1,0].plot([angle_rad_max, angle_rad_max], [0, 5.1], color='red', linestyle='--')
            axes[1,0].plot([angle_rad_min, angle_rad_min], [0, 5.1], color='blue', linestyle='--')
            axes[1,0].set_thetamin(sector_left)
            axes[1,0].set_thetamax(sector_right)
            axes[1,0].set_theta_zero_location('N')
            axes[1,0].set_xticks(np.pi/180. * np.linspace(sector_left, sector_right, 10, endpoint=False))
            axes[1,0].figure.axes[2].set_axis_off()

            depth_vector = vector[int(min_pct* len(vector)):int(max_pct*len(vector))][::-1]
            x_original = np.linspace(0, 1, len(depth_vector)) # x-coordinates for original data
            f = interp1d(x_original, depth_vector)
            x_new = np.linspace(0, 1, rgb_image.shape[1]) # x-coordinates for new data
            interp_vector = f(x_new)
            tiled_vector = np.tile(interp_vector, (rgb_image.shape[0],1))
            axes[0,1].imshow(rgb_image)
            axes[0,1].imshow(tiled_vector, cmap='autumn', vmin=0, vmax=5, alpha=0.45)
            axes[0,1].figure.dpi = dpi
            axes[0,1].axis('off')
            idx = f'{img_basename}_{laser_basename}'

            axes[1,1].axis('off')

            plt.subplots_adjust(hspace=0.01, wspace=0.01)
            plt.savefig(f"output/depth/{idx}.png",bbox_inches='tight', pad_inches=0.01, dpi=dpi)
            plt.close(fig)


def main():
    dirs = ['elb', 'erb', 'heracleia', 'mocap', 'nh', 'uc', 'wh']
    # dirs = ['elb']
    all_dfs = []
    for d in dirs:
        parent = Path(f"/home/edward/data/extracted/{d}")
        subfolders = [f for f in parent.iterdir() if f.is_dir()]
        for folder in subfolders:
            df = find_matching_pairs(folder)
            if df is not None and not df.empty:
                all_dfs.append(df)
    
    combined_df = pd.concat(all_dfs)
    combined_df.to_csv('/home/edward/data/trav/all_image_depth_pair.csv')
    return combined_df


if __name__ == '__main__':
    # main()
    draw_2_by_2_images()
