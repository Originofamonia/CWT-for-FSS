"""
1. make 1 csv file to have all image and depth pairs
2. make videos containing those pairs to check if files match by list order
"""

import os
import pickle
import pandas as pd
from glob import glob
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


def draw_2_by_2_images(csv_file='/home/edward/data/trav/all_image_depth_pair.csv'):
    """
    1. read '/home/edward/data/trav/all_image_depth_pair.csv'
    2. draw a 2*2 figure: 1 img, 2 depth heatmap overlay img, 3 depth sector map
    3. save as images
    """
    # img_path, depth_path
    df = pd.read_csv(csv_file)
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
        img_basename = os.path.splitext(os.path.basename(row['image']))[0]
        laser_basename = os.path.splitext(os.path.basename(row['depth']))[0]
        rgb_image = plt.imread(row['image'])

        with open(row['depth'], 'rb') as f:
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
            plt.savefig(f"output/labeled/{idx}.png",bbox_inches='tight', pad_inches=0.01, dpi=dpi)
            plt.close(fig)


def make_video(image_dir="output/depth"):
    """
    Images are in f"output/depth/{idx}.png"
    """
    image_files = sorted(glob(f"{image_dir}/*.png"))

    clip = ImageSequenceClip(image_files, fps=24)
    clip.write_videofile("output/labeled/output_video.mp4", codec="libx264")


def split_labeled_and_unlabeled():
    """
    1. read mask folder as labeled done
    2. '/home/edward/data/trav/all_image_depth_pair.csv' contains all pairs
    3. all pairs - mask folder pairs = unlabeled pairs
    """
    img_pattern = "/home/edward/data/segmentation_indoor_images/*/*/images/*"
    label_pattern = "/home/edward/data/segmentation_indoor_images/*/*/labels/*.npy"
    img_files = glob(img_pattern)

    label_files = glob(label_pattern)
    label_set = set(label_files)
    records = []
    for img_path in img_files:
        img = Path(img_path)
        label_path = img.parent.parent / "labels" / (img.stem + ".npy")
        label_path_str = str(label_path)

        if label_path_str in label_set:
            records.append({"image": img_path, "label": label_path_str})
        else:
            records.append({"image": img_path, "label": None})

    df = pd.DataFrame(records)
    df.to_csv("/home/edward/data/segmentation_indoor_images/labeled_pairs.csv", index=False)


def append_depth_to_labeled_csv():
    """
    append depth data to "/home/edward/data/segmentation_indoor_images/labeled_pairs.csv"
    """
    df_labeled = pd.read_csv("/home/edward/data/segmentation_indoor_images/labeled_pairs.csv")  # [2553,2]
    # df_rgbd = pd.read_csv('/home/edward/data/trav/merged_rgbd.csv')  # [843,2]
    df_all = pd.read_csv('/home/edward/data/trav/all_image_depth_pair.csv')  # [94070,2]
    df_labeled['dir_stem'] = df_labeled['image'].apply(lambda x: f"{Path(x).parents[2].name}_{Path(x).stem}")
    df_all['dir_stem'] = df_all['img_path'].apply(lambda x: f"{Path(x).parents[2].name}_{Path(x).stem}")
    df_labeled = pd.merge(df_labeled, df_all[['dir_stem', 'depth_path']], on='dir_stem', how='left')
    df_labeled.drop(columns=['dir_stem'], inplace=True)
    df_labeled.rename(columns={'depth_path': 'depth'}, inplace=True)
    df_labeled.to_csv("/home/edward/data/segmentation_indoor_images/labeled_rgbd_pairs.csv", index=False)


def check_labeled_pairs():
    df = pd.read_csv("/home/edward/data/segmentation_indoor_images/labeled_rgbd_pairs.csv")
    df['img_stem'] = df['image'].apply(lambda x: Path(x).stem)
    df['depth_stem'] = df['depth'].apply(lambda x: Path(x).stem)
    intersection = set(df['img_stem']).intersection(df['depth_stem'])
    print(intersection)  # is 0?!!


def save_all_image_depth_pairs():
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
    combined_df.to_csv('/home/edward/data/trav/all_image_depth_pair.csv', index=False)
    return combined_df


if __name__ == '__main__':
    # save_all_image_depth_pairs()
    draw_2_by_2_images("/home/edward/data/segmentation_indoor_images/labeled_rgbd_pairs.csv")
    make_video("output/labeled")
    # split_labeled_and_unlabeled()
    # append_depth_to_labeled_csv()
    # check_labeled_pairs()
