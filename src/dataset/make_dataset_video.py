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
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from moviepy import ImageSequenceClip
from multiprocessing import Pool


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
    1. read csv_file
    2. draw a 2*2 figure: 1 img, 2 depth heatmap overlay img, 3 depth sector map
    3. save as images
    """
    # img_path, depth_path
    df = pd.read_csv(csv_file)
    df['label'] = df['label'].apply(lambda x: os.path.join('/home/edward/data/trav/unlabeled_masks', os.path.basename(x)))
    df.to_csv(csv_file, index=False)
    exit()
    colors = ['#00000000', 'lime']
    cmap = ListedColormap(colors)
    alpha = 0.6
    dpi = 200
    sector_left = -45 #-135
    sector_right = 45 # 135
    angle_min = -26
    angle_max = 36
    angle_rad_min = np.deg2rad(angle_min)
    angle_rad_max = np.deg2rad(angle_max)
    min_pct = (angle_min+45)/90  # percentile for cropping
    max_pct = (angle_max+45)/90
    
    for i, row in tqdm(df.iterrows(), total=len(df)):
        img_basename = os.path.splitext(os.path.basename(row['image']))[0]
        laser_basename = os.path.splitext(os.path.basename(row['depth']))[0]
        rgb_image = plt.imread(row['image'])
        mask = np.load(row['label'])

        with open(row['depth'], 'rb') as f:
            data = pickle.load(f)

            vector = np.array(data['ranges'][::-1])[540:900]
            angles = np.linspace(np.deg2rad(sector_left), np.deg2rad(sector_right), len(vector), endpoint=False)

            fig, axes = plt.subplots(2, 2)
            axes[0,0].imshow(rgb_image)
            axes[0,0].figure.dpi = dpi
            axes[0,0].axis('off')

            axes[1,0].imshow(rgb_image)
            axes[1,0].imshow(mask, cmap=cmap, alpha=alpha)
            axes[1,0].figure.dpi = dpi
            axes[1,0].axis('off')

            axes[1,1] = plt.subplot(224, projection='polar')
            axes[1,1].plot(angles, vector)
            axes[1,1].plot([angle_rad_max, angle_rad_max], [0, 5.1], color='red', linestyle='--')
            axes[1,1].plot([angle_rad_min, angle_rad_min], [0, 5.1], color='blue', linestyle='--')
            axes[1,1].set_thetamin(sector_left)
            axes[1,1].set_thetamax(sector_right)
            axes[1,1].set_theta_zero_location('N')
            axes[1,1].set_xticks(np.pi/180. * np.linspace(sector_left, sector_right, 10, endpoint=False))
            axes[1,1].figure.axes[3].set_axis_off()
            # axes[1,1].set_frame_on(False)

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

            plt.subplots_adjust(hspace=0.01, wspace=0.01)
            plt.savefig(f"output/unlabeled/{idx}.png",bbox_inches='tight', pad_inches=0.01, dpi=dpi)
            plt.close(fig)


def process_chunk(df_chunk):
    colors = ['#00000000', 'lime']
    cmap = ListedColormap(colors)
    alpha = 0.6
    dpi = 200
    sector_left = -45
    sector_right = 45
    angle_min = -26
    angle_max = 36
    angle_rad_min = np.deg2rad(angle_min)
    angle_rad_max = np.deg2rad(angle_max)
    min_pct = (angle_min + 45) / 90
    max_pct = (angle_max + 45) / 90

    for _, row in tqdm(df_chunk.iterrows(), total=len(df_chunk)):
        img_basename = os.path.splitext(os.path.basename(row['image']))[0]
        laser_basename = os.path.splitext(os.path.basename(row['depth']))[0]
        rgb_image = plt.imread(row['image'])
        mask = np.load(row['label'])

        with open(row['depth'], 'rb') as f:
            data = pickle.load(f)

            vector = np.array(data['ranges'][::-1])[540:900]
            angles = np.linspace(np.deg2rad(sector_left), np.deg2rad(sector_right), len(vector), endpoint=False)

            fig, axes = plt.subplots(2, 2)
            axes[0,0].imshow(rgb_image)
            axes[0,0].figure.dpi = dpi
            axes[0,0].axis('off')

            axes[1,0].imshow(rgb_image)
            axes[1,0].imshow(mask, cmap=cmap, alpha=alpha)
            axes[1,0].figure.dpi = dpi
            axes[1,0].axis('off')

            # Remove the default axes[1,1] and replace with polar subplot
            fig.delaxes(axes[1,1])
            polar_ax = fig.add_subplot(2, 2, 4, projection='polar')
            polar_ax.plot(angles, vector)
            polar_ax.plot([angle_rad_max, angle_rad_max], [0, 5.1], color='red', linestyle='--')
            polar_ax.plot([angle_rad_min, angle_rad_min], [0, 5.1], color='blue', linestyle='--')
            polar_ax.set_thetamin(sector_left)
            polar_ax.set_thetamax(sector_right)
            polar_ax.set_theta_zero_location('N')
            polar_ax.set_xticks(np.pi/180. * np.linspace(sector_left, sector_right, 10, endpoint=False))
            # Hide polar spine rectangle but keep the polar grid and ticks
            for spine in polar_ax.spines.values():
                spine.set_visible(False)

            depth_vector = vector[int(min_pct * len(vector)):int(max_pct * len(vector))][::-1]
            x_original = np.linspace(0, 1, len(depth_vector))
            f_interp = interp1d(x_original, depth_vector)
            x_new = np.linspace(0, 1, rgb_image.shape[1])
            interp_vector = f_interp(x_new)
            tiled_vector = np.tile(interp_vector, (rgb_image.shape[0], 1))

            axes[0,1].imshow(rgb_image)
            axes[0,1].imshow(tiled_vector, cmap='autumn', vmin=0, vmax=5, alpha=0.45)
            axes[0,1].figure.dpi = dpi
            axes[0,1].axis('off')

            idx = f'{img_basename}_{laser_basename}'
            # os.makedirs("output/unlabeled", exist_ok=True)
            plt.subplots_adjust(hspace=0.01, wspace=0.01)
            plt.savefig(f"output/unlabeled/{idx}.png", bbox_inches='tight', pad_inches=0.01, dpi=dpi)
            plt.close(fig)


def draw_2_by_2_images_parallel(csv_file, num_parts=4):
    df = pd.read_csv(csv_file)
    df_chunks = np.array_split(df, num_parts)
    with Pool(num_parts) as pool:
        pool.map(process_chunk, df_chunks)


def make_video(image_dir="output/depth"):
    """
    Images are in f"output/depth/{idx}.png"
    """
    image_files = sorted(glob(f"{image_dir}/*.png"))

    clip = ImageSequenceClip(image_files, fps=24)
    clip.write_videofile("output/unlabeled_dataset.mp4", codec="libx264")


def make_labeled_dataset():
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


def split_unlabeled_dataset():
    labeled_df = pd.read_csv('/home/edward/data/segmentation_indoor_images/labeled_rgbd_pairs.csv')
    labeled_df = labeled_df[labeled_df['label'].notna() & (labeled_df['label'] != '')]
    all_df = pd.read_csv('/home/edward/data/trav/all_image_depth_pair.csv')
    unlabeled_df = all_df[~all_df["depth_path"].isin(labeled_df["depth"])]
    unlabeled_df.to_csv('/home/edward/data/trav/unlabeled_image_depth_pair.csv', index=False)


def save_rgbd_and_masks():
    """
    save labeled RGB-D and masks for arch figure
    """
    df = pd.read_csv("/home/edward/data/segmentation_indoor_images/labeled_rgbd_pairs.csv")
    colors = ['gray', 'lime']
    cmap = ListedColormap(colors)
    alpha = 0.6
    dpi = 200
    sector_left = -45 #-135
    sector_right = 45 # 135
    angle_min = -26
    angle_max = 36
    angle_rad_min = np.deg2rad(angle_min)
    angle_rad_max = np.deg2rad(angle_max)
    min_pct = (angle_min+45)/90  # percentile for cropping
    max_pct = (angle_max+45)/90
    
    for i, row in tqdm(df.iterrows(), total=len(df)):
        img_basename = os.path.splitext(os.path.basename(row['image']))[0]
        laser_basename = os.path.splitext(os.path.basename(row['depth']))[0]
        rgb_image = plt.imread(row['image'])
        mask = np.load(row['label'])

        with open(row['depth'], 'rb') as f:
            data = pickle.load(f)

            vector = np.array(data['ranges'][::-1])[540:900]
            angles = np.linspace(np.deg2rad(sector_left), np.deg2rad(sector_right), len(vector), endpoint=False)

        # Save RGB image
        fig_rgb, ax_rgb = plt.subplots()
        ax_rgb.imshow(rgb_image)
        ax_rgb.axis('off')
        fig_rgb.dpi = dpi
        plt.savefig(f"output/arch/{img_basename}_rgb.png", bbox_inches='tight', pad_inches=0.01, dpi=dpi)
        plt.close(fig_rgb)

        # Create foreground and background masks
        foreground_mask = np.where(mask == 1, mask, 0)
        background_mask = np.where(mask == 0, 1, 0)

        # Save label mask
        fig_mask, ax_mask = plt.subplots()
        ax_mask.imshow(mask, cmap=cmap, alpha=alpha)
        ax_mask.axis('off')
        fig_mask.dpi = dpi
        plt.savefig(f"output/arch/{img_basename}_{laser_basename}_mask.png", bbox_inches='tight', pad_inches=0.01, dpi=dpi)
        plt.close(fig_mask)

        # Save background mask with 'plasma' colormap
        fig_bg, ax_bg = plt.subplots()
        ax_bg.imshow(background_mask, cmap='plasma', alpha=alpha)
        ax_bg.axis('off')
        fig_bg.dpi = dpi
        plt.savefig(f"output/arch/{img_basename}_{laser_basename}_background.png", bbox_inches='tight', pad_inches=0.01, dpi=dpi)
        plt.close(fig_bg)

        # Save depth sector (polar plot)
        fig_polar = plt.figure()
        ax_polar = fig_polar.add_subplot(111, projection='polar')
        ax_polar.plot(angles, vector)
        ax_polar.plot([angle_rad_max, angle_rad_max], [0, 5.1], color='red', linestyle='--')
        ax_polar.plot([angle_rad_min, angle_rad_min], [0, 5.1], color='blue', linestyle='--')
        ax_polar.set_thetamin(sector_left)
        ax_polar.set_thetamax(sector_right)
        ax_polar.set_theta_zero_location('N')
        ax_polar.set_xticks(np.pi/180. * np.linspace(sector_left, sector_right, 10, endpoint=False))
        fig_polar.dpi = dpi
        plt.savefig(f"output/arch/{img_basename}_{laser_basename}_depth.png", bbox_inches='tight', pad_inches=0.01, dpi=dpi)
        plt.close(fig_polar)


if __name__ == '__main__':
    # save_all_image_depth_pairs()
    # draw_2_by_2_images("/home/edward/data/trav/unlabeled_masks.csv")
    # make_video("output/unlabeled")
    # make_labeled_dataset()
    # append_depth_to_labeled_csv()
    # check_labeled_pairs()
    # split_unlabeled_dataset()
    # draw_2_by_2_images_parallel("/home/edward/data/trav/unlabeled_masks.csv", num_parts=8)
    save_rgbd_and_masks()
