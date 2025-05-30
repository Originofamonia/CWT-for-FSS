from multiprocessing import Pool
from collections import defaultdict
from functools import partial
from typing import Callable, Dict, Iterable, List, Tuple, TypeVar
from tqdm import tqdm
import os
import cv2
import numpy as np

IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm']

A = TypeVar("A")
B = TypeVar("B")


def mmap_(fn: Callable[[A], B], iter: Iterable[A]) -> List[B]:
    return Pool().map(fn, iter)


def is_image_file(filename: str) -> bool:
    filename_lower = filename.lower()
    return any(filename_lower.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(
        data_root: str,
        data_list: str,
        class_list: List[int]
) -> Tuple[List[Tuple[str, str]], Dict[int, List[Tuple[str, str]]]]:

    if not os.path.isfile(data_list):
        raise (RuntimeError("Image list file do not exist: " + data_list + "\n"))
    '''
        Recovers all tupples (img_path, label_path) relevant to the current experiments (class_list
        is used as filter)

        input:
            data_root : Path to the data directory
            data_list : Path to the .txt file that contain the train/test split of images
            class_list: List of classes to keep
        returns:
            image_label_list: List of (img_path, label_path) that contain at least 1 object of a class
                              in class_list
            class_file_dict: Dict of all (img_path, label_path that contain at least 1 object of a class
                              in class_list, grouped by classes.
    '''
    image_label_list: List[Tuple[str, str]] = []
    list_read = open(data_list).readlines()

    print(f"Processing data for {class_list}")
    class_file_dict: Dict[int, List[Tuple[str, str]]] = defaultdict(list)

    process_partial = partial(process_image, data_root=data_root, class_list=class_list)

    for sublist, subdict in mmap_(process_partial, tqdm(list_read)):
        image_label_list += sublist

        for (k, v) in subdict.items():
            class_file_dict[k] += v

    return image_label_list, class_file_dict


def make_trav_dataset(
        data_root: str,
        scenes: List[str],
        split: str,
        class_list: List[int]
    ) -> Tuple[List[Tuple[str, str]], Dict[int, List[Tuple[str, str]]]]:
    '''
        Recovers all tupples (img_path, label_path) relevant to the current experiments (class_list
        is used as filter)

        input:
            data_root : Path to the data directory
            data_list : Path to the .txt file that contain the train/test split of images
            class_list: List of classes to keep
        returns:
            image_label_list: List of (img_path, label_path) that contain at least 1 object of a class
                              in class_list
            class_file_dict: Dict of all (img_path, label_path that contain at least 1 object of a class
                              in class_list, grouped by classes.
    '''
    if not os.path.isdir(data_root):
        raise (RuntimeError("Image dir do not exist: " + data_root + "\n"))
    scene_dirs = [os.path.join(data_root, x) for x in scenes]
    if split == 'train':
        images_dirs = [os.path.join(x, 'positive') for x in scene_dirs]
    else:  # 'val'
        images_dirs = [os.path.join(x, 'challenging') for x in scene_dirs]
    
    # images = []
    # targets = []
    image_target_pairs = []

    for scene in images_dirs:
        img_dir = os.path.join(scene, 'images')
        target_dir = os.path.join(scene, 'labels')

        for filename in os.listdir(img_dir):
            abs_img_path = os.path.join(img_dir, filename)
            target_name = filename.rstrip('.jpg') + '.npy'
            abs_label_path = os.path.join(target_dir, target_name)
            if os.path.exists(abs_img_path) and os.path.exists(abs_label_path):
                image_target_pairs.append((abs_img_path, abs_label_path))
                # images.append(abs_img_path)
                # targets.append(abs_label_path)

    image_label_list: List[Tuple[str, str]] = []
    # list_read = open(data_list).readlines()

    print(f"Processing data for {class_list}")
    class_file_dict: Dict[int, List[Tuple[str, str]]] = defaultdict(list)

    process_partial = partial(process_trav_image, class_list=class_list)

    for sublist, subdict in mmap_(process_partial, tqdm(image_target_pairs)):
        image_label_list += sublist

        for (k, v) in subdict.items():
            class_file_dict[k] += v

    return image_label_list, class_file_dict


def process_trav_image(
        pair: Tuple,
        # abs_img_path: str,
        # abs_label_path: str,
        # data_root: str,
        class_list: List
    ) -> Tuple[List, Dict]:
    '''
    Reads and parses a line corresponding to 1 file
    input:
        line : A line corresponding to 1 file, in the format path_to_image.jpg path_to_image.png
        data_root : Path to the data directory
        class_list: List of classes to keep

    '''
    # line = line.strip()
    # line_split = line.split(' ')
    # image_name = os.path.join(data_root, line_split[0])
    # label_name = os.path.join(data_root, line_split[1])
    item: Tuple[str, str] = (pair[0], pair[1])
    # label = cv2.imread(pair[1], cv2.IMREAD_GRAYSCALE)
    label = np.load(pair[1])
    label_class = np.unique(label).tolist()

    # if 0 in label_class:
    #     label_class.remove(0)
    # if 255 in label_class:
    #     label_class.remove(255)
    for label_class_ in label_class:
        assert label_class_ in list(range(0, 256)), label_class_

    c: int
    new_label_class = []
    for c in label_class:
        if c in class_list:
            tmp_label = np.zeros_like(label)
            target_pix = np.where(label == c)
            tmp_label[target_pix[0], target_pix[1]] = 1
            if tmp_label.sum() >= 2 * 32 * 32:
                new_label_class.append(c)

    label_class = new_label_class

    image_label_list: List[Tuple[str, str]] = []
    class_file_dict: Dict[int, List[Tuple[str, str]]] = defaultdict(list)

    if len(label_class) > 0:
        image_label_list.append(item)

        for c in label_class:
            assert c in class_list
            class_file_dict[c].append(item)

    return image_label_list, class_file_dict


def process_image(
        line: str,
        data_root: str,
        class_list: List
) -> Tuple[List, Dict]:
    '''
        Reads and parses a line corresponding to 1 file

        input:
            line : A line corresponding to 1 file, in the format path_to_image.jpg path_to_image.png
            data_root : Path to the data directory
            class_list: List of classes to keep

    '''
    line = line.strip()
    line_split = line.split(' ')
    image_name = os.path.join(data_root, line_split[0])
    label_name = os.path.join(data_root, line_split[1])
    item: Tuple[str, str] = (image_name, label_name)
    label = cv2.imread(label_name, cv2.IMREAD_GRAYSCALE)
    label_class = np.unique(label).tolist()

    if 0 in label_class:
        label_class.remove(0)
    if 255 in label_class:
        label_class.remove(255)
    for label_class_ in label_class:
        assert label_class_ in list(range(1, 81)), label_class_

    c: int
    new_label_class = []
    for c in label_class:
        if c in class_list:
            tmp_label = np.zeros_like(label)
            target_pix = np.where(label == c)
            tmp_label[target_pix[0], target_pix[1]] = 1
            if tmp_label.sum() >= 2 * 32 * 32:
                new_label_class.append(c)

    label_class = new_label_class

    image_label_list: List[Tuple[str, str]] = []
    class_file_dict: Dict[int, List[Tuple[str, str]]] = defaultdict(list)

    if len(label_class) > 0:
        image_label_list.append(item)

        for c in label_class:
            assert c in class_list
            class_file_dict[c].append(item)

    return image_label_list, class_file_dict
