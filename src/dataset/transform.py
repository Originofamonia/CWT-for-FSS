import random
import math
import numpy as np
import numbers
import collections
import cv2
import torch
import PIL
import PIL.ImageOps
import PIL.ImageEnhance
import PIL.ImageDraw
from PIL import Image
from torchvision import transforms
from collections.abc import Iterable

# ==================================================================================================
# Transforms have been borrowed from https://github.com/hszhao/semseg/blob/master/util/transform.py
# ==================================================================================================
PARAMETER_MAX = 10


class Compose(object):
    def __init__(self, segtransform):
        self.segtransform = segtransform

    def __call__(self, image, label=None):
        if label is None:
            for t in self.segtransform:
                image = t(image, None)
            return image
        else:
            for t in self.segtransform:
                image, label = t(image, label)
            return image, label


class ToTensorPIL(object):
    # Converts numpy.ndarray (H x W x C) to a torch.FloatTensor of shape (C x H x W).
    def __call__(self, image, label):
        image = transforms.ToTensor()(image)

        if label is not None:
            if not isinstance(label, np.ndarray):
                raise (RuntimeError("segtransform.ToTensor() only handle np.ndarray"
                                    "[eg: data readed by cv2.imread()].\n"))
            if not len(label.shape) == 2:
                raise (RuntimeError("segtransform.ToTensor() only handle np.ndarray labellabel with 2 dims.\n"))
            label = torch.from_numpy(label)
            if not isinstance(label, torch.LongTensor):
                label = label.long()
            return image, label
        else:
            return image


class ToTensor(object):
    # Converts numpy.ndarray (H x W x C) to a torch.FloatTensor of shape (C x H x W).
    def __call__(self, image, label):
        if not isinstance(image, np.ndarray):
            raise (RuntimeError("segtransform.ToTensor() only handle np.ndarray"
                                "[eg: data readed by cv2.imread()].\n"))
        if len(image.shape) > 3 or len(image.shape) < 2:
            raise (RuntimeError("segtransform.ToTensor() only handle np.ndarray with 3 dims or 2 dims.\n"))
        if len(image.shape) == 2:
            image = np.expand_dims(image, axis=2)

        image = torch.from_numpy(image.transpose((2, 0, 1)))
        if not isinstance(image, torch.FloatTensor):
            image = image.float().div(255)
        if label is not None:
            if not isinstance(label, np.ndarray):
                raise (RuntimeError("segtransform.ToTensor() only handle np.ndarray"
                                    "[eg: data readed by cv2.imread()].\n"))
            if not len(label.shape) == 2:
                raise (RuntimeError("segtransform.ToTensor() only handle np.ndarray labellabel with 2 dims.\n"))
            label = torch.from_numpy(label)
            if not isinstance(label, torch.LongTensor):
                label = label.long()
            return image, label
        else:
            return image


class Normalize(object):
    # Normalize tensor with mean and standard deviation along channel: channel = (channel - mean) / std
    def __init__(self, mean, std=None):
        if std is None:
            assert len(mean) > 0
        else:
            assert len(mean) == len(std)
        self.mean = mean
        self.std = std

    def __call__(self, image, label):
        if self.std is None:
            for t, m in zip(image, self.mean):
                t.sub_(m)
        else:
            for t, m, s in zip(image, self.mean, self.std):
                t.sub_(m).div_(s)
        if label is not None:
            return image, label
        else:
            return image


class Resize(object):
    # Resize the input to the given size, 'size' is a 2-element tuple or list in the order of (h, w).
    def __init__(self, size, padding=None):
        self.size = size
        self.padding = padding

    def __call__(self, image, label):

        def find_new_hw(ori_h, ori_w, test_size):
            if ori_h >= ori_w:
                ratio = test_size * 1.0 / ori_h
                new_h = test_size
                new_w = int(ori_w * ratio)
            elif ori_w > ori_h:
                ratio = test_size * 1.0 / ori_w
                new_h = int(ori_h * ratio)
                new_w = test_size

            if new_h % 8 != 0:
                new_h = (int(new_h / 8)) * 8
            else:
                new_h = new_h
            if new_w % 8 != 0:
                new_w = (int(new_w / 8)) * 8
            else:
                new_w = new_w
            return new_h, new_w

        # Step 1: resize while keeping the h/w ratio. The largest side (i.e height or width) is reduced to $size.
        #                                             The other is reduced accordingly
        test_size = self.size
        new_h, new_w = find_new_hw(image.shape[0], image.shape[1], test_size)

        image_crop = cv2.resize(image, dsize=(int(new_w), int(new_h)),
                                interpolation=cv2.INTER_LINEAR)

        # Step 2: Pad wtih 0 whatever needs to be padded to get a ($size, $size) image
        back_crop = np.zeros((test_size, test_size, 3))
        if self.padding:
            back_crop[:, :, 0] = self.padding[0]
            back_crop[:, :, 1] = self.padding[1]
            back_crop[:, :, 2] = self.padding[2]
        back_crop[:new_h, :new_w, :] = image_crop
        image = back_crop

        # Step 3: Do the same for the label (the padding is 255)
        if label is not None:
            s_mask = label
            new_h, new_w = find_new_hw(s_mask.shape[0], s_mask.shape[1], test_size)
            s_mask = cv2.resize(s_mask.astype(np.float32), dsize=(int(new_w), int(new_h)),
                                interpolation=cv2.INTER_NEAREST)
            back_crop_s_mask = np.ones((test_size, test_size)) * 255
            back_crop_s_mask[:new_h, :new_w] = s_mask
            label = back_crop_s_mask

            return image, label
        else:
            return image, new_h, new_w


class RandScale(object):
    # Randomly resize image & label with scale factor in [scale_min, scale_max]
    def __init__(self, scale, aspect_ratio=None):
        assert (isinstance(scale, Iterable) and len(scale) == 2)
        if isinstance(scale, Iterable) and len(scale) == 2 \
                and isinstance(scale[0], numbers.Number) and isinstance(scale[1], numbers.Number) \
                and 0 < scale[0] < scale[1]:
            self.scale = scale
        else:
            raise (RuntimeError("segtransform.RandScale() scale param error.\n"))
        if aspect_ratio is None:
            self.aspect_ratio = aspect_ratio
        elif isinstance(aspect_ratio, Iterable) \
                and len(aspect_ratio) == 2 \
                and isinstance(aspect_ratio[0], numbers.Number) \
                and isinstance(aspect_ratio[1], numbers.Number) \
                and 0 < aspect_ratio[0] < aspect_ratio[1]:
            self.aspect_ratio = aspect_ratio
        else:
            raise (RuntimeError("segtransform.RandScale() aspect_ratio param error.\n"))

    def __call__(self, image, label):
        temp_scale = self.scale[0] + (self.scale[1] - self.scale[0]) * random.random()
        temp_aspect_ratio = 1.0
        if self.aspect_ratio is not None:
            temp_aspect_ratio = self.aspect_ratio[0] + (self.aspect_ratio[1] - self.aspect_ratio[0]) * random.random()
            temp_aspect_ratio = math.sqrt(temp_aspect_ratio)
        scale_factor_x = temp_scale * temp_aspect_ratio
        scale_factor_y = temp_scale / temp_aspect_ratio
        image = cv2.resize(image, None, fx=scale_factor_x, fy=scale_factor_y,
                           interpolation=cv2.INTER_LINEAR)
        label = cv2.resize(label, None, fx=scale_factor_x, fy=scale_factor_y,
                           interpolation=cv2.INTER_NEAREST)
        return image, label


class Crop(object):
    """Crops the given ndarray image (H*W*C or H*W).
    Args:
        size (sequence or int): Desired output size of the crop. If size is an
        int instead of sequence like (h, w), a square crop (size, size) is made.
    """
    def __init__(self, size, crop_type='center', padding=None, ignore_label=255):
        if isinstance(size, int):
            self.crop_h = size
            self.crop_w = size
        elif isinstance(size, Iterable) and len(size) == 2 \
                and isinstance(size[0], int) and isinstance(size[1], int) \
                and size[0] > 0 and size[1] > 0:
            self.crop_h = size[0]
            self.crop_w = size[1]
        else:
            raise (RuntimeError("crop size error.\n"))
        if crop_type == 'center' or crop_type == 'rand':
            self.crop_type = crop_type
        else:
            raise (RuntimeError("crop type error: rand | center\n"))
        if padding is None:
            self.padding = padding
        elif isinstance(padding, list):
            if all(isinstance(i, numbers.Number) for i in padding):
                self.padding = padding
            else:
                raise (RuntimeError("padding in Crop() should be a number list\n"))
            if len(padding) != 3:
                raise (RuntimeError("padding channel is not equal with 3\n"))
        else:
            raise (RuntimeError("padding in Crop() should be a number list\n"))
        if isinstance(ignore_label, int):
            self.ignore_label = ignore_label
        else:
            raise (RuntimeError("ignore_label should be an integer number\n"))

    def __call__(self, image, label):
        h, w = image.shape[:2]
        pad_h = max(self.crop_h - h, 0)
        pad_w = max(self.crop_w - w, 0)
        pad_h_half = int(pad_h / 2)
        pad_w_half = int(pad_w / 2)
        if pad_h > 0 or pad_w > 0:
            if self.padding is None:
                raise (RuntimeError("segtransform.Crop() need padding while padding argument is None\n"))
            image = cv2.copyMakeBorder(image, pad_h_half, pad_h - pad_h_half, pad_w_half,
                                       pad_w - pad_w_half, cv2.BORDER_CONSTANT, value=self.padding)
            # image = np.zeros(3,)
            if label is not None:
                label = cv2.copyMakeBorder(label, pad_h_half, pad_h - pad_h_half, pad_w_half,
                                           pad_w - pad_w_half, cv2.BORDER_CONSTANT, value=self.ignore_label)
        h, w = image.shape[:2]
        if self.crop_type == 'rand':
            h_off = random.randint(0, h - self.crop_h)
            w_off = random.randint(0, w - self.crop_w)
        else:
            h_off = int((h - self.crop_h) / 2)
            w_off = int((w - self.crop_w) / 2)
        image = image[h_off:h_off+self.crop_h, w_off:w_off+self.crop_w]
        if label is not None:
            label = label[h_off:h_off+self.crop_h, w_off:w_off+self.crop_w]
            return image, label
        else:
            return image


class RandRotate(object):
    # Randomly rotate image & label with rotate factor in [rotate_min, rotate_max]
    def __init__(self, rotate, padding, ignore_label=255, p=0.5):
        assert (isinstance(rotate, Iterable) and len(rotate) == 2)
        if isinstance(rotate[0], numbers.Number) and isinstance(rotate[1], numbers.Number) \
                and rotate[0] < rotate[1]:
            self.rotate = rotate
        else:
            raise (RuntimeError("segtransform.RandRotate() scale param error.\n"))
        assert padding is not None
        assert isinstance(padding, list) and len(padding) == 3
        if all(isinstance(i, numbers.Number) for i in padding):
            self.padding = padding
        else:
            raise (RuntimeError("padding in RandRotate() should be a number list\n"))
        assert isinstance(ignore_label, int)
        self.ignore_label = ignore_label
        self.p = p

    def __call__(self, image, label):
        if random.random() < self.p:
            angle = self.rotate[0] + (self.rotate[1] - self.rotate[0]) * random.random()
            h, w = label.shape
            matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)
            image = cv2.warpAffine(image, matrix, (w, h), flags=cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_CONSTANT, borderValue=self.padding)
            label = cv2.warpAffine(label, matrix, (w, h), flags=cv2.INTER_NEAREST,
                                   borderMode=cv2.BORDER_CONSTANT, borderValue=self.ignore_label)
        return image, label


class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image, label):
        if random.random() < self.p:
            image = cv2.flip(image, 1)
            label = cv2.flip(label, 1)
        return image, label


class RandomVerticalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image, label):
        if random.random() < self.p:
            image = cv2.flip(image, 0)
            label = cv2.flip(label, 0)
        return image, label


class RandomGaussianBlur(object):
    def __init__(self, radius=5):
        self.radius = radius

    def __call__(self, image, label):
        if random.random() < 0.5:
            image = cv2.GaussianBlur(image, (self.radius, self.radius), 0)
        return image, label


class Contrast(object):
    def __init__(self, v=0.9, max_v=0.05, bias=0):
        self.v = _float_parameter(v, max_v) + bias

    def __call__(self, image, label):
        image = Image.fromarray(np.uint8(image)).convert('RGB')
        return PIL.ImageEnhance.Contrast(image).enhance(self.v), label


class Brightness(object):
    def __init__(self, v=1.8, max_v=0.1, bias=0):
        self.v = _float_parameter(v, max_v) + bias

    def __call__(self, image, label):
        image = Image.fromarray(np.uint8(image)).convert('RGB')
        return PIL.ImageEnhance.Brightness(image).enhance(self.v), label


class Sharpness(object):
    def __init__(self, v=0.9, max_v=0.05, bias=0):
        self.v = _float_parameter(v, max_v) + bias

    def __call__(self, image, label):
        image = Image.fromarray(np.uint8(image)).convert('RGB')
        return PIL.ImageEnhance.Sharpness(image).enhance(self.v), label


class AutoContrast(object):
    def __call__(self, image, label):
        image = Image.fromarray(np.uint8(image)).convert('RGB')
        return PIL.ImageOps.autocontrast(image), label


def _float_parameter(v, max_v):
    return float(v) * max_v / PARAMETER_MAX


class RGB2BGR(object):
    # Converts image from RGB order to BGR order, for model initialized from Caffe
    def __call__(self, image, label):
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        return image, label


class BGR2RGB(object):
    # Converts image from BGR order to RGB order, for model initialized from Pytorch
    def __call__(self, image, label):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image, label
