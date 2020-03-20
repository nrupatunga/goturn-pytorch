"""
File: image_io.py
Author: Nrupatunga
Email: nrupatunga.s@byjus.com
Github: https://github.com/nrupatunga
Description: Image IO
"""

import numpy as np
import torch
from PIL import Image
from torchvision import get_image_backend

try:
    import accimage
except ImportError:
    accimage = None


def _pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def _accimage_loader(path):
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return _pil_loader(path)


def _is_pil_image(img):
    if accimage is not None:
        return isinstance(img, (Image.Image, accimage.Image))
    else:
        return isinstance(img, Image.Image)


def _is_numpy_image(img):
    return isinstance(img, np.ndarray) and (img.ndim in {2, 3})


def _is_tensor_image(img):
    return torch.is_tensor(img) and img.ndimension() == 3


def image_to_tensor(pic, scale=255):
    """Convert a ``PIL Image`` or ``numpy.ndarray`` to tensor.
    See ``ToTensor`` for more details.
    Args:
    pic (PIL Image or numpy.ndarray): Image to be converted to tensor.
    Returns:
    Tensor: Converted image.
    """

    if not(_is_pil_image(pic) or _is_numpy_image(pic)):
        raise TypeError('pic should be PIL Image or ndarray. Got {}'.format(type(pic)))

    if isinstance(pic, np.ndarray):
        # handle numpy array
        if pic.ndim == 2:
            pic = pic[:, :, None]

        img = torch.from_numpy(pic.transpose((2, 0, 1)))
        # backward compatibility
        if isinstance(img, torch.ByteTensor):
            return img.float().div(scale)
        else:
            return img

    if accimage is not None and isinstance(pic, accimage.Image):
        nppic = np.zeros([pic.channels, pic.height, pic.width], dtype=np.float32)
        pic.copyto(nppic)
        return torch.from_numpy(nppic)

    # handle PIL Image
    if pic.mode == 'I':
        img = torch.from_numpy(np.array(pic, np.int32, copy=False))
    elif pic.mode == 'I;16':
        img = torch.from_numpy(np.array(pic, np.int16, copy=False))
    elif pic.mode == 'F':
        img = torch.from_numpy(np.array(pic, np.float32, copy=False))
    elif pic.mode == '1':
        img = scale * torch.from_numpy(np.array(pic, np.uint8, copy=False))
    else:
        img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
    # PIL image mode: L, P, I, F, RGB, YCbCr, RGBA, CMYK
    if pic.mode == 'YCbCr':
        nchannel = 3
    elif pic.mode == 'I;16':
        nchannel = 1
    else:
        nchannel = len(pic.mode)
    img = img.view(pic.size[1], pic.size[0], nchannel)
    # put it from HWC to CHW format
    # this transpose takes 80% of the loading time/CPU
    img = img.transpose(0, 1).transpose(0, 2).contiguous()
    if isinstance(img, torch.ByteTensor):
        return img.float().div(scale)
    else:
        return img


def load_grayscale(path):
    if get_image_backend() == 'accimage':
        img = _accimage_loader(path)
    else:
        img = _pil_loader(path)

    channels = img.split()
    if len(channels) == 3:
        img = Image.merge("RGB", [channels[2], channels[1], channels[0]])
    return img.convert('L')


def load(path, image_size=None):
    if get_image_backend() == 'accimage':
        img = _accimage_loader(path)
    else:
        img = _pil_loader(path)

    channels = img.split()
    if len(channels) == 1:
        img = img.convert('L')
    else:  # Make sure it is BGR for caffenet
        img = Image.merge("RGB", [channels[2], channels[1], channels[0]])

    if image_size is not None:
        if (image_size[0] == 1 and len(channels) == 3):
            img = img.convert('L')
        if image_size[1] == img.width and image_size[2] == img.height:
            return img

        return img.resize((image_size[1], image_size[2]), Image.BILINEAR)
    else:
        return img


def resize(img, size):
    """resize numpy array
    """

    if _is_numpy_image(img):
        img = Image.fromarray(img)

    return img.resize(size, Image.BILINEAR)
