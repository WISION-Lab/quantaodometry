import numpy as np
import torch


def binary2rgb(mean_binary_patch, factor=1.0, epsilon=1e-6, quantile=None):
    """Convert average binary patches to RGB
    Invert the process by which binary frames are simulated. The result can be either
    linear RGB values or sRGB values depending on how the binary frames were constructed.

    Assuming each binary patch was created by a Bernoulli process with p=1-exp(-factor*rgb),
    then the average of binary frames tends to p. We can therefore recover the original rgb
    values as -log(1-bin)/factor.

    Args:
        mean_binary_patch: Binary avg to convert to rgb
        factor: Arbitrary Brightness factor. Defaults to 1.0.
    :return:
        RGB value corresponding to specificed factor.
    """
    module = torch if torch.is_tensor(mean_binary_patch) else np
    intensity = -module.log(module.clip(1 - mean_binary_patch, epsilon, 1)) / factor

    if quantile is not None:
        intensity = intensity / module.quantile(intensity, quantile)
        intensity = module.clip(intensity, 0, 1)

    return intensity


def srgb_to_linearrgb(img):
    """Performs sRGB to linear RGB color space conversion by reversing gamma
    correction and obtaining values that represent the scene's intensities.

    Args:
        img: Tensor or np array to perform conversion.
    :returns:
        linear rgb image tensor or np array.

    """
    # https://github.com/blender/blender/blob/master/source/blender/blenlib/intern/math_color.c
    module, img = (torch, img.clone()) if torch.is_tensor(img) else (np, np.copy(img))
    mask = img < 0.04045
    img[mask] = module.clip(img[mask], 0.0, module.inf) / 12.92
    img[~mask] = ((img[~mask] + 0.055) / 1.055) ** 2.4
    return img


def linearrgb_to_srgb(img):
    """Performs linear RGB to sRGB inverse color space conversion to apply gamma correction or display purposes

    Args:
        img: Tensor or np array to perform conversion.
    :returns:
        srgb image tensor or np array.
    """
    # https://github.com/blender/blender/blob/master/source/blender/blenlib/intern/math_color.c
    module, img = (torch, img.clone()) if torch.is_tensor(img) else (np, np.copy(img))
    mask = img < 0.0031308
    img[img < 0.0] = 0.0
    img[mask] = img[mask] * 12.92
    img[~mask] = module.clip(1.055 * img[~mask] ** (1.0 / 2.4) - 0.055, 0.0, 1.0)
    return img
