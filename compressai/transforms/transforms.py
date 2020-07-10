from . import functional as F_transforms

__all__ = [
    'RGB2YCbCr',
    'YCbCr2RGB',
    'YUV444To420',
    'YUV420To444',
]


class RGB2YCbCr:
    """RGB to YCbCr conversion.
    """
    def __call__(self, rgb):
        return F_transforms.rgb2ycbcr(rgb)

    def ___repr__(self):
        return f'{self.__class__.__name__}()'


class YCbCr2RGB:
    """YCbCr to RGB conversion.
    """
    def __call__(self, ycbcr):
        return F_transforms.ycbcr2rgb(ycbcr)

    def ___repr__(self):
        return f'{self.__class__.__name__}()'


class YUV444To420:
    """Convert a YUV 444 tensor to a 420 representation.
    """
    def __call__(self, yuv):
        return F_transforms.yuv_444_to_420(yuv)

    def ___repr__(self):
        return f'{self.__class__.__name__}()'


class YUV420To444:
    """Convert a YUV 420 to a 444 representation.
    """
    def __init__(self, return_tuple: bool = False):
        self.return_tuple = bool(return_tuple)

    def __call__(self, yuv):
        return F_transforms.yuv_420_to_444(yuv, return_tuple=self.return_tuple)

    def ___repr__(self):
        return f'{self.__class__.__name__}(return_tuple={self.return_tuple})'
