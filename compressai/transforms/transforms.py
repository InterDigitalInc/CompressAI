from . import functional as F_transforms

__all__ = [
    "RGB2YCbCr",
    "YCbCr2RGB",
    "YUV444To420",
    "YUV420To444",
]


class RGB2YCbCr:
    """Convert a RGB tensor to YCbCr.
    The tensor is expected to be in the [0, 1] floating point range, with a
    shape of (3xHxW) or (Nx3xHxW).
    """

    def __call__(self, rgb):
        """
        Args:
            rgb (torch.Tensor): 3D or 4D floating point RGB tensor

        Returns:
            ycbcr(torch.Tensor): converted tensor
        """
        return F_transforms.rgb2ycbcr(rgb)

    def __repr__(self):
        return f"{self.__class__.__name__}()"


class YCbCr2RGB:
    """Convert a YCbCr tensor to RGB.
    The tensor is expected to be in the [0, 1] floating point range, with a
    shape of (3xHxW) or (Nx3xHxW).
    """

    def __call__(self, ycbcr):
        """
        Args:
            ycbcr(torch.Tensor): 3D or 4D floating point RGB tensor

        Returns:
            rgb(torch.Tensor): converted tensor
        """
        return F_transforms.ycbcr2rgb(ycbcr)

    def __repr__(self):
        return f"{self.__class__.__name__}()"


class YUV444To420:
    """Convert a YUV 444 tensor to a 420 representation.

    Args:
        mode (str): algorithm used for downsampling: ``'avg_pool'``. Default
            ``'avg_pool'``

    Example:
        >>> x = torch.rand(1, 3, 32, 32)
        >>> y, u, v = YUV444To420()(x)
        >>> y.size()  # 1, 1, 32, 32
        >>> u.size()  # 1, 1, 16, 16
    """

    def __init__(self, mode: str = "avg_pool"):
        self.mode = str(mode)

    def __call__(self, yuv):
        """
        Args:
            yuv (torch.Tensor or (torch.Tensor, torch.Tensor, torch.Tensor)):
                444 input to be downsampled. Takes either a (Nx3xHxW) tensor or
                a tuple of 3 (Nx1xHxW) tensors.

        Returns:
            (torch.Tensor, torch.Tensor, torch.Tensor): Converted 420
        """
        return F_transforms.yuv_444_to_420(yuv, mode=self.mode)

    def __repr__(self):
        return f"{self.__class__.__name__}()"


class YUV420To444:
    """Convert a YUV 420 input to a 444 representation.

    Args:
        mode (str): algorithm used for upsampling: ``'bilinear'`` | ``'nearest'``.
            Default ``'bilinear'``
        return_tuple (bool): return input as tuple of tensors instead of a
            concatenated tensor, 3 (Nx1xHxW) tensors instead of one (Nx3xHxW)
            tensor (default: False)

    Example:
        >>> y = torch.rand(1, 1, 32, 32)
        >>> u, v = torch.rand(1, 1, 16, 16), torch.rand(1, 1, 16, 16)
        >>> x = YUV420To444()((y, u, v))
        >>> x.size()  # 1, 3, 32, 32
    """

    def __init__(self, mode: str = "bilinear", return_tuple: bool = False):
        self.mode = str(mode)
        self.return_tuple = bool(return_tuple)

    def __call__(self, yuv):
        """
        Args:
            yuv (torch.Tensor, torch.Tensor, torch.Tensor): 420 input frames in
                (Nx1xHxW) format

        Returns:
            (torch.Tensor or (torch.Tensor, torch.Tensor, torch.Tensor)): Converted
                444
        """
        return F_transforms.yuv_420_to_444(yuv, return_tuple=self.return_tuple)

    def __repr__(self):
        return f"{self.__class__.__name__}(return_tuple={self.return_tuple})"
