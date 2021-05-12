import functools
import pathlib
from typing import Any, Callable, Dict, Optional, Tuple, Union

import torchvision
from torch import nn
from torchvision import transforms

from flash.data.data_source import DefaultDataKeys, DefaultDataSources
from flash.data.process import Preprocess
from flash.data.transforms import ApplyToKeys
from flash.vision.classification import ImageClassificationData, ImageClassificationPreprocess

from ._utils import raise_not_supported

__all__ = ["StyleTransferPreprocess", "StyleTransferData"]


def _apply_to_input(default_transforms_fn) -> Callable[..., Dict[str, ApplyToKeys]]:

    @functools.wraps(default_transforms_fn)
    def wrapper(*args: Any, **kwargs: Any) -> Dict[str, ApplyToKeys]:
        default_transforms = default_transforms_fn(*args, **kwargs)
        return {hook: ApplyToKeys(DefaultDataKeys.INPUT, transform) for hook, transform in default_transforms.items()}

    return wrapper


class StyleTransferPreprocess(ImageClassificationPreprocess):

    def __init__(
        self,
        train_transform: Optional[Union[Dict[str, Callable]]] = None,
        predict_transform: Optional[Union[Dict[str, Callable]]] = None,
        image_size: Union[int, Tuple[int, int]] = 256,
    ):
        if isinstance(image_size, int):
            image_size = (image_size, image_size)
        super().__init__(
            train_transform=train_transform,
            predict_transform=predict_transform,
            image_size=image_size,
        )

    @property
    @_apply_to_input
    def default_train_transforms(self) -> Dict[str, Callable]:
        return dict(
            to_tensor_transform=torchvision.transforms.ToTensor(),
            per_sample_transform_on_device=nn.Sequential(
                transforms.Resize(min(self.image_size)),
                transforms.CenterCrop(self.image_size),
            ),
        )

    @property
    def default_val_transforms(self) -> None:
        # Style transfer doesn't support a validation phase, so we return nothing here
        return None

    @property
    def default_test_transforms(self) -> None:
        # Style transfer doesn't support a test phase, so we return nothing here
        return None

    @property
    @_apply_to_input
    def default_predict_transforms(self) -> Dict[str, Callable]:

        return dict(
            to_tensor_transform=torchvision.transforms.ToTensor(),
            per_sample_transform_on_device=transforms.Resize(min(self.image_size)),
        )


class StyleTransferData(ImageClassificationData):
    preprocess_cls = StyleTransferPreprocess

    @classmethod
    def from_folders(
        cls,
        train_folder: Optional[Union[str, pathlib.Path]] = None,
        predict_folder: Optional[Union[str, pathlib.Path]] = None,
        train_transform: Optional[Union[str, Dict]] = None,
        predict_transform: Optional[Union[str, Dict]] = None,
        preprocess: Optional[Preprocess] = None,
        **kwargs: Any,
    ) -> "StyleTransferData":
        if any(param in kwargs for param in ("val_folder", "val_transform")):
            raise_not_supported("validation")
        if any(param in kwargs for param in ("test_folder", "test_transform")):
            raise_not_supported("test")

        preprocess = preprocess or cls.preprocess_cls(train_transform, predict_transform)

        return cls.from_data_source(
            DefaultDataSources.PATHS,
            train_data=train_folder,
            predict_data=predict_folder,
            preprocess=preprocess,
            **kwargs,
        )
