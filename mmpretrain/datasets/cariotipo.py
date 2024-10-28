# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional, Union

from mmengine import fileio
from mmengine.logging import MMLogger

from mmpretrain.registry import DATASETS
from .categories import IMAGENET_CATEGORIES
from .custom import CustomDataset

@DATASETS.register_module()
class Cariotipo(CustomDataset):
    """ImageNet21k Dataset.

    Since the dataset ImageNet21k is extremely big, contains 21k+ classes
    and 1.4B files. We won't provide the default categories list. Please
    specify it from the ``classes`` argument.
    The dataset directory structure is as follows,

    ImageNet21k dataset directory ::

        imagenet21k
        ├── train
        │   ├──class_x
        |   |   ├── x1.jpg
        |   |   ├── x2.jpg
        |   |   └── ...
        │   ├── class_y
        |   |   ├── y1.jpg
        |   |   ├── y2.jpg
        |   |   └── ...
        |   └── ...
        └── meta
            └── train.txt


    Args:
        data_root (str): The root directory for ``data_prefix`` and
            ``ann_file``. Defaults to ''.
        data_prefix (str | dict): Prefix for training data. Defaults to ''.
        ann_file (str): Annotation file path. Defaults to ''.
        metainfo (dict, optional): Meta information for dataset, such as class
            information. Defaults to None.
        multi_label (bool): Not implement by now. Use multi label or not.
            Defaults to False.
        **kwargs: Other keyword arguments in :class:`CustomDataset` and
            :class:`BaseDataset`.

    Examples:
        >>> from mmpretrain.datasets import ImageNet21k
        >>> train_dataset = ImageNet21k(data_root='data/imagenet21k', split='train')
        >>> train_dataset
        Dataset ImageNet21k
            Number of samples:  14197088
            Annotation file:    data/imagenet21k/meta/train.txt
            Prefix of images:   data/imagenet21k/train
    """  # noqa: E501

    IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif')

    def __init__(self,
                 data_root: str = '',
                 split: str = '',
                 data_prefix: Union[str, dict] = '',
                 ann_file: str = '',
                 metainfo: Optional[dict] = None,
                 multi_label: bool = False,
                 **kwargs):
        if multi_label:
            raise NotImplementedError(
                'The `multi_label` option is not supported by now.')
        self.multi_label = multi_label

        if split:
            splits = ['train']
            assert split in splits, \
                f"The split must be one of {splits}, but get '{split}'.\
                If you want to specify your own validation set or test set,\
                please set split to None."

            self.split = split
            data_prefix = split if data_prefix == '' else data_prefix

            if not ann_file:
                _ann_path = fileio.join_path(data_root, 'meta', f'{split}.txt')
                if fileio.exists(_ann_path):
                    ann_file = fileio.join_path('meta', f'{split}.txt')

        logger = MMLogger.get_current_instance()

        if not ann_file:
            logger.warning(
                'The ImageNet21k dataset is large, and scanning directory may '
                'consume long time. Considering to specify the `ann_file` to '
                'accelerate the initialization.')

        kwargs = {'extensions': self.IMG_EXTENSIONS, **kwargs}
        super().__init__(
            data_root=data_root,
            data_prefix=data_prefix,
            ann_file=ann_file,
            metainfo=metainfo,
            **kwargs)

        if self.CLASSES is None:
            logger.warning(
                'The CLASSES is not stored in the `ImageNet21k` class. '
                'Considering to specify the `classes` argument if you need '
                'do inference on the ImageNet-21k dataset')
