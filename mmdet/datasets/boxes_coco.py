import numpy as np
from .api_wrappers import COCO, COCOeval

from .coco import CocoDataset
from .builder import DATASETS


@DATASETS.register_module
class BoxesCocoStyleDataset(CocoDataset):

    CLASSES = ('box',)

