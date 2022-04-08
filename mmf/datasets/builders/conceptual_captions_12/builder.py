# Copyright (c) Facebook, Inc. and its affiliates.

from mmf.common.registry import registry
from mmf.datasets.builders.coco import COCOBuilder

from .dataset import ConceptualCaptions12Dataset


@registry.register_builder("conceptual_captions_12")
class ConceptualCaptions12Builder(COCOBuilder):
    def __init__(self):
        super().__init__()
        self.dataset_name = "conceptual_captions_12"
        self.set_dataset_class(ConceptualCaptions12Dataset)

    @classmethod
    def config_path(cls):
        return "configs/datasets/conceptual_captions_12/defaults.yaml"
