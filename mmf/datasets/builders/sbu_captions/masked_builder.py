# # Copyright (c) Facebook, Inc. and its affiliates.

# from mmf.common.registry import registry
# from mmf.datasets.builders.coco import MaskedCOCOBuilder

# from .masked_dataset import MaskedSBUDataset


# @registry.register_builder("masked_sbu")
# class MaskedSBUBuilder(MaskedCOCOBuilder):
#     def __init__(self):
#         super().__init__()
#         self.dataset_name = "masked_sbu"
#         self.set_dataset_class(MaskedSBUDataset)

#     @classmethod
#     def config_path(cls):
#         return "configs/datasets/sbu_captions/masked.yaml"

# Copyright (c) Facebook, Inc. and its affiliates.
import logging


from mmf.common.registry import registry
from mmf.datasets.mmf_dataset_builder import MMFDatasetBuilder
from mmf.utils.general import get_mmf_root

from .masked_dataset import MaskedSBUDataset

import os

logger = logging.getLogger(__name__)


@registry.register_builder("masked_sbu")
class MaskedSBUBuilder(MMFDatasetBuilder):
    def __init__(self):
        self.dataset_name = "masked_sbu"
        super().__init__(dataset_name=self.dataset_name)
        self.set_dataset_class(MaskedSBUDataset)
        self.download_url = "https://www.cs.virginia.edu/~vicente/sbucaptions/SBUCaptionedPhotoDataset.tar.gz"

    def build(self, config, dataset):
        download_folder = os.path.join(
            config.data_dir, "sbu"
        )

        file_name = self.download_url.split("/")[-1]
        local_filename = os.path.join(download_folder, file_name)

        # Either if the zip file is already present or if there are some
        # files inside the folder we don't continue download process
        if os.path.exists(local_filename):
            return

        raise Exception("You need to download %s into the folder %s" % (self.download_url, download_folder))


    def load(self, config, dataset, *args, **kwargs):
        self.dataset = self.dataset_class(config, dataset, 0)
        return self.dataset

    @classmethod
    def config_path(cls):
        return "configs/datasets/sbu_captions/masked.yaml"

