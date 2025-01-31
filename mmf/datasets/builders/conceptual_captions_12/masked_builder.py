# Copyright (c) Facebook, Inc. and its affiliates.
import logging


from mmf.common.registry import registry
from mmf.datasets.mmf_dataset_builder import MMFDatasetBuilder
from mmf.utils.general import get_mmf_root

from .masked_dataset import MaskedConceptualCaptions12Dataset

import os
import wget

logger = logging.getLogger(__name__)


@registry.register_builder("masked_conceptual_captions_12")
class MaskedConceptualCaptions12Builder(MMFDatasetBuilder):
    def __init__(self):
        self.dataset_name = "masked_conceptual_captions_12"
        super().__init__(dataset_name=self.dataset_name)
        self.set_dataset_class(MaskedConceptualCaptions12Dataset)
        self.download_url = "https://storage.googleapis.com/conceptual_12m/cc12m.tsv"

    def build(self, config, dataset):
        download_folder = os.path.join(
            config.data_dir, "cc12"
        )

        
        file_name = self.download_url.split("/")[-1]
        local_filename = os.path.join(download_folder, file_name)

        if not os.path.exists(local_filename):
            os.makedirs(download_folder, exist_ok=True)
            wget.download(self.download_url, out=download_folder)
        


    def load(self, config, dataset, *args, **kwargs):
        self.dataset = self.dataset_class(config, dataset, 0)
        return self.dataset

    @classmethod
    def config_path(cls):
        return "configs/datasets/conceptual_captions_12/masked.yaml"
