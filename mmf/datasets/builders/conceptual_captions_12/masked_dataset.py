# Copyright (c) Facebook, Inc. and its affiliates.

from mmf.datasets.builders.coco import MaskedCOCODataset


class MaskedConceptualCaptions12Dataset(MaskedCOCODataset):
    def __init__(self, config, dataset_type, imdb_file_index, *args, **kwargs):
        import pdb; pdb.set_trace()

        # HERE we need to make sure this datset works properly 
        
        super().__init__(config, dataset_type, imdb_file_index, *args, **kwargs)
        self.dataset_name = "masked_conceptual_captions_12"
        self._two_sentence = config.get("two_sentence", True)
        self._false_caption = config.get("false_caption", True)
        self._two_sentence_probability = config.get("two_sentence_probability", 0.5)
        self._false_caption_probability = config.get("false_caption_probability", 0.5)
