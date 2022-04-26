# # Copyright (c) Facebook, Inc. and its affiliates.

# from mmf.datasets.builders.coco import MaskedCOCODataset


# class MaskedSBUDataset(MaskedCOCODataset):
#     def __init__(self, config, dataset_type, imdb_file_index, *args, **kwargs):
#         super().__init__(config, dataset_type, imdb_file_index, *args, **kwargs)
#         self.dataset_name = "masked_sbu"
#         self._two_sentence = config.get("two_sentence", True)
#         self._false_caption = config.get("false_caption", True)
#         self._two_sentence_probability = config.get("two_sentence_probability", 0.5)
#         self._false_caption_probability = config.get("false_caption_probability", 0.5)


# Copyright (c) Facebook, Inc. and its affiliates.

from mmf.datasets.base_dataset import BaseDataset
from mmf.common.sample import Sample

from PIL import Image
import requests
from io import BytesIO
import pandas as pd
import os

class MaskedSBUDataset(BaseDataset):
    def __init__(self, config, dataset, imdb_file_index, *args, **kwargs):
        if "name" in kwargs:
            name = kwargs["name"]
        elif "dataset_name" in kwargs:
            name = kwargs["dataset_name"]
        else:
            name = "masked_sbu"
        super().__init__(name, config, dataset, index=imdb_file_index)

        self.photos = []
        self.captions = []
        file1 = os.path.join(self.config.data_dir, config.features)
        file2 = os.path.join(self.config.data_dir, config.annotations)

        for line1, line2 in zip(open(file1), open(file2)):
            url = line1.rstrip()
            caption = line2.rstrip()
            self.photos.append(url)
            self.captions.append(caption)      

    def init_processors(self):
        super().init_processors()

    def __getitem__(self, idx):
        current_sample = Sample()
        url, caption = self.photos[idx], self.captions[idx]
        caption_data = {"text": caption}
        try:
            response = requests.get(url)
            if response.status_code == 200:
                img = Image.open(BytesIO(response.content))
                current_sample.image = self.image_processor(img)
            else:
                return None
        except:
            return None
        processed_question = self.text_processor(caption_data)
        current_sample.text = processed_question["text"]
        if "input_ids" in processed_question:
            current_sample.update(processed_question)

        return current_sample

    
    def __len__(self):
        return len(self.photos)

