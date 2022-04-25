# Copyright (c) Facebook, Inc. and its affiliates.

from mmf.datasets.base_dataset import BaseDataset
from mmf.common.sample import Sample

from PIL import Image
import requests
from io import BytesIO
import pandas as pd
import os

class MaskedConceptualCaptions12Dataset(BaseDataset):
    def __init__(self, config, dataset, imdb_file_index, *args, **kwargs):
        if "name" in kwargs:
            name = kwargs["name"]
        elif "dataset_name" in kwargs:
            name = kwargs["dataset_name"]
        else:
            name = "masked_conceptual_captions_12"
        super().__init__(name, config, dataset, index=imdb_file_index)
        self.data = pd.read_table(os.path.join(config.data_dir, config.features))        

    def init_processors(self):
        super().init_processors()

    def __getitem__(self, idx):
        current_sample = Sample()
        row = self.data.iloc[idx]
        url, caption = row[0], row[1]
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
        return len(self.data)

