import json
import os
import zipfile
import re

from typing import Text

from torch.utils.data import Dataset


class FilteredGutenbergDammitDataset(Dataset):

    def __init__(self,
                 archive_path: Text,
                 filter_field: Text = "Subject",
                 pattern: Text = r'^Poet.*',
                 preprocess=None):
        super().__init__()

        self.preprocess = preprocess
        zip_file = zipfile.ZipFile(archive_path)
        with zip_file.open('gutenberg-dammit-files/gutenberg-metadata.json') as f:
            gutenberg_metadata = json.load(f)
        filtered_paths = self.filtered_iter(gutenberg_metadata, filter_field, pattern)
        lines = self.zips_to_lines(zip_file, filtered_paths)
        self.data = [{'text': line} for line in lines if line]
        self.total = len(self.data)

    def __getitem__(self, idx: int):
        if self.preprocess:
            return self.preprocess(self.data[idx])
        else:
            return self.data[idx]

    def __len__(self):
        return self.total

    def zips_to_lines(self, zip_file, path_name_iter):
        for path_name in path_name_iter:
            with zip_file.open(os.path.join('gutenberg-dammit-files/', path_name)) as f:
                for line in f.read().decode('utf-8').split('\n'):
                    yield line.strip()

    def filtered_iter(self, data, field, pattern, english_only=True):
        for item in data:
            matches_pattern = any([re.match(pattern, subj) for subj in item.get(field, [])])
            is_english = "English" in item.get("Language")
            if matches_pattern and (is_english or not english_only):
                yield item.get('gd-path')
