import json
from argparse import Namespace
from functools import partial
import jsonlines

import torch
import torch.nn.functional as F
import tqdm
from torch.utils.data import DataLoader

from gutenberg_poetic.models.poetry_classifier import PoetryClassifier
from gutenberg_poetic.models.poetry_classifier import preprocess
from gutenberg_poetic.datasets.jsonl_dataset import JsonlDataset
from gutenberg_poetic.helpers.utils import batch_iterable


def load_model(model_path, vocab_path='pretrained'):
    model_args = {
        'vocab_path': vocab_path
    }

    model = PoetryClassifier(Namespace(**model_args))
    model.load_state_dict(torch.load(model_path))
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda(device=0)

    return model


def predict_batch(model, batch):
    with torch.no_grad():
        if torch.cuda.is_available():
            batch = {k: v.cuda() for k, v in batch.items()}
        output = model.forward(batch)
        return F.softmax(output['logits'], dim=1)[:, 1].tolist()


def preprocess(tokenizer, input_batch, max_length=256):
    text = [" [SEP] ".join(ex['window']) for ex in input_batch]
    return tokenizer(
        text = text,
        max_length = max_length,
        return_tensors = 'pt',
        padding='max_length',
        truncation=True,
        is_split_into_words=False
    )


def predict_data(data_path, model, batch_size=512):
    with jsonlines.open(data_path) as reader:
        batches = list(batch_iterable(reader, 24))
    predict = partial(predict_batch, model)
    progress = tqdm.tqdm(total=len(batches))
    preds = []
    for data_batch in batches:
        preds += predict(preprocess(model.tokenizer, data_batch))
        progress.update(batch_size)
    return preds
