import torch
import numpy as np
from transformers import BertTokenizer


class Dataset(torch.utils.data.Dataset):

    def __init__(self, df, labels_dict, bert_model):

        self.tokenizer = BertTokenizer.from_pretrained(bert_model)
        self.labels = [labels_dict[label] for label in df['category']]
        # Call BertTokenizer function to transform input texts into format that BERT expects.
        self.texts = [self.tokenizer(text,
                                     padding='max_length', max_length=512, truncation=True,
                                     return_tensors="pt") for text in df['text']]

    def classes(self):
        return self.labels

    def __len__(self):
        return len(self.labels)

    def get_batch_labels(self, idx):
        # Fetch a batch of labels
        return np.array(self.labels[idx])

    def get_batch_texts(self, idx):
        # Fetch a batch of inputs
        return self.texts[idx]

    def __getitem__(self, idx):

        batch_texts = self.get_batch_texts(idx)
        batch_y = self.get_batch_labels(idx)

        return batch_texts, batch_y
