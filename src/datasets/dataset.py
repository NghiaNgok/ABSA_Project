import pandas as pd
import re
import torch
from torch.utils.data import Dataset

def extract_annotations(sentence):
    pattern = r'\[(\d+),(\d+),(\w+)\]'
    matches = re.findall(pattern, sentence)
    return [(int(start), int(end), label) for start, end, label in matches]

class CustomABSA(Dataset):
    def __init__(self, dataframe):
        self.data = dataframe
        self.sentences = []
        self.annotations = []

        for index, row in dataframe.iterrows():
            sentence = row['sentence']
            annots = extract_annotations(sentence)
            clean_sentence = re.sub(r'\[\d+,\d+,\w+\]', '', sentence).strip()
            self.sentences.append(clean_sentence)
            self.annotations.append(annots)

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        sentence = self.sentences[idx]
        annotations = self.annotations[idx]

        label_vector = [0] * len(sentence)

        for start, end, label in annotations:
            start = max(0, start)
            end = min(len(sentence) - 1, end)

            if label == 'p':
                sentiment_label = 2  # positive
            elif label == 'neg':
                sentiment_label = 1  # negative
            else:
                sentiment_label = 0  # neutral

            for i in range(start, end + 1):
                label_vector[i] = sentiment_label

        return sentence, torch.tensor(label_vector)

def collate_fn(batch):
    sentences, labels = zip(*batch)

    max_len = max(len(s) for s in sentences)

    padded_sentences = []
    padded_labels = []

    for s, l in zip(sentences, labels):
        padded_sentences.append(s.ljust(max_len))
        padded_labels.append(torch.cat([l, torch.zeros(max_len - len(l))]))

    return padded_sentences, torch.stack(padded_labels)
