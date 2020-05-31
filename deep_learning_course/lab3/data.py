import torch
import numpy as np
from dataclasses import dataclass, field
from typing import List
import csv
from collections.abc import Iterable


@dataclass
class Instance:
    '''Data wrapper'''
    label: str
    text: List[str] = field(default_factory=list)


class NLPDataset(torch.utils.data.Dataset):

    def __init__(self):
        self.text_vocab = None
        self.label_vocab = None
        self.instances = []

    @staticmethod
    def from_file(file_path, create_vocab=True, max_size=-1, min_freq=0):
        dataset = NLPDataset()
        with open(file_path) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=",", skipinitialspace=True)
            for row in csv_reader:
                dataset.instances += [Instance(row[1], row[0].split())]

        text_freq, label_freq = dataset.calc_freq()
        if create_vocab:
            dataset.text_vocab = Vocab(text_freq, max_size=max_size, min_freq=min_freq)
            dataset.label_vocab = Vocab(label_freq, max_size=max_size, min_freq=min_freq, special_chars=False)
        return dataset

    def calc_freq(self):
        wordlist = []
        for instance in self.instances:
            wordlist += instance.text

        wordcnt_text = {}
        for word in wordlist:
            if word not in wordcnt_text:
                wordcnt_text[word] = wordlist.count(word)

        wordlist.clear()
        positive, negative = 0, 0
        for instance in self.instances:
            if instance.label == "positive":
                positive += 1
            else:
                negative += 1

        return wordcnt_text, {'positive': positive, 'negative': negative}

    def __getitem__(self, idx):
        instance = self.instances[idx]
        text_enc = self.text_vocab.encode(instance.text)
        label_enc = self.label_vocab.encode(instance.label)
        return text_enc, label_enc

    def __len__(self):
        return len(self.instances)


class Vocab:

    def __init__(self, frequencies, max_size, min_freq, special_chars=True):
        self.max_size = max_size
        self.min_freq = min_freq
        self.stoi = {'<PAD>': 0, '<UNK>': 1} if special_chars else {}
        self.itos = {0: '<PAD>', 1: '<UNK>'} if special_chars else {}
        self.make_vocabs(frequencies)

    def make_vocabs(self, frequencies):
        frequencies = {k: v for k, v in sorted(frequencies.items(), key=lambda x: x[1]) if v >= self.min_freq}
        length = self.max_size if self.max_size != -1 else len(frequencies)
        offset = len(self.stoi)
        for idx in range(length):
            word, cnt = frequencies.popitem()
            self.stoi[word] = idx + offset
            self.itos[idx + offset] = word

    def encode(self, tokens):
        assert isinstance(tokens, (str, Iterable))
        if isinstance(tokens, str):
            if tokens not in self.stoi:
                tokens = '<UNK>'
            return torch.tensor(self.stoi[tokens])
        elif isinstance(tokens, Iterable):
            idxs = []
            for token in tokens:
                if token not in self.stoi:
                    token = '<UNK>'
                idxs += [self.stoi[token]]
            return torch.tensor(idxs)

    def __len__(self):
        return len(self.stoi)


def generate_embedding_matrix(vocab, dim=300, file_path=None):
    if file_path:
        words_dict = {}
        with open(file_path, 'r') as f:
            for line in f:
                split = line.split()
                words_dict[split[0]] = split[1:]
            dim = len(list(words_dict.values())[0])

    mat = torch.randn((len(vocab), dim))
    mat[0].zero_()
    if file_path:
        for word, idx in list(vocab.stoi.items())[2:]:
            if word in words_dict:
                mat[idx] = torch.tensor(np.array(words_dict[word]).astype(np.float))
    pretrained = file_path is None
    return torch.nn.Embedding.from_pretrained(mat, freeze=pretrained, padding_idx=0)


def pad_collate_fn(batch, padding_idx=0):
    """
    Arguments:
        batch:
            list of instances returned by `Dataset.__getitem__`
    Returns:
        A tensor representing the input batch
    """

    texts, labels = zip(*batch)  # Assuming the instance is in tuple-like form
    lengths = torch.tensor([len(text) for text in texts])  # Needed for later
    texts = torch.nn.utils.rnn.pad_sequence(texts, batch_first=True, padding_value=padding_idx)
    labels = torch.tensor(labels)
    return texts, labels, lengths


# train_dataset = NLPDataset.from_file('./data/sst_train_raw.csv')
# 
# batch_size = 2
# shuffle = False
# train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
#                                                shuffle=shuffle, collate_fn=pad_collate_fn)
# texts, labels, lengths = next(iter(train_dataloader))
# embedded_matrix = generate_embedding_matrix(train_dataset.text_vocab,
#                                             file_path='./data/sst_glove_6b_300d.txt')
# print(embedded_matrix(texts).size())
# print(embedded_matrix(torch.LongTensor([2])))
