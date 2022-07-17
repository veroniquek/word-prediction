import string

from collections import defaultdict, Counter

from nltk import ngrams

from dataset.dataset_template import DatasetLoader


class BigramDataset(DatasetLoader):
    def __init__(self, dataset='npc'):

        # obtain the wordstream
        super(BigramDataset, self).__init__(dataset)

        self.last_index = -1
        self.total_words = 0
        self.unique_words = 0

        self.unigram_count = defaultdict(int)
        self.bigram_count = defaultdict(lambda: defaultdict(int))

        # fill the datastructure
        for token in self.word_stream:
            self.process_token(token)

    def process_token(self, token):
        self.total_words += 1

        # token we already have (just need to increase count)
        if token in self.w2i:
            index = self.w2i[token]
            self.unigram_count[index] += 1

        # new token
        else:
            new_index = self.unique_words
            self.unigram_count[new_index] = 1
            self.w2i[token] = new_index
            self.i2w[new_index] = token
            self.unique_words += 1

        if self.last_index > -1:
            self.bigram_count[self.last_index][self.w2i[token]] += 1

        # update index of previous word
        self.last_index = self.w2i[token]

    def __len__(self):
        return len(self.word_stream - 1)

    def __getitem__(self, idx):
        # return a pair of words where the first is the input and the second is the output
        return (self.word_stream[idx],
                self.word_stream[idx + 1])
