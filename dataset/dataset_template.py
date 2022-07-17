import string

import nltk
import string
from torch.utils.data import Dataset

DATASET_OPTIONS = ['nps', 'gutenberg']

class DatasetLoader(Dataset):
    def __init__(self, dataset='nps', list_of_words=None):
        super(DatasetLoader, self).__init__()
        # mappings between words in the dataset and indices
        self.w2i = {}
        self.i2w = {}

        # TODO: split in train and test data
        # stream of all data (sequence of words)

        word_stream = []

        if list_of_words:
            word_stream = list_of_words

        elif dataset == 'nps':
            nltk.download('nps_chat')
            word_stream = nltk.corpus.nps_chat.words()

        elif dataset == 'gutenberg':
            nltk.download('gutenberg')
            word_stream = nltk.corpus.gutenberg.words()

        else:
            try:
                raise Exception()

            except Exception:
                print(f"Could not find dataset with name: {dataset}, options are: {DATASET_OPTIONS} "
                      f"or providing a fixed list of words")
                quit()



        cleaned = filter(lambda a: self.contains_no_punctuation(a), word_stream)
        self.word_stream = list(map(lambda a: a.lower(), cleaned))

    def contains_no_punctuation(self, token):
        if any(map(lambda a: a in string.punctuation, token)):
            return False

        if " " in token:
            return False

        if token == '':
            return False

        return True

    def __len__(self):
        raise NotImplementedError("__len__ has to be implemented by the subclass")

    def __getitem__(self, idx):
        raise NotImplementedError("__getitem__ has to be implemented by the subclass")
