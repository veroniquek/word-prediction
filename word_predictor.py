import argparse
import numpy as np

from pip._vendor.distlib.compat import raw_input

from dataset.bigram_dataset import BigramDataset
from dataset.dataset_template import DatasetLoader
from dataset.trigram_dataset import TrigramDataset
from models.bigram_model import BigramTrainer
from models.trigram_model import TrigramTrainer

PREDICTOR_OPTIONS = ['bigram', 'trigram']


class WordCompleter(object):
    def __init__(self, dataset: DatasetLoader, model='bigram', number_of_predictions=3, store_folder=None,
                 load_folder=None, show_probabilities=False):
        self.model = model

        if model == 'bigram':
            self.predictor = BigramTrainer(load_file=load_folder,
                                           store_folder=store_folder,
                                           dataset=dataset,
                                           )

        elif model == 'trigram':
            self.predictor = TrigramTrainer(load_file=load_folder,
                                            store_folder=store_folder,
                                            dataset=dataset,
                                            )

        else:
            try:
                raise Exception()

            except Exception as e:
                print(f"Could not find predictor for model: {model}, options are: {PREDICTOR_OPTIONS}")
                quit()

        # store trained model
        if store_folder:
            self.predictor.dataset.store_dataset(store_folder)
            self.predictor.store_model(store_folder)

        self.number_of_predictions = number_of_predictions
        self.show_probabilities = show_probabilities

    def get_prev_words(self, sentence):
        if len(sentence) < 2:
            return ""
        elif self.model == 'ngram':
            return sentence[-2]
        else:
            return sentence[:-1]

    def get_letters_so_far(self, sentence):
        if len(sentence) < 1:
            return ""

        else:
            return sentence[-1]

    def get_matches(self, options_and_probs, letters_so_far):
        """
        :param options_and_probs: options of words to choose from next and their corresponding probabilities
        :param letters_so_far: beginning of the next word for word completion, will be '' if none
        :return: 3 predictions for the next word
        """
        if letters_so_far == '':
            matches = options_and_probs
        else:
            matches = [(s, p) for (s, p) in options_and_probs if s and s.startswith(letters_so_far)]

        if matches == []:
            return [""]

        possible_words = np.array([w for w, _ in matches])
        probabilities = np.array([p for _, p in matches])
        probabilities = probabilities / np.sum(probabilities)

        # sort the words from highest to lowest probabilities
        inds = (-probabilities).argsort()
        sorted_probabilities = probabilities[inds]
        sorted_words = possible_words[inds]

        indices = np.random.choice(a=range(len(sorted_words)), p=sorted_probabilities,
                                   size=min(len(sorted_words), self.number_of_predictions), replace=False)

        indices.sort()

        if self.show_probabilities:
            return list(zip(sorted_words[indices], sorted_probabilities[indices]))

        else:
            return sorted_words[indices]

    def get_predictions(self, text):
        words = text.split(' ')

        # divide the sentence into full words and prefix of word to predict
        letters_so_far = self.get_letters_so_far(words)
        prev_word = self.get_prev_words(words)

        # predict probabilities for all the possible next words
        options_and_probs = self.predictor.predict_next_word_by_probabilities(prev_word)

        return self.get_matches(options_and_probs, letters_so_far)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Demo')

    parser.add_argument('--dataset', '-d', type=str, default='nps', required=False, help='dataset to run')
    parser.add_argument('--model', '-m', type=str, default='bigram', required=False, help='model to run')
    parser.add_argument('--probabilities', '-p', type=str, default=False, required=False,
                        help='Display probabilities of predictions')

    parser.add_argument('--load', '-ld', type=str, required=False, help='path to load model from')
    parser.add_argument('--store', '-st', type=str, default=False, help='store the model to a folder')

    arguments = parser.parse_args()

    print(f"Running {arguments.model} on {arguments.dataset}")

    store_folder = None
    load_folder = None

    if arguments.load:
        load_folder = arguments.load

    elif arguments.store:
        load_folder = None
        store_folder = arguments.store

    if arguments.model == 'bigram':
        dataset = BigramDataset(arguments.dataset)

    elif arguments.model == 'trigram':
        dataset = TrigramDataset(arguments.dataset)

    completer = WordCompleter(dataset=dataset,
                              model=arguments.model,
                              load_folder=load_folder,
                              store_folder=store_folder,
                              show_probabilities=arguments.probabilities
                              )

    print("Type your words. Type 'q' to quit")
    inp = raw_input("Input: ")
    while inp != 'q':
        predictions = completer.get_predictions(inp.lower())
        print("Predictions: ", predictions)
        inp = raw_input("Input: ")
