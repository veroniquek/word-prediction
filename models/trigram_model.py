import math
import codecs

from collections import defaultdict


class TrigramTrainer:
    def __init__(self, dataset=None, load_file=None, store_folder=None, laplace_smoothing=True):
        self.dataset = dataset

        # bigram and unigram probabilities
        self.bigram_prob = defaultdict(lambda: defaultdict(float))
        self.trigram_prob = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
        self.unigram_prob = defaultdict(float)

        self.laplace_smoothing = laplace_smoothing

        if load_file:
            self.load_data(load_file)
        else:
            self.train()

        if store_folder:
            self.store_model(store_folder)
            self.dataset.store_dataset(store_folder)

        self.train()

    def train(self):
        """
        Computes unigram counts and bigram probabilities
        """
        V, N = self.dataset.unique_words, self.dataset.total_words

        for first_token in self.dataset.bigram_count:
            # compute unigram stuff
            count = self.dataset.unigram_count[first_token]

            self.unigram_prob[first_token] = count / self.dataset.total_words

            for second_token in self.dataset.bigram_count[first_token]:
                # compute bigram stuff
                count = self.dataset.bigram_count[first_token][second_token]
                total_count = self.dataset.unigram_count[first_token]

                if self.laplace_smoothing:
                    log_prob = math.log((count + 1) / (total_count + V))
                else:
                    log_prob = math.log(count / total_count)

                self.bigram_prob[first_token][second_token] = log_prob

                for third_token in self.dataset.trigram_count[first_token][second_token]:
                    # compute trigram stuff
                    count = self.dataset.trigram_count[first_token][second_token][third_token]
                    total_count = self.dataset.bigram_count[first_token][second_token]

                    if self.laplace_smoothing:
                        log_prob = math.log((count + 1) / (total_count + V))
                    else:
                        log_prob = math.log(count / total_count)

                    # TODO: Something here is computed wrongly..
                    self.trigram_prob[first_token][second_token][third_token] = log_prob

    def predict_next_word_by_probabilities(self, input_sentence, min_length=3):
        """
        Returns a list of all words in the corpus and the corresponding probability of them
        being the next word in the sentence given the previous word (token)
        """

        if len(input_sentence) < 1:
            prev_word = ''
            prevprev_word = ''

        elif len(input_sentence) < 2:
            prev_word = input_sentence[-1]
            prevprev_word = ''
        else:
            prev_word = input_sentence[-1]
            prevprev_word = input_sentence[-2]

        # no words given, return random words
        if prev_word == '' or prev_word not in self.dataset.w2i:
            return list(zip(list(self.dataset.w2i.keys()), list(self.unigram_prob.values())))

        # only two words are given: work with bigram probabilities
        prev_word_index = self.dataset.w2i.get(prev_word, -1)
        prevprev_word_index = self.dataset.w2i.get(prevprev_word, -1)

        # return all possible words, sorted by their probabilities
        bigram_probs = {second_word: self.bigram_prob[prev_word_index][second_word]
                        for second_word in self.bigram_prob[prev_word_index]}
        probabilities = list(map(lambda x: math.exp(x), list(bigram_probs.values())))
        next_indices = list(self.bigram_prob[prev_word_index].keys())

        bigram_results = [(self.dataset.i2w[i], p) for i, p in zip(next_indices, probabilities)]

        if prevprev_word == '' or prevprev_word not in self.dataset.w2i \
                or prev_word_index not in self.trigram_prob[prevprev_word_index]:
            return bigram_results

        trigram_probs = {third_word: self.trigram_prob[prevprev_word_index][prev_word_index][third_word]
                         for third_word in self.trigram_prob[prevprev_word_index][prev_word_index]}
        probabilities = list(map(lambda x: math.exp(x), list(trigram_probs.values())))
        next_indices = list(self.trigram_prob[prevprev_word_index][prev_word_index].keys())

        trigram_results = [(self.dataset.i2w[i], p) for i, p in zip(next_indices, probabilities)]

        return trigram_results

    def stats(self):
        """
           Creates a list of rows to print of the language model.
        """
        rows_to_print = []
        V, N = self.dataset.unique_words, self.dataset.total_words
        rows_to_print.append(f"{V} {N}")
        for token in self.dataset.w2i:
            index = self.dataset.w2i[token]
            prob = self.unigram_prob[index]
            rows_to_print.append(f"{index} {token} " + "{0:.15f}".format(prob))

        for first_token in self.bigram_prob:
            for second_token in self.bigram_prob[first_token]:
                rows_to_print.append(
                    f"{first_token} {second_token} " + "{0:.15f}".format(self.bigram_prob[first_token][second_token]))

        # YOUR CODE HERE
        rows_to_print.append("-1")
        return rows_to_print

    def store_model(self, folder_path):
        stats = self.stats()
        with codecs.open(folder_path + "/probabilities.txt", 'w', 'utf-8') as f:
            for row in stats:
                f.write(row + '\n')

    def load_data(self, folder):
        try:
            with codecs.open(folder + "/probabilities.txt", 'r', 'utf-8') as f:
                unique_words, total_words = map(int, f.readline().strip().split(' '))

                for _ in range(unique_words):
                    index, token, prob = f.readline().strip().split(' ')
                    self.unigram_prob[index] = float(prob)

                line = f.readline().strip()
                while line != '-1':
                    index1, index2, prob = line.split(' ')
                    self.bigram_prob[int(index1)][int(index2)] = float(prob)
                    line = f.readline().strip()
                return True

        except IOError:
            print("Couldn't find bigram probabilities file {}/probabilities.txt".format(folder))
            return False
