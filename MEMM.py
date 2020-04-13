import numpy as np
from scipy import special
from collections import OrderedDict


class MEMM():

    def __init__(self):
        self.v = []
        self.features = MEMM_features()

    def fit(self, file_path, threshold=None):
        """
            Fits the model on the data, ultimately finds the vector of weights
            :param file_path: full path of the file to read
                return vector of weights
        """
        self.features.get_feature_statistics(file_path)
        self.features.get_feature_indices(threshold)
        self.initialize_v()

    def predict(self, file_path):
        assert (self.v)

    def initialize_v(self):
        self.v = np.random.rand(self.features.n_total_features) / 100

    def history_generator(self, file_path):
        with open(file_path) as f:
            for line in f:
                words = line.split(' ')
                del words[-1]
                for word_idx in range(len(words)):
                    cur_word, cur_tag = words[word_idx].split('_')

    # def linearTerm(self, file_path):


class MEMM_features():

    def __init__(self):
        self.n_total_features = 0
        self.f_indexes = {}
        self.f_statistics = {}

    def get_feature_statistics(self, file_path):
        f100_stats = self.get_f100_stats(file_path)
        f101_stats = self.get_f101_stats(file_path)
        f102_stats = self.get_f102_stats(file_path)
        f103_stats = self.get_f103_stats(file_path)
        f104_stats = self.get_f104_stats(file_path)
        f105_stats = self.get_f105_stats(file_path)
        self.f_statistics = {100: f100_stats, 101: f101_stats, 102: f102_stats, 103: f103_stats,
                             104: f104_stats, 105: f105_stats}

    def get_feature_indices(self, threshold=None):

        if (threshold is not None):
            for i, f_stats in self.f_statistics.items():
                f = {k: v for (k, v) in f_stats.items() if v > threshold}
                self.f_statistics[i] = f

        current_length = 0
        for i, f_index in self.f_statistics.items():
            f_index_dict = dict.fromkeys(f_index.keys(), 0)
            f_index_dict.update(zip(f_index_dict, range(current_length, len(f_index_dict) + current_length)))
            current_length += len(f_index_dict)
            self.f_indexes[i] = f_index_dict

        self.n_total_features = current_length + 1

    def get_f100_stats(self, file_path):
        """
            Extract out of text all word/tag pairs
            :param file_path: full path of the file to read
                return all word/tag pairs with index of appearance
        """
        words_tags_count_dict = {}

        with open(file_path) as f:
            for line in f:
                words = line.split(' ')
                del words[-1]
                for word_idx in range(len(words)):
                    cur_word, cur_tag = words[word_idx].split('_')
                    if (cur_word, cur_tag) not in words_tags_count_dict.keys():
                        words_tags_count_dict[(cur_word, cur_tag)] = 1
                    else:
                        words_tags_count_dict[(cur_word, cur_tag)] += 1
        return words_tags_count_dict

    def get_f101_stats(self, file_path):

        prefix_count_dict = {}

        with open(file_path) as f:
            for line in f:
                words = line.split(' ')
                del words[-1]
                for word_idx in range(len(words)):
                    cur_word, cur_tag = words[word_idx].split('_')
                    for prefix in prefixes_dict.keys():
                        if cur_word.lower().startswith(prefix) and cur_tag.startswith(prefixes_dict[prefix]):
                            if (prefix, prefixes_dict[prefix]) not in prefix_count_dict:
                                prefix_count_dict[(prefix, prefixes_dict[prefix])] = 1
                            else:
                                prefix_count_dict[(prefix, prefixes_dict[prefix])] += 1
        return prefix_count_dict

    def get_f102_stats(self, file_path):

        suffix_count_dict = {}

        with open(file_path) as f:
            for line in f:
                words = line.split(' ')
                del words[-1]
                for word_idx in range(len(words)):
                    cur_word, cur_tag = words[word_idx].split('_')
                    for suffix in suffixes_dict.keys():
                        if cur_word.lower().endswith(suffix) and cur_tag.startswith(suffixes_dict[suffix]):
                            if (suffix, suffixes_dict[suffix]) not in suffix_count_dict:
                                suffix_count_dict[(suffix, suffixes_dict[suffix])] = 1
                            else:
                                suffix_count_dict[(suffix, suffixes_dict[suffix])] += 1
        return suffix_count_dict

    def get_f103_stats(self, file_path):

        trigram_tags_count_dict = {}

        with open(file_path) as f:
            for line in f:
                words = line.split(' ')
                del words[-1]
                tags = [i.split('_')[1] for i in words]
                tags.insert(0, '*')
                tags.insert(0, '*')
                tags.append('STOP')
                for i, _ in enumerate(tags[:-2]):
                    if (tags[i], tags[i + 1], tags[i + 2]) not in trigram_tags_count_dict:
                        trigram_tags_count_dict[(tags[i], tags[i + 1], tags[i + 2])] = 1
                    else:
                        trigram_tags_count_dict[(tags[i], tags[i + 1], tags[i + 2])] += 1
            return trigram_tags_count_dict

    def get_f104_stats(self, file_path):

        bigram_tags_count_dict = {}

        with open(file_path) as f:
            for line in f:
                words = line.split(' ')
                del words[-1]
                tags = [i.split('_')[1] for i in words]
                tags.insert(0, '*')
                tags.insert(0, '*')
                tags.append('STOP')
                for i, _ in enumerate(tags[:-1]):
                    if (tags[i], tags[i + 1]) not in bigram_tags_count_dict:
                        bigram_tags_count_dict[(tags[i], tags[i + 1])] = 1
                    else:
                        bigram_tags_count_dict[(tags[i], tags[i + 1])] += 1

        return bigram_tags_count_dict

    def get_f105_stats(self, file_path):

        unigram_tags_count_dict = {}

        with open(file_path) as f:
            for line in f:
                words = line.split(' ')
                del words[-1]
                tags = [i.split('_')[1] for i in words]
                tags.insert(0, '*')
                tags.insert(0, '*')
                tags.append('STOP')
                for i, _ in enumerate(tags):
                    if (tags[i]) not in unigram_tags_count_dict:
                        unigram_tags_count_dict[(tags[i])] = 1
                    else:
                        unigram_tags_count_dict[(tags[i])] += 1

        return unigram_tags_count_dict

    def represent_input_with_features(self, history):
        """
            Extract feature vector in per a given history
            :param history: touple{word, pptag, ptag, ctag, nword, pword}
            :param word_tags_dict: word\tag dict
                Return a list with all features that are relevant to the given history
        """
        word, pptag, ptag, ctag, nword, pword, features = history

        # append features belonging to f100
        if (word, ctag) in self.f_indexes[100]:
            features.append(self.f_indexes[100][(word, ctag)])

        # append features belonging to f101
        for prefix in reversed(range(2, 5)):
            prefix = word[:prefix].lower()
            if (prefix, ctag) in self.f_indexes[101]:
                features.append(self.f_indexes[101][(prefix, ctag)])
                break

        # append features belonging to f102
        for suffix in reversed(range(2, 5)):
            suffix = word[-suffix:].lower()
            if (suffix, ctag) in self.f_indexes[102]:
                features.append(self.f_indexes[102][(suffix, ctag)])
                break

        # append features belonging to f103
        if (pptag,ptag,ctag) in self.f_indexes[103]:
            features.append(self.f_indexes[103][(pptag,ptag,ctag)])

        # append features belonging to f104
        if (ptag,ctag) in self.f_indexes[104]:
            features.append(self.f_indexes[104][(ptag,ctag)])

        # append features belonging to f105
        if (ctag) in self.f_indexes[105]:
            features.append(self.f_indexes[105][(ctag)])


        return features


prefixes = (('re', 'VB'), ('dis', 'VB'), ('over', 'VB'), ('un', 'VB'), ('mis', 'VB'), ('out', 'VB'),
            ('co', 'NN'), ('sub', 'NN'), ('un', 'JJ'), ('im', 'JJ'), ('in', 'JJ'), ('ir', 'JJ'),
            ('il', 'JJ'), ('non', 'JJ'), ('dis', 'JJ'))

prefixes_dict = dict(prefixes)

suffixes = (('ise', 'VB'), ('ate', 'VB'), ('fy', 'VB'), ('en', 'VB'), ('tion', 'NN'), ('ity', 'NN'),
            ('er', 'NN'), ('ness', 'NN'), ('ism', 'NN'), ('ment', 'NN'), ('ant', 'NN'), ('ship', 'NN'),
            ('age', 'NN'), ('ery', 'NN'))

suffixes_dict = dict(suffixes)


# %%
def calc_objective_per_iter(w_i, arg_1, arg_2):
    """
        Calculate max entropy likelihood for an iterative optimization method
        :param w_i: weights vector in iteration i
        :param arg_i: arguments passed to this function, such as lambda hyperparameter for regularization

            The function returns the Max Entropy likelihood (objective) and the objective gradient
    """

    ## Calculate the terms required for the likelihood and gradient calculations
    linear_term = 0
    normalization_term = 0
    regularization = 0
    empirical_counts = 0
    expected_counts = 0
    regularization_grad = 0
    likelihood = linear_term - normalization_term - regularization
    grad = empirical_counts - expected_counts - regularization_grad

    return (-1) * likelihood, (-1) * grad
