import numpy as np
from scipy.optimize import fmin_l_bfgs_b
import time


class MEMM():

    def __init__(self):
        self.v = None
        self.tags = []
        self.features = MEMM_features()

    def fit(self, file_path, threshold=None):
        """
            Fits the model on the data, ultimately finds the vector of weights
            :param file_path: full path of the file to read
                return vector of weights
        """
        with open(file_path) as file:
            data = list(file)
        self.features.get_feature_statistics(data)
        self.features.get_feature_indices(threshold)
        v0 = np.random.rand(self.features.n_total_features) / 100
        self.tags = list(self.features.f_indexes[105].keys())
        history_dict, extended_history_dict = self.history_generator(data)
        linear_coefficient = self.linear_coefficient_calc(data, history_dict)
        args = (data, linear_coefficient, 1, history_dict, extended_history_dict)
        optimal_params = fmin_l_bfgs_b(func=self.likelihood_grad, x0=v0, args=args, maxiter=100, iprint=100)
        self.v = optimal_params[0]

    def predict(self, file_path):
        assert (self.v)

    def history_generator(self, file):
        history_dict = {}
        extended_history_dict = {}
        for line in file:
            words = line.split(' ')
            del words[-1]
            words.insert(0, '_*')
            words.insert(0, '_*')
            words.append('_STOP')
            for i, word_idx in enumerate(words[2:-1]):
                word, ctag = word_idx.split('_')
                pword, ptag = words[i + 1].split('_')
                ppword, pptag = words[i].split('_')
                nword, ntag = words[i + 3].split('_')
                history = (word, pptag, ptag, ctag, nword, pword)
                f = self.features.represent_input_with_features(history)
                if history not in history_dict.keys():
                    history_dict[history] = f
                    extended_history = []
                    for y in self.tags:
                        history = (word, pptag, ptag, y, nword, pword)
                        f = self.features.represent_input_with_features(history)
                        extended_history.append(f)
                    extended_history_dict[history] = extended_history
        return history_dict, extended_history_dict

    def linear_coefficient_calc(self, file, history_dict):
        sum_f = np.zeros(self.features.n_total_features)
        for _, feature in history_dict.items():
            for i in feature:
                sum_f[i] += 1
        return sum_f

    def normalization_term_and_expected_counts(self, v, extended_history_dict):
        normalization_term = 0
        expectedCounts = np.zeros(len(v))
        for history, features in extended_history_dict.items():
            inner_log = 0
            expectedInnerSum = np.zeros(len(v))
            indexes = set()
            for f in features:
                exp = 0
                for i in f:
                    exp += v[i]
                exponent = np.exp(exp)
                inner_log += exponent
                for i in f:
                    expectedInnerSum[i] += exponent
                    indexes.add(i)
            for i in indexes:
                expectedCounts[i] += expectedInnerSum[i] / inner_log
            normalization_term += np.log(inner_log)
        return (normalization_term, expectedCounts)

    def likelihood_grad(self, v, file, linear_coef, lamda, history_dict, extended_history_dict):
        """
            Calculate max entropy likelihood for an iterative optimization method
            :param v_i: weights vector in iteration i
            :param arg_i: arguments passed to this function, such as lambda hyperparameter for regularization

                The function returns the Max Entropy likelihood (objective) and the objective gradient
        """

        ## Calculate the terms required for the likelihood and gradient calculations
        linear_term = v.dot(linear_coef)
        start = time.time()
        normalization_term, expected_counts = self.normalization_term_and_expected_counts(v, extended_history_dict)
        end = time.time()
        print('time=', (end - start) / 60)
        regularization = 0.5 * lamda * (v.dot(v))
        empirical_counts = linear_coef
        regularization_grad = lamda * v
        likelihood = linear_term - normalization_term - regularization
        grad = empirical_counts - expected_counts - regularization_grad

        return (-1) * likelihood, (-1) * grad

    def loglinear_model(self, words):
        loglinear_model_dict = {}
        words.insert(0, '*')
        words.insert(0, '*')
        words.append('STOP')
        for i, word in enumerate(words[2:-1]):
            for u in self.tags:
                for v in self.tags:
                    current = {}
                    for c in self.tags:
                        history = (word, u, v, c, words[i + 3], words[i + 1])
                        f = self.features.represent_input_with_features(history)
                        exp = 0
                        for i in f:
                            exp += v[i]
                        current[c] = np.exp(exp)
                    denominator_sum = sum(list(current.values()))
                    for c in self.tags:
                        loglinear_model_dict[(word, u, v, c, words[i + 3], words[i + 1])] = current[c] / denominator_sum
        return loglinear_model_dict

    def find_max(self,pi , u, v,s_k_2 ,log_linear_model, word):
        max = 0
        argmax = s_k_2[0]
        for t in s_k_2:
            temp = pi[(t, u)] * log_linear_model[(t, u, v), word]
            if temp > max:
                max = temp
                argmax = t
        return max, argmax

    def viterbi(self, words, log_linear_model):
        n = len(words)
        T = np.zeros(n)
        pi_array = []
        bp_array = []
        pi_array.append({('*', '*'): 1})
        bp_array.append(0)
        s_k_1 = s_k_2 = s_k = self.tags
        for k, word in enumerate(words):
            k += 1
            pi = {}
            bp = {}
            if k == 0:
                s_k_1 = ['*']
                s_k_2 = ['*']
            elif k == 1:
                s_k_2 = ['*']
            for u in s_k_1:
                for v in s_k:
                    pi[(u, v)], bp[(u, v)] = self.find_max(pi_array[k - 1], u, v, s_k_2, log_linear_model, word)
            pi_array.append(pi)
            bp_array.append(bp)
        T[n - 1], T[n] = max(pi_array[n], key=pi_array[n].get)
        for i in reversed(range(n-2)):
            i=i+1
            T[i] = bp_array[i+2][(T[i+1],T[i+2])]
        return T



class MEMM_features():

    def __init__(self):
        self.n_total_features = 0
        self.f_indexes = {}
        self.f_statistics = {}

    def get_feature_statistics(self, file):
        f100_stats = self.get_f100_stats(file)
        f101_stats = self.get_f101_stats(file)
        f102_stats = self.get_f102_stats(file)
        f103_stats = self.get_f103_stats(file)
        f104_stats = self.get_f104_stats(file)
        f105_stats = self.get_f105_stats(file)
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

    def get_f100_stats(self, file):
        """
            Extract out of text all word/tag pairs
            :param file: full path of the file to read
                return all word/tag pairs with index of appearance
        """
        words_tags_count_dict = {}

        for line in file:
            words = line.split(' ')
            del words[-1]
            for word_idx in range(len(words)):
                cur_word, cur_tag = words[word_idx].split('_')
                if (cur_word, cur_tag) not in words_tags_count_dict.keys():
                    words_tags_count_dict[(cur_word, cur_tag)] = 1
                else:
                    words_tags_count_dict[(cur_word, cur_tag)] += 1
        return words_tags_count_dict

    def get_f101_stats(self, file):

        prefix_count_dict = {}

        for line in file:
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

    def get_f102_stats(self, file):

        suffix_count_dict = {}

        for line in file:
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

    def get_f103_stats(self, file):

        trigram_tags_count_dict = {}

        for line in file:
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

    def get_f104_stats(self, file):

        bigram_tags_count_dict = {}

        for line in file:
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

    def get_f105_stats(self, file):

        unigram_tags_count_dict = {}

        for line in file:
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
        features = []
        word, pptag, ptag, ctag, nword, pword = history

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
        if (pptag, ptag, ctag) in self.f_indexes[103]:
            features.append(self.f_indexes[103][(pptag, ptag, ctag)])

        # append features belonging to f104
        if (ptag, ctag) in self.f_indexes[104]:
            features.append(self.f_indexes[104][(ptag, ctag)])

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



