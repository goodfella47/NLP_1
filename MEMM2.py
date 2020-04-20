import numpy as np
from scipy.optimize import fmin_l_bfgs_b
import time

default_features = ('f100', 'f101', 'f102', 'f103', 'f104', 'f105', 'f106', 'f107', 'f201')

prefixes = (
    're', 'dis', 'over', 'un', 'mis', 'out', 'co', 'sub', 'un', 'im', 'in', 'ir', 'il', 'non', 'de', 'ex', 'pre', 'pro')

suffixes = (
    'ly', 'ing', 'ise', 'ate', 'fy', 'en', 'tion', 'ity', 'er', 'ness', 'ism', 'ment', 'ant', 'ship', 'age', 'ery',
    'int',
    'able', 'al', 'est', 'ful', 'ible', 'ily', 'es', 'ies', 's')


class MEMM():

    def __init__(self, threshold=None, lamda=1, features=default_features):
        self.v = None
        self.tags = []
        self.tags_with = []  # same as tags but with additional '*' and 'STOP' tags
        self.lamda = lamda
        self.threshold = threshold
        self.features = self.initialize_features(features)
        self.n_total_features = 0
        self.accuracy = 0

    def initialize_features(self, features_in_str):
        features_group = {}
        for f in features_in_str:
            features_group[f] = (eval(f + '()'))
        return features_group

    def fit(self, file_path):
        """
            Fits the model on the data
            :param file_path: full path of the file to read
        """
        self.get_tags(file_path)
        self.activate_features_on_data(file_path)
        weights = self.find_vector_of_weights(file_path)
        self.v = weights

    def predict(self, file_path, hide_tags=True):
        assert (any(self.v))
        num_of_predicted_words = 0
        num_falsely_predicted_words = 0
        with open(file_path) as file:
            for line in file:
                words = line.rstrip('\n').split(' ')
                num_of_predicted_words += len(words)
                if hide_tags:
                    splitted_words = [word.split('_')[0] for word in words]
                    tags = [word.split('_')[1] for word in words]
                corrected_words = splitted_words.copy()
                corrected_words.insert(0, '*')
                corrected_words.insert(0, '*')
                corrected_words.append('STOP')
                loglinear_model = self.loglinear_model(corrected_words)
                T = self.viterbi(corrected_words, loglinear_model)
                print(T)
                print(tags)
                num_falsely_predicted_words += list_difference(T, tags)
                print(num_falsely_predicted_words / num_of_predicted_words)
                print('-----------------------------------------------------------------------------------------------')

    def get_tags(self, file_path):
        tags = set()
        history_iter = history_generator(file_path)
        for _, _, _, ctag, _, _ in history_iter:
            tags.add(ctag)
        tags_with = tags.copy()
        tags_with.add('*')
        tags_with.add('STOP')
        self.tags = list(tags)
        self.tags_with = list(tags_with)

    def activate_features_on_data(self, file_path):
        for f in self.features.values():
            history_iter = history_generator(file_path)
            f.activate(history_iter)
        self.assign_feature_vector_indices()

    def assign_feature_vector_indices(self):

        f_statistics = {}
        for f_name, f in self.features.items():
            f_statistics[f_name] = f.f_statistics

        # filter features by a threshold
        if (self.threshold is not None):
            for i, f_stats in f_statistics.items():
                f = {k: v for (k, v) in f_stats.items() if v > self.threshold}
                f_statistics[i] = f

        current_length = 0
        for i, f_index in f_statistics.items():
            f_index_dict = dict.fromkeys(f_index.keys(), 0)
            f_index_dict.update(zip(f_index_dict, range(current_length, len(f_index_dict) + current_length)))
            current_length += len(f_index_dict)
            self.features[i].indices = f_index_dict

        self.n_total_features = current_length + 1

    def find_vector_of_weights(self, file_path):
        v0 = np.random.rand(self.n_total_features) / 100
        history_dict, extended_history_dict = complete_history_with_features(file_path,
                                                                             self.history_representation_with_features,
                                                                             self.tags_with)
        linear_coefficient = self.linear_coefficient_calc(history_dict)
        args = (linear_coefficient, self.lamda, extended_history_dict)
        optimal_params = fmin_l_bfgs_b(func=self.likelihood_gradient_function, x0=v0, args=args, maxiter=600,
                                       iprint=100)
        return optimal_params[0]

    def linear_coefficient_calc(self, history_dict):
        sum_f = np.zeros(self.n_total_features)
        for _, feature in history_dict.items():
            for i in feature:
                sum_f[i] += 1
        return sum_f

    def history_representation_with_features(self, history):
        """
            Extract feature vector in per a given history
            :param history: touple{word, pptag, ptag, ctag, nword, pword}
            :param word_tags_dict: word\tag dict
                Return a list with all features that are relevant to the given history
        """
        features = []
        for f in self.features.values():
            f.assign_feature_vec(features, history)
        return features

    def normalization_term_and_expected_counts(self, v, extended_history_dict):
        normalization_term = 0
        expectedCounts = np.zeros(len(v))
        for history, features in extended_history_dict.items():
            inner_log = 0
            expectedInnerSum = np.zeros(len(v))
            indexes = set()
            for f in features:
                vdotf = 0
                for i in f:
                    vdotf += v[i]
                exp_vdotf = np.exp(vdotf)
                inner_log += exp_vdotf
                for i in f:
                    expectedInnerSum[i] += exp_vdotf
                    indexes.add(i)
            for i in indexes:
                expectedCounts[i] += expectedInnerSum[i] / inner_log
            normalization_term += np.log(inner_log)
        return (normalization_term, expectedCounts)

    def likelihood_gradient_function(self, v, linear_coef, lamda, extended_history_dict):
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

    def activate_features(self, features_in_str):
        features_group = []
        for f in features_in_str:
            features_group.append(eval(f + '()'))

    def assign_weights(self, v):
        self.v = v.copy()

    def loglinear_model(self, words):
        loglinear_model_dict = {}
        for i, word in enumerate(words[2:-1]):
            for u in self.tags_with:
                for v in self.tags_with:
                    current = {}
                    for c in self.tags_with:
                        history = (word, u, v, c, words[i + 3], words[i + 1])
                        f = self.history_representation_with_features(history)
                        exp = 0
                        for j in f:
                            exp += self.v[j]
                        current[c] = np.exp(exp)
                    denominator_sum = sum(list(current.values()))
                    for c in self.tags_with:
                        loglinear_model_dict[(word, u, v, c, words[i + 3], words[i + 1])] = current[c] / denominator_sum
        return loglinear_model_dict

    def find_max(self, pi, u, v, Sk_2, log_linear_model, words, k):
        max = 0
        argmax = Sk_2[0]
        for t in Sk_2:
            temp = pi[(t, u)] * log_linear_model[(words[k + 2], t, u, v, words[k + 3], words[k + 1])]
            if temp > max:
                max = temp
                argmax = t
        return (max, argmax)

    def viterbi(self, words, log_linear_model):
        n = len(words) - 3
        T = [''] * n
        pi_array = []
        bp_array = []
        pi_array.append({('*', '*'): 1})
        bp_array.append(0)
        Sk = self.tags_with
        for k, word in enumerate(words[2:-1]):
            k = k + 1
            Sk_1, Sk_2 = self.find_Sk(k)
            pi = {}
            bp = {}
            for u in Sk_1:
                for v in Sk:
                    pi[(u, v)], bp[(u, v)] = self.find_max(pi_array[k - 1], u, v, Sk_2, log_linear_model, words, k - 1)
            pi_array.append(pi)
            bp_array.append(bp)
        T[n - 2], T[n - 1] = max(pi_array[n], key=pi_array[n].get)
        for i in reversed(range(n - 2)):
            T[i] = bp_array[i + 3][(T[i + 1], T[i + 2])]
        return T

    def find_Sk(self, k):
        if k == 1:
            return (['*'], ['*'])
        if k == 2:
            return (self.tags, ['*'])
        else:
            return (self.tags, self.tags)


# f1** class of features for MEMM by

class MEMM_feature_class():
    def __init__(self):
        self.num_of_features = 0
        self.statistics = {}
        self.indices = {}

    def activate(self, history):
        pass

    def get_stats(selfs, history):
        """
            Extract out of text all word/tag pairs
            :param file: history generator
                return all word/tag pairs with index of appearance
        """
        pass

    def assign_feature_vec(self, features, history):
        pass


class f100(MEMM_feature_class):
    def __init__(self):
        super().__init__()

    def activate(self, history):
        self.get_stats(history)

    def get_stats(self, history_generator):
        stats_dict = {}
        for history in history_generator:
            word, _, _, ctag, _, _ = history
            word = word.lower()
            if (word, ctag) not in stats_dict.keys():
                stats_dict[(word, ctag)] = 1
            else:
                stats_dict[(word, ctag)] += 1
        self.f_statistics = stats_dict

    def assign_feature_vec(self, features, history):
        word, _, _, ctag, _, _ = history
        word = word.lower()
        if (word, ctag) in self.indices:
            features.append(self.indices[(word, ctag)])


class f101(MEMM_feature_class):
    def __init__(self, prefix=prefixes):
        self.prefixes = prefix
        super().__init__()

    def activate(self, history):
        self.get_stats(history)

    def get_stats(self, history_generator):
        stats_dict = {}
        for history in history_generator:
            word, _, _, ctag, _, _ = history
            for prefix in self.prefixes:
                if word.lower().startswith(prefix):
                    if (prefix, ctag) not in stats_dict:
                        stats_dict[(prefix, ctag)] = 1
                    else:
                        stats_dict[(prefix, ctag)] += 1
        self.f_statistics = stats_dict

    def assign_feature_vec(self, features, history):
        word, _, _, ctag, _, _ = history
        for prefix in reversed(range(2, 5)):
            prefix = word[:prefix].lower()
            if (prefix, ctag) in self.indices:
                features.append(self.indices[(prefix, ctag)])
                break


class f102(MEMM_feature_class):
    def __init__(self, suffix=suffixes):
        self.suffixes = suffix
        super().__init__()

    def activate(self, history):
        self.get_stats(history)

    def get_stats(self, history_generator):
        stats_dict = {}
        for history in history_generator:
            word, _, _, ctag, _, _ = history
            for suffix in self.suffixes:
                if word.lower().startswith(suffix):
                    if (suffix, ctag) not in stats_dict:
                        stats_dict[(suffix, ctag)] = 1
                    else:
                        stats_dict[(suffix, ctag)] += 1
        self.f_statistics = stats_dict

    def assign_feature_vec(self, features, history):
        word, _, _, ctag, _, _ = history
        for suffix in reversed(range(2, 5)):
            suffix = word[:suffix].lower()
            if (suffix, ctag) in self.indices:
                features.append(self.indices[(suffix, ctag)])
                break


class f103(MEMM_feature_class):
    def __init__(self):
        super().__init__()

    def activate(self, history):
        self.get_stats(history)

    def get_stats(self, history_generator):
        stats_dict = {}
        for history in history_generator:
            _, pptag, ptag, ctag, _, _ = history
            if (pptag, ptag, ctag) not in stats_dict:
                stats_dict[(pptag, ptag, ctag)] = 1
            else:
                stats_dict[(pptag, ptag, ctag)] += 1
        self.f_statistics = stats_dict

    def assign_feature_vec(self, features, history):
        _, pptag, ptag, ctag, _, _ = history
        if (pptag, ptag, ctag) in self.indices:
            features.append(self.indices[(pptag, ptag, ctag)])


class f104(MEMM_feature_class):
    def __init__(self):
        super().__init__()

    def activate(self, history):
        self.get_stats(history)

    def get_stats(self, history_generator):
        stats_dict = {}
        for history in history_generator:
            _, _, ptag, ctag, _, _ = history
            if (ptag, ctag) not in stats_dict:
                stats_dict[(ptag, ctag)] = 1
            else:
                stats_dict[(ptag, ctag)] += 1
        self.f_statistics = stats_dict

    def assign_feature_vec(self, features, history):
        _, _, ptag, ctag, _, _ = history
        if (ptag, ctag) in self.indices:
            features.append(self.indices[(ptag, ctag)])


class f105(MEMM_feature_class):
    def __init__(self):
        super().__init__()

    def activate(self, history):
        self.get_stats(history)

    def get_stats(self, history_generator):
        stats_dict = {}
        for history in history_generator:
            _, _, _, ctag, _, _ = history
            if (ctag) not in stats_dict:
                stats_dict[(ctag)] = 1
            else:
                stats_dict[(ctag)] += 1
        self.f_statistics = stats_dict

    def assign_feature_vec(self, features, history):
        _, _, _, ctag, _, _ = history
        if (ctag) in self.indices:
            features.append(self.indices[(ctag)])


class f106(MEMM_feature_class):
    def __init__(self):
        super().__init__()

    def activate(self, history):
        self.get_stats(history)

    def get_stats(self, history_generator):
        stats_dict = {}
        for history in history_generator:
            _, _, _, ctag, _, pword = history
            if (pword, ctag) not in stats_dict:
                stats_dict[(pword, ctag)] = 1
            else:
                stats_dict[(pword, ctag)] += 1
        self.f_statistics = stats_dict

    def assign_feature_vec(self, features, history):
        _, _, _, ctag, _, pword = history
        if (pword, ctag) in self.indices:
            features.append(self.indices[(pword, ctag)])


class f107(MEMM_feature_class):
    def __init__(self):
        super().__init__()

    def activate(self, history):
        self.get_stats(history)

    def get_stats(self, history_generator):
        stats_dict = {}
        for history in history_generator:
            _, _, _, ctag, nword, _ = history
            if (nword, ctag) not in stats_dict:
                stats_dict[(nword, ctag)] = 1
            else:
                stats_dict[(nword, ctag)] += 1
        self.f_statistics = stats_dict

    def assign_feature_vec(self, features, history):
        _, _, _, ctag, nword, _ = history
        if (nword, ctag) in self.indices:
            features.append(self.indices[(nword, ctag)])


# f2** class of features by Robert and Ofri
class f201(MEMM_feature_class):
    def __init__(self):
        super().__init__()

    def activate(self, history):
        self.get_stats(history)

    def get_stats(self, history_generator):
        stats_dict = {'Big_Letter': 0}
        for history in history_generator:
            word, _, _, _, _, _ = history
            if str.isupper(word[0]):
                stats_dict['Big_Letter'] += 1
        self.f_statistics = stats_dict

    def assign_feature_vec(self, features, history):
        word, _, _, _, _, _ = history
        if str.isupper(word[0]):
            features.append(self.indices['Big_Letter'])


class f202(MEMM_feature_class):
    def __init__(self):
        super().__init__()

    def activate(self, history):
        self.get_stats(history)

    def get_stats(self, history_generator):
        stats_dict = {'end_of_sentence': 0}
        for history in history_generator:
            _, _, _, _, nword, _ = history
            if nword == 'STOP':
                stats_dict['end_of_sentence'] += 1
        self.f_statistics = stats_dict

    def assign_feature_vec(self, features, history):
        _, _, _, _, nword, _ = history
        if nword == 'STOP':
            features.append(self.indices['end_of_sentence'])


class f203(MEMM_feature_class):
    def __init__(self):
        super().__init__()

    def activate(self, history):
        self.get_stats(history)

    def get_stats(self, history_generator):
        stats_dict = {'start_of_sentence': 0}
        for history in history_generator:
            _, pptag, ptag, _, _, _ = history
            if pptag == '*' and ptag == '*':
                stats_dict['start_of_sentence'] += 1
        self.f_statistics = stats_dict

    def assign_feature_vec(self, features, history):
        _, pptag, ptag, _, _, _ = history
        if pptag == '*' and ptag == '*':
            features.append(self.indices['start_of_sentence'])


class f204(MEMM_feature_class):
    def __init__(self):
        super().__init__()

    def activate(self, history):
        self.get_stats(history)

    def get_stats(self, history_generator):
        stats_dict = {'2nd_in_sentence': 0}
        for history in history_generator:
            _, pptag, ptag, _, _, _ = history
            if pptag != '*' and ptag == '*':
                stats_dict['2nd_in_sentence'] += 1
        self.f_statistics = stats_dict

    def assign_feature_vec(self, features, history):
        _, pptag, ptag, _, _, _ = history
        if pptag == '*' and ptag != '*':
            features.append(self.indices['2nd_in_sentence'])


def history_generator(file_path):
    with open(file_path) as f:
        for line in f:
            words = line.rstrip('\n').split(' ')
            words.insert(0, '*_*')
            words.insert(0, '*_*')
            words.append('STOP_STOP')
            for i, word_idx in enumerate(words[2:-1]):
                word, ctag = word_idx.split('_')
                pword, ptag = words[i + 1].split('_')
                ppword, pptag = words[i].split('_')
                nword, ntag = words[i + 3].split('_')
                yield (word, pptag, ptag, ctag, nword, pword)


def complete_history_with_features(file_path, feature_representation, tags):
    history_dict = {}
    extended_history_dict = {}
    with open(file_path) as f:
        for line in f:
            words = line.rstrip('\n').split(' ')
            words.insert(0, '*_*')
            words.insert(0, '*_*')
            words.append('STOP_STOP')
            for i, word_idx in enumerate(words[2:-1]):
                word, ctag = word_idx.split('_')
                pword, ptag = words[i + 1].split('_')
                ppword, pptag = words[i].split('_')
                nword, ntag = words[i + 3].split('_')
                history = (word, pptag, ptag, ctag, nword, pword)
                f = feature_representation(history)
                if history not in history_dict.keys():
                    history_dict[history] = f
                    extended_history = []
                    for y in tags:
                        history = (word, pptag, ptag, y, nword, pword)
                        f = feature_representation(history)
                        extended_history.append(f)
                    extended_history_dict[history] = extended_history
    return history_dict, extended_history_dict


def list_difference(list1, list2):
    assert (len(list1) == len(list2))
    n = 0
    for i, j in zip(list1, list2):
        if i == j: n += 1
    return n
