import numpy as np
from scipy.optimize import fmin_l_bfgs_b

default_features = (
    'f100', 'f101', 'f102', 'f103', 'f104', 'f105', 'f106', 'f107', 'f201', 'f202', 'f203', 'f204', 'f205', 'f206',
    'f207',
    'f208', 'f209', 'f210')


class MEMM():
    """
        MEMM part of speech tagger implementation
        Execution example:

        memm = MEMM()
        memm.fit(train_file_path)
        memm.predict(predict_file_path)

    """

    def __init__(self, threshold=1, lamda=1, features=default_features):
        self.v = None  # learned vector of weights
        self.tags = []
        self.tags_with = []  # same as tags but with additional '*' and 'STOP' tags
        self.lamda = lamda  # regularization parameter
        self.threshold = threshold  # minimum feature frequency
        self.features = self.initialize_features(features)  # stores all feature objects
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

    def predict(self, file_path, hide_tags=True, beam_width=2, prediction_path='predicted.wtag'):
        """
            This function predicts the tags for each word,
            given a corpus of sentences and writes the word,tag pairs to a new file

            :param file_path: full path of the file to read
            :param hide_tags: True if tags are present predict file
            :param beam_width: Width for beam search
            :param prediction_path: full path to write the predicted word,tag pairs
        """
        assert (any(self.v))
        num_of_predicted_words = 0  # accuracy denominator
        num_falsely_predicted_words = 0  # accuracy nominator
        first_line = True
        with open(prediction_path, 'w+') as predict_file:
            with open(file_path) as file:  # iterates over all sentences and predicts tags
                for line in file:
                    if not first_line:
                        predict_file.write('\n')
                    if first_line:
                        first_line = False
                    words = line.rstrip('\n').split(' ')
                    num_of_predicted_words += len(words)
                    if hide_tags:
                        splitted_words = [word.split('_')[0] for word in words]
                        real_tags = [word.split('_')[1] for word in words]
                    else:
                        splitted_words = words
                    corrected_words = splitted_words.copy()
                    corrected_words.insert(0, '*')
                    corrected_words.insert(0, '*')
                    corrected_words.append('STOP')
                    predicted_tags = self.viterbi(corrected_words, beam_width)
                    accuracy = num_falsely_predicted_words / num_of_predicted_words
                    predicted_word_tag_pairs = [word + '_' + tag for word, tag in zip(splitted_words, predicted_tags)]
                    to_write = ' '.join(predicted_word_tag_pairs)
                    predict_file.write(to_write)
                    print(predicted_word_tag_pairs)
        if hide_tags:
            print('accuracy =', accuracy)

    def predict_with_predefined_weights(self, train_path, to_predict_path, weights, hide_tags=False, beam_width=None,
                                        prediction_path='predicted.wtag'):
        """
            This function is same as predict, assigns learned weights with pkl file prior to the prediction procedure
            :param: same as predict
        """
        self.get_tags(train_path)
        self.activate_features_on_data(train_path)
        self.assign_weights(weights)
        self.predict(file_path=to_predict_path, hide_tags=hide_tags, beam_width=beam_width,
                     prediction_path=prediction_path)

    def get_tags(self, file_path):
        """
            This function iterates over the corpus words and extracts the tags to self.tags
            :param file_path: full path of the file to read
        """
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
        """
            This function gets statistics such as feature count from the features
            :param file_path: full path of the file to read
        """
        for f in self.features.values():
            history_iter = history_generator(file_path)
            f.activate(history_iter)
        self.assign_feature_vector_indices()

    def assign_feature_vector_indices(self):
        """
            This function filters the features with frequency less than self.threshold
            and assigns unique id to every feature
        """
        f_statistics = {}
        for f_name, f in self.features.items():
            f_statistics[f_name] = f.f_statistics

        # filter features by a threshold
        if (self.threshold is not None):
            for i, f_stats in f_statistics.items():
                f = {k: v for (k, v) in f_stats.items() if v >= self.threshold}
                f_statistics[i] = f
        # assign unique id to features
        current_length = 0
        for i, f_index in f_statistics.items():
            f_index_dict = dict.fromkeys(f_index.keys(), 0)
            f_index_dict.update(zip(f_index_dict, range(current_length, len(f_index_dict) + current_length)))
            current_length += len(f_index_dict)
            self.features[i].indices = f_index_dict

        self.n_total_features = current_length + 1

    def find_vector_of_weights(self, file_path):
        """
            This function iterates over the corpus words and extracts the tags to self.tags
            :param file_path: full path of the file to read
        """
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
        """
            This function calculates normalization term and expected counts for likelihood gradient function
            :param v: vector of weights for the current iteration
            :param extended_history_dict: history with (history:feature_vector) pairs
                Return numeric:normalization term and numeric:expected counts
        """
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
        normalization_term, expected_counts = self.normalization_term_and_expected_counts(v, extended_history_dict)
        regularization = 0.5 * lamda * (v.dot(v))
        empirical_counts = linear_coef
        regularization_grad = lamda * v
        likelihood = linear_term - normalization_term - regularization
        grad = empirical_counts - expected_counts - regularization_grad
        return (-1) * likelihood, (-1) * grad

    def activate_features(self, features_in_str):
        """
             This function assigns the class of features MEMM will use for training
             :param features_in_str: list of strings, each string represent a feature class
         """
        features_group = []
        for f in features_in_str:
            features_group.append(eval(f + '()'))

    def assign_weights(self, v):
        """
             This function assigns vector of weights to the class from an external source
             :param v: vector of weights
         """
        self.v = v.copy()

    def loglinear_calc(self, history):
        """
             This is the q function of the Viterbi algorithm
             :param history: history representation of word in sentence
         """
        word, pptag, ptag, ctag, nword, pword = history
        current = {}
        for ntag in self.tags_with:
            history_with_ntag = (word, pptag, ptag, ntag, nword, pword)
            f = self.history_representation_with_features(history_with_ntag)
            exp = 0
            for j in f:
                exp += self.v[j]
            current[ntag] = np.exp(exp)
        denominator_sum = sum(list(current.values()))
        return current[ctag] / denominator_sum

    def find_max(self, pi, u, v, Sk_2, words, k):
        """
             Helper function for the Viterbi algorithm
         """
        max = 0
        argmax = Sk_2[0]
        for t in Sk_2:
            if (t, u) in pi.keys():
                history = (words[k + 2], t, u, v, words[k + 3], words[k + 1])
                temp = pi[(t, u)] * self.loglinear_calc(history)
                if temp > max:
                    max = temp
                    argmax = t
        return (max, argmax)

    def viterbi(self, words, beam_width):
        """
             The Viterbi algorithm
             :param words: list of words of current sentence
             :param beam_width: integer for beam search
                The function returns a list of predicted tags for the sentence
         """
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
                    pi[(u, v)], bp[(u, v)] = self.find_max(pi_array[k - 1], u, v, Sk_2, words, k - 1)
            if (beam_width):
                pi, bp = filter_by_beam(pi, bp, beam_width)
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


# feature classes for MEMM
class MEMM_feature_class():
    # superclass of feature classes

    def __init__(self):
        self.num_of_features = 0
        self.statistics = {}
        self.indices = {}

    def activate(self, history):
        """
            This function calls for the first functions to execute upon initializing a feature.
            :param history: generator of history word representation of every word in the corpus
        """
        pass

    def get_stats(selfs, history):
        """
            This function calculates the frequency of every feature in the corpus file
            :param file: generator of history word representation of every word in the corpus
        """
        pass

    def assign_feature_vec(self, features, history):
        pass


class f100(MEMM_feature_class):
    """
        <current word,current tag> feature representation
    """

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
    """
        <prefix,current tag> feature representation
    """

    def __init__(self):
        super().__init__()

    def activate(self, history):
        self.get_stats(history)

    def get_stats(self, history_generator):
        stats_dict = {}
        for history in history_generator:
            word, _, _, ctag, _, _ = history
            for i in reversed(range(1, min(10, len(word) + 1))):
                prefix = word[:i].lower()
                if (prefix, ctag) not in stats_dict:
                    stats_dict[(prefix, ctag)] = 1
                else:
                    stats_dict[(prefix, ctag)] += 1
        self.f_statistics = stats_dict

    def assign_feature_vec(self, features, history):
        word, _, _, ctag, _, _ = history
        for i in reversed(range(1, min(10, len(word) + 1))):
            prefix = word[:i].lower()
            if (prefix, ctag) in self.indices:
                features.append(self.indices[(prefix, ctag)])


class f102(MEMM_feature_class):
    """
        <suffix,current tag> feature representation
    """

    def __init__(self):
        super().__init__()

    def activate(self, history):
        self.get_stats(history)

    def get_stats(self, history_generator):
        stats_dict = {}
        for history in history_generator:
            word, _, _, ctag, _, _ = history
            for i in reversed(range(1, min(10, len(word) + 1))):
                suffix = word[-i:].lower()
                if (suffix, ctag) not in stats_dict:
                    stats_dict[(suffix, ctag)] = 1
                else:
                    stats_dict[(suffix, ctag)] += 1
        self.f_statistics = stats_dict

    def assign_feature_vec(self, features, history):
        word, _, _, ctag, _, _ = history
        for i in reversed(range(1, min(10, len(word) + 1))):
            suffix = word[-i:].lower()
            if (suffix, ctag) in self.indices:
                features.append(self.indices[(suffix, ctag)])


class f103(MEMM_feature_class):
    """
        <previous previous tag, previous tag, current tag> feature representation
    """

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
    """
        <previous tag, current tag> feature representation
    """

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
    """
        <current tag> feature representation
    """

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
    """
        <previous word, current tag> feature representation
    """

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
    """
        <next word, current tag> feature representation
    """

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


class f201(MEMM_feature_class):
    """
        <is there an upper case in current word, current tag> feature representation
    """

    def __init__(self):
        super().__init__()

    def activate(self, history):
        self.get_stats(history)

    def get_stats(self, history_generator):
        stats_dict = {}
        for history in history_generator:
            word, _, _, ctag, _, _ = history
            if (any(x.isupper() for x in word)):
                if ('supper', ctag) not in stats_dict:
                    stats_dict[('supper', ctag)] = 1
                else:
                    stats_dict[('supper', ctag)] += 1
        self.f_statistics = stats_dict

    def assign_feature_vec(self, features, history):
        word, _, _, ctag, _, _ = history
        if (any(x.isupper() for x in word)) and ('supper', ctag) in self.indices:
            features.append(self.indices[('supper', ctag)])


class f202(MEMM_feature_class):
    """
        <is there a hyphen in current word, current tag> feature representation
    """

    def __init__(self):
        super().__init__()

    def activate(self, history):
        self.get_stats(history)

    def get_stats(self, history_generator):
        stats_dict = {}
        for history in history_generator:
            word, _, _, ctag, _, _ = history
            if '-' in word:
                if ('-', ctag) not in stats_dict:
                    stats_dict[('-', ctag)] = 1
                else:
                    stats_dict[('-', ctag)] += 1
        self.f_statistics = stats_dict

    def assign_feature_vec(self, features, history):
        word, _, _, ctag, _, _ = history
        if '-' in word and ('-', ctag) in self.indices:
            features.append(self.indices[('-', ctag)])


class f203(MEMM_feature_class):
    """
        <is there a digit in current word, current tag> feature representation
    """

    def __init__(self):
        super().__init__()

    def activate(self, history):
        self.get_stats(history)

    def get_stats(self, history_generator):
        stats_dict = {}
        for history in history_generator:
            word, _, _, ctag, _, _ = history
            if (any(x.isdigit() for x in word)):
                if ('digit', ctag) not in stats_dict:
                    stats_dict[('digit', ctag)] = 1
                else:
                    stats_dict[('digit', ctag)] += 1
        self.f_statistics = stats_dict

    def assign_feature_vec(self, features, history):
        word, _, _, ctag, _, _ = history
        if (any(x.isdigit() for x in word)) and ('digit', ctag) in self.indices:
            features.append(self.indices[('digit', ctag)])


class f204(MEMM_feature_class):
    """
        <word, previous tag, current tag> feature representation
    """

    def __init__(self):
        super().__init__()

    def activate(self, history):
        self.get_stats(history)

    def get_stats(self, history_generator):
        stats_dict = {}
        for history in history_generator:
            word, _, ptag, ctag, _, _ = history
            if (word, ptag, ctag) not in stats_dict:
                stats_dict[(word, ptag, ctag)] = 1
            else:
                stats_dict[(word, ptag, ctag)] += 1
        self.f_statistics = stats_dict

    def assign_feature_vec(self, features, history):
        word, _, ptag, ctag, _, _ = history
        if (word, ptag, ctag) in self.indices:
            features.append(self.indices[(word, ptag, ctag)])


class f205(MEMM_feature_class):
    """
        <word, previous tag, current tag> feature representation
    """

    def __init__(self):
        super().__init__()

    def activate(self, history):
        self.get_stats(history)

    def get_stats(self, history_generator):
        stats_dict = {}
        for history in history_generator:
            word, _, _, ctag, _, pword = history
            if (word, pword, ctag) not in stats_dict:
                stats_dict[(word, pword, ctag)] = 1
            else:
                stats_dict[(word, pword, ctag)] += 1
        self.f_statistics = stats_dict

    def assign_feature_vec(self, features, history):
        word, _, _, ctag, _, pword = history
        if (word, pword, ctag) in self.indices:
            features.append(self.indices[(word, pword, ctag)])


class f206(MEMM_feature_class):
    """
        <word, next word, current tag> feature representation
    """

    def __init__(self):
        super().__init__()

    def activate(self, history):
        self.get_stats(history)

    def get_stats(self, history_generator):
        stats_dict = {}
        for history in history_generator:
            word, _, _, ctag, nword, _ = history
            if (word, nword, ctag) not in stats_dict:
                stats_dict[(word, nword, ctag)] = 1
            else:
                stats_dict[(word, nword, ctag)] += 1
        self.f_statistics = stats_dict

    def assign_feature_vec(self, features, history):
        word, _, _, ctag, nword, _ = history
        if (word, nword, ctag) in self.indices:
            features.append(self.indices[(word, nword, ctag)])


class f207(MEMM_feature_class):
    """
        <are all letters uppercase in current word, current tag> feature representation
    """

    def __init__(self):
        super().__init__()

    def activate(self, history):
        self.get_stats(history)

    def get_stats(self, history_generator):
        stats_dict = {}
        for history in history_generator:
            word, _, _, ctag, _, _ = history
            if (word.isupper()):
                if ('allUpper', ctag) not in stats_dict:
                    stats_dict[('allUpper', ctag)] = 1
                else:
                    stats_dict[('allUpper', ctag)] += 1
        self.f_statistics = stats_dict

    def assign_feature_vec(self, features, history):
        word, _, _, ctag, _, _ = history
        if (word.isupper()) and ('allUpper', ctag) in self.indices:
            features.append(self.indices[('allUpper', ctag)])


class f208(MEMM_feature_class):
    """
        <XXxx_dd current word representation, current tag> feature representation
        where X is an uppercase letter, x is a lowercase letter, _ is _ and d is a digit
    """

    def __init__(self):
        super().__init__()

    def activate(self, history):
        self.get_stats(history)

    def get_stats(self, history_generator):
        stats_dict = {}
        for history in history_generator:
            word, _, _, ctag, _, _ = history
            transWord = word_shape_transformer(word)
            if (transWord, ctag) not in stats_dict:
                stats_dict[(transWord, ctag)] = 1
            else:
                stats_dict[(transWord, ctag)] += 1
        self.f_statistics = stats_dict

    def assign_feature_vec(self, features, history):
        word, _, _, ctag, _, _ = history
        transWord = word_shape_transformer(word)
        if (transWord, ctag) in self.indices:
            features.append(self.indices[(transWord, ctag)])


class f209(MEMM_feature_class):
    """
        <XXxx_dd previous word representation, current tag> feature representation
        where X is an uppercase letter, x is a lowercase letter, _ is _ and d is a digit
    """

    def __init__(self):
        super().__init__()

    def activate(self, history):
        self.get_stats(history)

    def get_stats(self, history_generator):
        stats_dict = {}
        for history in history_generator:
            _, _, _, ctag, _, pword = history
            transWord = word_shape_transformer(pword)
            if (transWord, ctag) not in stats_dict:
                stats_dict[(transWord, ctag)] = 1
            else:
                stats_dict[(transWord, ctag)] += 1
        self.f_statistics = stats_dict

    def assign_feature_vec(self, features, history):
        word, _, _, ctag, _, pword = history
        transWord = word_shape_transformer(pword)
        if (transWord, ctag) in self.indices:
            features.append(self.indices[(transWord, ctag)])


class f210(MEMM_feature_class):
    """
        <XXxx_dd next word representation, current tag> feature representation
        where X is an uppercase letter, x is a lowercase letter, _ is _ and d is a digit
    """

    def __init__(self):
        super().__init__()

    def activate(self, history):
        self.get_stats(history)

    def get_stats(self, history_generator):
        stats_dict = {}
        for history in history_generator:
            _, _, _, ctag, nword, _ = history
            transWord = word_shape_transformer(nword)
            if (transWord, ctag) not in stats_dict:
                stats_dict[(transWord, ctag)] = 1
            else:
                stats_dict[(transWord, ctag)] += 1
        self.f_statistics = stats_dict

    def assign_feature_vec(self, features, history):
        _, _, _, ctag, nword, _ = history
        transWord = word_shape_transformer(nword)
        if (transWord, ctag) in self.indices:
            features.append(self.indices[(transWord, ctag)])


def history_generator(file_path):
    """
           This function generates history from every word in the corpus
           :param file_path: full file path of the corpus
                returns history generator
       """
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
    """
            This function creates dictinary of <history word representation, feature vector of current history>
            :param file_path: full file path of the corpus
            :param feature_representation: function that gets history and outputs the feature representing it
            :param tags: list of every tag presented in the corpus
    """
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


def word_shape_transformer(word):
    transWord = ''
    for char in word:
        if char.isupper():
            transWord += 'X'
        elif char.isalpha():
            transWord += 'x'
        elif char.isdigit():
            transWord += 'd'
        else:
            transWord += char
    return transWord


def filter_by_beam(pi, bp, beam_width):
    filtered_pi = {}
    filtered_bp = {}
    for _ in range(beam_width):
        curr_max_key = max(pi, key=pi.get)
        filtered_pi[curr_max_key] = pi[curr_max_key]
        filtered_bp[curr_max_key] = bp[curr_max_key]
        del pi[curr_max_key]
    return filtered_pi, filtered_bp
