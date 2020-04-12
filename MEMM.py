import numpy as np
from scipy import special
from collections import OrderedDict


prefixes = (('re','VB'),('dis','VB'),('over','VB'),('un','VB'),('mis','VB'),('out','VB'),
            ('co','NN'),('sub','NN'),('un','JJ'),('im','JJ'),('in','JJ'),('ir','JJ'),
            ('il','JJ'),('non','JJ'),('dis','JJ'))


suffixes = (('ise','VB'),('ate','VB'),('fy','VB'),('en','VB'),('tion','NN'),('ity','NN'),
            ('er','NN'),('ness','NN'),('ism','NN'),('ment','NN'),('ant','NN'),('ship','NN'),
            ('age','NN'),('ery','NN'))


class MEMM():

    def __init__(self):
        self.n_total_features = 0
        self.words_tags_count_dict = OrderedDict()

    def fit(self, file_path):
        """
            Fits the model on the data, ultimately finds the vector of weights
            :param file_path: full path of the file to read
                return vector of weights
        """
        self.get_word_tag_pair_count(file_path)

    def predict(self, file_path):
        pass

    def get_word_tag_pair_count(self, file_path):
        """
            Extract out of text all word/tag pairs
            :param file_path: full path of the file to read
                return all word/tag pairs with index of appearance
        """
        with open(file_path) as f:
            for line in f:
                words = line.split(' ')
                del words[-1]  # delete "." in end of sentence
                for word_idx in range(len(words)):
                    cur_word, cur_tag = words[word_idx].split('_')
                    if (cur_word, cur_tag) not in self.words_tags_count_dict:
                        self.words_tags_count_dict[(cur_word, cur_tag)] = 1
                    else:
                        self.words_tags_count_dict[(cur_word, cur_tag)] += 1
        self.n_total_features = len(self.words_tags_count_dict)


