import numpy as np
from scipy import special
from collections import OrderedDict


class MEMM():

    def __init__(self):
        self.v = []
        self.features = MEMM_features()
        self.tag_sentences=[]

    def open_file(self,file_path):
        with open(file_path) as f:
            for line in f:
                words = line.split(' ')
                del words[-1]  # delete "." in end of sentence
                tag_sentences.append(words)

    def fit(self, file_path):
        """
            Fits the model on the data, ultimately finds the vector of weights
            :param file_path: full path of the file to read
                return vector of weights
        """
        self.open_file(file_path)
        self.features.get_feature_statistics(file_path)

    def predict(self, file_path):
        assert (self.v)


class MEMM_features():

    def __init__(self):
        self.n_total_features = 0
        self.f100_index = self.f101_index = self.f102_index = self.f103_index = self.f104_index = self.f105_index = {}
        self.f_indexes = {100: self.f100_index, 101: self.f101_index, 102: self.f102_index, 103: self.f103_index,
                          104: self.f104_index, 105: self.f105_index}
        self.f100_stats = self.f101_stats = self.f102_stats = self.f103_stats = self.f104_stats = self.f105_stats = {}
        self.f_stats = {100: self.f100_stats, 101: self.f101_stats, 102: self.f102_stats, 103: self.f103_stats,
                        104: self.f104_stats, 105: self.f105_stats}

    def get_feature_statistics(self, file_path):
        self.f100_stats = self.get_f100_stats(file_path)
        self.f101_stats = self.get_f101_stats(file_path)
        self.f102_stats = self.get_f102_stats(file_path)
        self.f103_stats = self.get_f103_stats(file_path)
        self.f104_stats = self.get_f104_stats(file_path)
        self.f105_stats = self.get_f105_stats(file_path)

    def get_feature_indices(self, threshold=None):
        if (threshold):
            for f in self.f_indexes.values():


    def get_f100_stats(self, file_path):
        """
            Extract out of text all word/tag pairs
            :param file_path: full path of the file to read
                return all word/tag pairs with index of appearance
        """
        words_tags_count_dict = {}

        for tuple in tag_words:
            if tuple not in words_tags_count_dict:
                words_tags_count_dict[tuple] = 1
            else:
                words_tags_count_dict[tuple] += 1
        return words_tags_count_dict

    def get_f101_stats(self, file_path):

        prefix_count_dict = {}

        with open(file_path) as f:
            for line in f:
                words = line.split(' ')
                del words[-1]  # delete "." in end of sentence
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
                del words[-1]  # delete "." in end of sentence
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

        file_path = 'train1.wtag'
        with open(file_path) as f:
            for line in f:
                words = line.split(' ')
                del words[-1]  # delete "." in end of sentence
                tags = [i.split('_')[1] for i in words]
                tags.insert(0, '*')
                tags.insert(0, '*')
                tags.append('STOP')
                for i,_ in enumerate(tags[:-2]):
                    if (tags[i],tags[i+1],tags[i+2]) not in trigram_tags_count_dict:
                        trigram_tags_count_dict[(tags[i],tags[i+1],tags[i+2])] = 1
                    else:
                        trigram_tags_count_dict[(tags[i],tags[i+1],tags[i+2])] += 1

    def get_f104_stats(self, file_path):

        bigram_tags_count_dict = {}

        file_path = 'train1.wtag'
        with open(file_path) as f:
            for line in f:
                words = line.split(' ')
                del words[-1]  # delete "." in end of sentence
                tags = [i.split('_')[1] for i in words]
                tags.insert(0, '*')
                tags.insert(0, '*')
                tags.append('STOP')
                for i, _ in enumerate(tags[:-1]):
                    if (tags[i], tags[i + 1]) not in bigram_tags_count_dict:
                        bigram_tags_count_dict[(tags[i], tags[i + 1])] = 1
                    else:
                        bigram_tags_count_dict[(tags[i], tags[i + 1])] += 1
        print(len(bigram_tags_count_dict))

    def get_f105_stats(self, file_path):

        unigram_tags_count_dict = {}

        with open(file_path) as f:
            for line in f:
                words = line.split(' ')
                del words[-1]  # delete "." in end of sentence
                tags = [i.split('_')[1] for i in words]
                tags.insert(0, '*')
                tags.insert(0, '*')
                tags.append('STOP')
                for i, _ in enumerate(tags):
                    if (tags[i]) not in unigram_tags_count_dict:
                        unigram_tags_count_dict[(tags[i])] = 1
                    else:
                        unigram_tags_count_dict[(tags[i])] += 1
        print(unigram_tags_count_dict)





# %%
prefixes = (('re', 'VB'), ('dis', 'VB'), ('over', 'VB'), ('un', 'VB'), ('mis', 'VB'), ('out', 'VB'),
            ('co', 'NN'), ('sub', 'NN'), ('un', 'JJ'), ('im', 'JJ'), ('in', 'JJ'), ('ir', 'JJ'),
            ('il', 'JJ'), ('non', 'JJ'), ('dis', 'JJ'))

prefixes_dict = dict(prefixes)

suffixes = (('ise', 'VB'), ('ate', 'VB'), ('fy', 'VB'), ('en', 'VB'), ('tion', 'NN'), ('ity', 'NN'),
            ('er', 'NN'), ('ness', 'NN'), ('ism', 'NN'), ('ment', 'NN'), ('ant', 'NN'), ('ship', 'NN'),
            ('age', 'NN'), ('ery', 'NN'))

suffixes_dict = dict(suffixes)

# %%
