from MEMM import MEMM
import pickle
import numpy as np
from scipy import special

# %%
train_file = 'train1.wtag'
predict_file = 'test1.wtag'
memm = MEMM()
memm.fit(train_file, 5,lamda=0)

weights_path = 'trained_weights_data_1.pkl'
with open(weights_path, 'rb') as f:
  optimal_params = pickle.load(f)
pre_trained_weights = optimal_params
memm.assign_weights(pre_trained_weights)

memm.predict(predict_file,hide_tags=True)

print(memm.features.n_total_features)



# %%


# inference
#
# def find_max(s_k_2, u, v, log_linear_model, word, observation):
#     max = 0
#     argmax = s_k_2[0]
#     for t in s_k_2:
#         temp = observation[(t, u)][0] * log_linear_model[(t, u, v), word]
#         if temp > max:
#             max = temp
#             argmax = t
#     return max, argmax
#
#
# def argmax(tupels, log_linear_model):
#     max = 0
#     for tuple in tupels:
#         temp = tupels[tuple]
#         if temp > max:
#             max = temp
#             argmax = tuple
#     return argmax
#
#
# def max_prob(sentence, log_linear_model):
#     observations = []
#     s = []
#     T = []
#     for index, word in enumerate(sentence):
#         d = {}
#         for v in s[index]:
#             for u in s[index - 1]:
#                 d[(u, v)] = find_max(s[index - 2], u, v, log_linear_model, word, observations[index - 1])
#         observations.append(d)
#     T[index - 1], T[index] = argmax(observations[index], log_linear_model)
#     i = len(sentecne) - 3
#     while i >= 0:
#         T[i] = observations[i + 2][(T[i + 1], T[i + 2])]
#         i = i - 1
#     return T
#
#
#
# def find_max2(s_k_2, u, v, log_linear_model, word, observation):
#     max = 0
#     argmax = s_k_2[0]
#     for t in s_k_2:
#         temp = observation[(t, u)][0] * log_linear_model[(t, u, v), word]
#         if temp > max:
#             max = temp
#             argmax = t
#     return max, argmax
#
# def loglinear_model(sentence,tags):
#     for word in sentence:
#         for u in tags:
#             for v in tags:
#                 f =
#
#
# def viterbi(sentence, log_linear_model, taggs):
#     pi_array = []
#     bp_array = []
#     pi_array.append({('*', '*'): 1})
#     bp_array.append(0)
#     s_k_1 = s_k_2 = s_k = taggs
#     for k, word in enumerate(sentence):
#         k += 1
#         pi = {}
#         bp = {}
#         if k == 0:
#             s_k_1 = ['*']
#             s_k_2 = ['*']
#         elif k == 1:
#             s_k_2 = ['*']
#         for u in s_k_1:
#             for v in s_k:
#                 pi[(u, v)], bp[(u, v)] = find_max(pi_array[k - 1], u, v, s_k_2, log_linear_model, word)
#         pi_array.append(pi)
#         bp_array.append(bp)
