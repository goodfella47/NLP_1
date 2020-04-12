from MEMM import MEMM
import numpy as np
from scipy import special

file = 'train1.wtag'
memm = MEMM()
memm.fit(file)
print(memm.words_tags_count_dict[('The', 'DT')])



##inference


# def find_max(s,u,v,log_linear_model,word,observation):
#     max=0
#     argmax=s[0]
#     for t in s:
#         temp=observation[(t,u)][0]*log_linear_model[(t,u,v),word]
#         if temp>max:
#             max=temp
#             argmax=t
#     return max,argmax
#
#
# def argmax(tupels,log_linear_model):
#     max=0
#     for tuple in tupels:
#         temp=tupels[tuple]
#         if temp>max:
#             max=temp
#             argmax=tupl
#     return argmax
#
#
#
# def max_prob(sentence,log_linear_model):
#     observations=[]
#     s=[]
#     T=[]
#     for index,word in enumerate(sentence):
#         d={}
#         for v in s[index]:
#             for u in s[index-1]:
#                 d[(u, v)] =find_max(s[index-2],u,v,log_linear_model,word,observations[index-1])
#         observations[index] = insert(d)
#     T[index-1],T[index]=argmax(observations[index], log_linear_model)
#     i = len(sentecne) - 3
#     while i>=0:
#         T[i]=observations[i+2][(T[i+1],T[i+2])]
#         i=i-1
#     return T
#
