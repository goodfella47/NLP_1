from MEMM import MEMM

file = 'train1.wtag'
memm = MEMM()
memm.fit(file)
print(memm.words_tags_count_dict[('The', 'DT')])
