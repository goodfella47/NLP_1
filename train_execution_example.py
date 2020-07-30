from MEMM import MEMM


train_file = 'train1.wtag'
predict_file = 'test1.wtag'
memm = MEMM(lamda=1,threshold=1)

memm.fit(train_file)
memm.predict(predict_file,hide_tags=True,beam_width=2)




