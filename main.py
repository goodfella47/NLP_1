from MEMM2 import MEMM
import pickle
import time

# %%
train_file = 'train2_expanded.wtag'
predict_file = 'train1.wtag'
memm = MEMM(lamda=1,threshold=1)


# weights_path = 'train1_weights.pkl'
# with open(weights_path, 'rb') as f:
#   optimal_params = pickle.load(f)
# pre_trained_weights = optimal_params
# memm.assign_weights(pre_trained_weights)

start = time.time()
memm.fit(train_file)
end = time.time()
print('time=', (end - start) / 60)


weights_path = 'train2_final.pkl'
with open(weights_path, 'wb') as f:
    pickle.dump(memm.v, f)

# start = time.time()
# memm.predict(predict_file,hide_tags=True,beam_width=2)
# end = time.time()
# print('time=', (end - start) / 60)