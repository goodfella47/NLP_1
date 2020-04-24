from MEMM2 import MEMM
import pickle
import numpy as np
from scipy import special

# %%
train_file = 'train1.wtag'
predict_file = 'test1.wtag'
memm = MEMM(lamda=1,threshold=1)


# weights_path = 'trained_weights_data_3.pkl'
# with open(weights_path, 'rb') as f:
#   optimal_params = pickle.load(f)
# pre_trained_weights = optimal_params
# memm.assign_weights(pre_trained_weights)


memm.fit(train_file)

weights_path = 'trained_weights_data_13_threshold1.pkl' # i identifies which dataset this is trained on
with open(weights_path, 'wb') as f:
    pickle.dump(memm.v, f)

memm.predict(predict_file,hide_tags=True)
print(memm.features.n_total_features)




