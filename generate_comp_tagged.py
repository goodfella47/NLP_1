from MEMM import MEMM
import pickle

memm = MEMM()

weights_path = 'train1_weights.pkl'
with open(weights_path, 'rb') as f:
    optimal_params = pickle.load(f)
weights = optimal_params

memm.predict_with_predefined_weights('train1.wtag', 'comp1.words', weights=weights,
                                     prediction_path='comp_m1_305320400.wtag', beam_width=3)

weights_path = 'train2_weights.pkl'
with open(weights_path, 'rb') as f:
    optimal_params = pickle.load(f)
weights = optimal_params

memm.predict_with_predefined_weights('train2.wtag', 'comp2.words', weights=weights,
                                     prediction_path='comp_m2_305320400.wtag', beam_width=3)



