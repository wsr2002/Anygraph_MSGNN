import pickle
import MSGNN

with open('./zero-shot datasets/Photo/trn_mat.pkl', 'rb') as f:
    metrics = pickle.load(f)
#print(metrics==metrics.T)
print(metrics.shape)

