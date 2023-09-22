import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pickle

matplotlib.use("Agg")
seed = 219
np.random.seed(seed)
with open('GPCR_all_train_sequences.pkl', "rb") as fp_seq:
    x_train = pickle.load(fp_seq)
print('Shape of train input tensor: ', x_train.shape)
# print([k[:10] for k in x_train[:20]])
GPCR_all_train_labels = np.load('GPCR_all_train_labels.npy')
labels_train= GPCR_all_train_labels
print('Shape of train output tensor: ', labels_train.shape)
# print(labels_train[3])
with open('GPCR_all_test_sequences.pkl', "rb") as fp_seq:
    x_test = pickle.load(fp_seq)
print('Shape of test input tensor: ', x_test.shape)
GPCR_all_test_labels = np.load('GPCR_all_test_labels.npy')
labels_test= GPCR_all_test_labels
print('Shape of test output tensor: ', labels_test.shape)
with open('char_ix_train.pkl', "rb") as fp_label:
    embeddings_index = pickle.load(fp_label)
# print(embeddings_index)
embedding_matrix = np.load('embeddings_train.npy')
# print([len(x) for x in embedding_matrix])
# print(len(embedding_matrix))
# print(embedding_matrix[1])
print('Found embedding matrix with dimension: ', embedding_matrix.shape)
EMBEDDING_DIM=embedding_matrix.shape[1]
MAX_SEQUENCE_LENGTH=x_train.shape[1]

