import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pickle
from keras.layers import Embedding, Input, add, Dropout, Dense
from tensorflow import keras
from keras.layers import Conv1D, MaxPooling1D, BatchNormalization, Flatten, concatenate
from keras.models import Model, load_model
from keras import regularizers
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import Adam

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
pickle_file_path = 'epoch_92400_sample_23500000.pkl'
with open(pickle_file_path, 'rb') as file:
    loaded_matrix = pickle.load(file)
# print([len(x) for x in embedding_matrix])
# print(len(embedding_matrix))
# print(embedding_matrix[1])
print('Found embedding matrix with dimension: ', embedding_matrix.shape)
EMBEDDING_DIM=embedding_matrix.shape[1]
MAX_SEQUENCE_LENGTH=x_train.shape[1]

print(loaded_matrix[0])
print(len(loaded_matrix[1]))
print(len(loaded_matrix[2]))

input()
embedding_layer = Embedding(len(embedding_matrix), EMBEDDING_DIM, weights=[loaded_matrix], input_length=MAX_SEQUENCE_LENGTH, trainable=True)
sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
embedded_sequences = embedding_layer(sequence_input)
conv1 = Conv1D(250, kernel_size=20, kernel_initializer='glorot_uniform', activation='relu')(embedded_sequences)
pool1 = MaxPooling1D(pool_size=981, strides=1000)(conv1)


conv2 = Conv1D(250, kernel_size=8, kernel_initializer='glorot_uniform', activation='relu')(embedded_sequences)
pool2 = MaxPooling1D(pool_size=993, strides=1000)(conv2)

conv3 = Conv1D(250, kernel_size=19, kernel_initializer='glorot_uniform', activation='relu')(embedded_sequences)
pool3 = MaxPooling1D(pool_size=982, strides=1000)(conv3)

merg = add([pool1, pool2, pool3])
drop1 = Dropout(rate=0.35)(merg)
concat=concatenate([pool1, drop1], axis=-1)
flat = Flatten()(concat)
drop2 = Dropout(rate=0.35)(flat)
hidden1 = Dense(2000, activation='relu')(drop2)
batch1 = BatchNormalization(axis=-1, momentum=0.99)(hidden1)

output = Dense(86, kernel_regularizer=regularizers.l1_l2(l1=0.0005, l2=0.0005), activation='softmax')(batch1)
model = Model(inputs=sequence_input, outputs=output)

print(model.summary())

model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.0005), metrics=['accuracy'])

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)
mc = ModelCheckpoint('best_model_deepPff.h5', monitor='val_accuracy', mode='max', verbose=1, save_best_only=True)

history = model.fit(x_train, labels_train, validation_data=(x_test, labels_test), batch_size=30, epochs=1, verbose=1,
                    callbacks=[es, mc])

saved_model = load_model('best_model_deepPff.h5')