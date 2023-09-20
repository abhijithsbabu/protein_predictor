import pickle
import numpy as np
import os

GPCR = "/Datasets/GPCR/"
current_dir = os.getcwd()
BASE_PATH = current_dir + GPCR
embedding_size = 21
embeddings = [[0] * embedding_size]
char_ix = {'#PADDING#': 0}
sequences_num = {"GPCR": 832}
class_num = {"GPCR": 86}

with open('embeddings.21', 'r') as file:
    file.readline()
    for i, line in enumerate(file.readlines()):
        cols = line.split()  
        c = cols[0] 
        v = cols[1:] 
        char_ix[
            c] = i + 1  
        embeddings.append(v) 

assert len(char_ix) == len(embeddings) 
embeddings = np.array(embeddings, dtype=np.float32)
np.save('embeddings_test.npy', embeddings)

with open('char_ix_test.pkl', 'wb') as file:
    pickle.dump(char_ix, file)


def save_file_smy(class_tag):
    print(class_tag)
    protein_file = BASE_PATH + "test_sequence.txt".format(class_tag)
    label_file = BASE_PATH + "test_label.pkl".format(class_tag)

    N = int(sequences_num[class_tag])

    M = 1000  
    all_sequences = np.zeros((N, M), dtype=np.int32)


    with open(label_file, "rb") as fp_label:
        labels = pickle.load(fp_label)
        all_labels = np.array(labels, dtype=np.float32)

    with open(protein_file) as fp_seq:
        i = 0
        for line in fp_seq:
            seqence1 = line.split()[-1].replace('B', '')
            seqence2 = seqence1.replace('O', '')
            seqence3 = seqence2.replace('J', '')
            seqence4 = seqence3.replace('U', '')
            seqence5 = seqence4.replace('Z', '')
            seqence6 = seqence5.replace('b', '')
            seqence8 = seqence6.strip("'")
            seqence = seqence8.strip('_')
            for j, c in enumerate(seqence[:M]):
                ix = char_ix[c]
                all_sequences[i][j] = ix
            i += 1
    print("Total number of test sequences = " + str(len(all_sequences)))
    print(all_sequences)
    with open('{0}_all_test_sequences.pkl'.format(class_tag), "wb") as fp:
        pickle.dump(all_sequences, fp)

    np.save('{0}_all_test_labels.npy'.format(class_tag), all_labels)
    print("Total number of test Labels" + "(" + "outputs" + ")" + " = " + str(len(all_labels)))
    print(all_labels)

if __name__ == '__main__':
    for key in sequences_num.keys():
        save_file_smy(key)