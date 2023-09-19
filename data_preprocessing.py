import os
import pickle

GPCR = "/Datasets/GPCR"

current_dir = os.getcwd()
data_path = current_dir + GPCR
print(data_path)

train_file = "{0}/cv_0/train.txt".format(data_path)
test_file = "{0}/cv_0/test.txt".format(data_path)

train_label_file = "{0}/train_label.pkl".format(data_path)
test_label_file = "{0}/test_label.pkl".format(data_path)

train_sequence_file = "{0}/train_sequence.txt".format(data_path)
test_sequence_file = "{0}/test_sequence.txt".format(data_path)

all_labels = []
all_labels_1 = []

with open(train_file, "rb") as fp, open(train_sequence_file, "w") as fp_w:
    for line in fp:
        idx, sequence = line.strip().split()
        label = [0 for i in range(86)] 
        label[int(idx)] = 1 
        all_labels.append(label)
        fp_w.write("{0}\n".format(sequence))

count = 1
for one_hot in all_labels:
    print(count, one_hot)
    count += 1

with open(train_label_file, "wb") as fp_w:
    pickle.dump(all_labels, fp_w, protocol=2)

with open(test_file, "rb") as fp, open(test_sequence_file, "w") as fp_w:
    for line in fp:
        idx, sequence = line.strip().split()
        label = [0 for i in range(86)]
        label[int(idx)] = 1
        all_labels_1.append(label)
        fp_w.write("{0}\n".format(sequence))

count_1 = 1
for one_hot_1 in all_labels_1:
    print(count_1, one_hot_1)
    count_1 += 1

with open(test_label_file, "wb") as fp_w:
    pickle.dump(all_labels_1, fp_w, protocol=2)


