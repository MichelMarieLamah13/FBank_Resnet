import sys
import os

args = sys.argv[1:]



print(args[0])

train_list = args[0]
train_path = args[1]
data_list  = []
data_label = []
lines = open(train_list).read().splitlines()
dictkeys = list(set([x.split()[0] for x in lines]))
dictkeys.sort()
dictkeys = { key : ii for ii, key in enumerate(dictkeys) }
for index, line in enumerate(lines):
    speaker_label = dictkeys[line.split()[0]]
    file_name     = os.path.join(train_path, line.split()[1])
    data_label.append(speaker_label)
    data_list.append(file_name)

print (data_label)

