import sys
import random

with open('readme.txt') as f:
    lines = f.readlines()

res_dict = {}
j = 0
for i in range(0, len(lines)):
    x = lines[i].split()
    if not x[0] in res_dict:
        res_dict[x[0]]=[]

for i in range(0, len(lines)):
    #res_dict[lines[i]] = lst[i + 1]
    x = lines[i].split()
    res_dict[x[0]].append(x[1])

dic_spk={}
for i , j in enumerate(list(res_dict.keys())):
    dic_spk[i]=j

print(dic_spk)

#print(random.choice(res_dict['id01739']))
#print(list(res_dict.keys()))

