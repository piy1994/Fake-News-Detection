"""

This is the script to make the Word Embedding of the input article and headline. It takes the wordvectors from 
the glove.6B.50d.txt file, which is supposed to be in the same folder as this file.

"""

import numpy as np
import csv
import sys
import pickle

print("Start loading Word Vector Dictionary")
wordVec={}
with open('glove.6B.50d.txt', encoding="utf8") as glove:
    count=0
    for line in glove:
        temp=line.split()
        l=len(temp)
        wordVec[' '.join(temp[0:l-50])]=list(map(np.float,temp[l-50:l]))
        count=count+1
        print(count)
print("Finish loading Word Vector Dictionary")
print("Start loading training stances\n")



with open('train_stances.csv', encoding="utf8") as csvfile_stance:
    stanceReader=csv.reader(csvfile_stance)
    stances=[]
    for row in stanceReader:
        temp=[]
        for c,c_ in zip(row[0],row[0][1:]):
            if c.isalnum() or c.isspace():
                temp.append(c.lower())
            else:
                if not c_.isspace():
                    temp.append(" ")
        stances.append([''.join(temp),row[1],row[2]])
print("Finish loading training stances\n")



print("Start loading training bodies\n")
with open('train_bodies.csv', encoding="utf8") as csvfile_body:
    bodyReader=csv.reader(csvfile_body)
    bodies={}
    for row in bodyReader:
        temp=[]
        for c,c_ in zip(row[1],row[1][1:]):
            if c.isalnum() or c.isspace():
                temp.append(c.lower())
            else:
                if not c_.isspace():
                    temp.append(" ")
        bodies[row[0]]=''.join(temp)
print("Finish loading training bodies")


print("Start merging training stances and training bodies\n")
raw_training_set=[]
for set in stances:
    raw_training_set.append([set[0],bodies[set[1]],set[2]])


print("Finish loading training stanes and training bodies\n")


stanceReader=None
bodyReader=None
stances=None
bodies=None
print("Start embedding\n")
with open('processed_data_embed.p', "wb") as processed_data:
    data=[]
    for sample in raw_training_set:
        title=sample[0].split()
        body=sample[1].split()
        label=sample[2]
        titleVec=np.zeros(50)
        bodyVec=np.zeros(50)
        for i in range(len(title)):
            if title[i] in wordVec:
                word=wordVec[title[i]]
                titleVec=np.add(titleVec,word)
        titleVec=np.divide(titleVec,np.sqrt(np.dot(titleVec,titleVec)))
        for i in range(len(body)):
            if body[i] in wordVec:
                word=wordVec[body[i]]
                bodyVec=np.add(bodyVec,word)
        bodyVec=np.divide(bodyVec,np.sqrt((np.dot(bodyVec,bodyVec))))
        if label == "unrelated":
            label=0
        if label == "discuss":
            label=1
        if label == "agree":
            label=2
        if label == "disagree":
            label=3
        data.append([titleVec-bodyVec,label])
    pickle.dump(data,processed_data)
print("Finish embedding\n")