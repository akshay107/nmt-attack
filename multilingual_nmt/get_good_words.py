import numpy as np
import torch
import pickle
import sys
import io

src = str(sys.argv[1])
tgt = str(sys.argv[2])
folder = str(sys.argv[3])
vocab = pickle.load(open(folder+"processed.vocab.pickle","rb"))

lines = io.open("data/"+src+"_"+tgt+"/train."+src, encoding='utf-8', errors='ignore').readlines()

l = []
for line in lines:
    for word in line.split():
            if word not in l:
                    l.append(word)

good_words = []
index_words = []

for key in vocab:
    if vocab[key] in l:
            good_words.append(vocab[key])
            index_words.append(key)


for i in range(len(good_words)):
    assert good_words[i]==vocab[index_words[i]]


torch.save([good_words,index_words],folder+"good_words_in_vocab.pt")

print("Number of words:",len(good_words))


