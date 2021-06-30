import numpy as np

train_de = open("../en_de/train.de").readlines()
train_fr = open("../en_fr/train.fr").readlines()
train_inds = np.load("train_inds.npy")

dev_de = open("../en_de/dev.de").readlines()
dev_fr = open("../en_fr/dev.fr").readlines()
dev_inds = np.load("dev_inds.npy")

test_de = open("../en_de/test.de").readlines()
test_fr = open("../en_fr/test.fr").readlines()
test_inds = np.load("test_inds.npy")


train_de_new = open("./train.de","w")
train_fr_new = open("./train.fr","w")

dev_de_new = open("./dev.de","w")
dev_fr_new = open("./dev.fr","w")

test_de_new = open("./test.de","w")
test_fr_new = open("./test.fr","w")


for inds in train_inds:
    print(train_de[inds[0]].strip(),file=train_de_new)
    print(train_fr[inds[1]].strip(),file=train_fr_new)

for inds in dev_inds:
    print(dev_de[inds[0]].strip(),file=dev_de_new)
    print(dev_fr[inds[1]].strip(),file=dev_fr_new)

for inds in test_inds:
    print(test_de[inds[0]].strip(),file=test_de_new)
    print(test_fr[inds[1]].strip(),file=test_fr_new)

train_de_new.close()
train_fr_new.close()
dev_de_new.close()
dev_fr_new.close()
test_de_new.close()
test_fr_new.close()
