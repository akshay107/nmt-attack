# uncompyle6 version 3.3.4
# Python bytecode 3.6 (3379)
# Decompiled from: Python 3.6.3 |Anaconda, Inc.| (default, Nov 20 2017, 20:41:42) 
# [GCC 7.2.0]
# Embedded file name: /home/cvpr/abijith/OpenNMT-py_old/utils_translate.py
# Compiled at: 2019-05-10 13:47:13
# Size of source mod 2**32: 2904 bytes
import torch, os, sys, numpy as np
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec

class Word2VecSimilarity:

    def __init__(self, glove_input_file, word2vec_output_file=None):
        self.glove_input_file = glove_input_file
        if not word2vec_output_file:
            self.word2vec_output_file = self.glove_input_file + '.word2vec'
        else:
            self.word2vec_output_file = word2vec_output_file
        if not os.path.isfile(self.word2vec_output_file):
            glove2word2vec(self.glove_input_file, self.word2vec_output_file)
        self.word2vec_model = KeyedVectors.load_word2vec_format((self.word2vec_output_file), binary=False)
        self.vocab = self.word2vec_model.vocab

    def check_word(self, word, w, lists):
        for l in lists:
            if word in l:
                if w in l:
                    return True

        return False

    def filter_vocab(self, word, vocab, threshold=0.6):
        if sys.version_info.major == 3:
            types = [
             str]
        else:
            types = [
             str, unicode]
        if type(vocab) in types:
            vocab = [
             vocab]
        new_word = None
        if word not in word2vec_model.vocab:
            if word in vocab:
                index = vocab.index(word)
                l = torch.arange(len(vocab)).type(torch.LongTensor)
                return l[(l != index)]
            return torch.arange(len(vocab)).type(torch.LongTensor)
        else:
            list_one = [
             '1', 'one']
            list_two = ['two', 'twos', '2']
            list_three = ['three', '3', 'threes']
            list_four = ['4', 'four']
            list_six = ['6', 'six']
            list_an = ['1', 'an', 'a']
            lists = [
             list_one, list_two, list_three, list_four, list_six, list_an]
            filtered_vocab = []
            for i, w in enumerate(vocab):
                if w == word:
                    continue
                else:
                    if self.check_word(word, w, lists):
                        continue
                    else:
                        if not word.startswith(w):
                            if w.startswith(word):
                                continue
                if w not in word2vec_model.vocab:
                    filtered_vocab.append(i)
                elif self.word2vec_model.similarity(w, word) < threshold:
                    filtered_vocab.append(i)

            return torch.LongTensor(filtered_vocab)


word2vec_model = Word2VecSimilarity('/home/cvpr/abijith/glove.6B.200d.txt')
# okay decompiling __pycache__/utils_translate.cpython-36.pyc
