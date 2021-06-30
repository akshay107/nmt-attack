# uncompyle6 version 3.3.4
# Python bytecode 3.6 (3379)
# Decompiled from: Python 3.6.3 |Anaconda, Inc.| (default, Nov 20 2017, 20:41:42) 
# [GCC 7.2.0]
# Embedded file name: /home/cvpr/abijith/OpenNMT-py_old/new_attacks.py
# Compiled at: 2019-06-17 11:18:46
# Size of source mod 2**32: 15555 bytes
import torch, torch.nn as nn
from torch.nn import Parameter
import numpy as np
from onmt.utils.logging import logger

def sample(values, how='min'):
    index = values.argmin().item()
    loss = values[index]
    if how == 'prob':
        p = torch.nn.functional.softmax(values, -1).cpu().numpy()
        index = np.random.choice((values.shape[0]), p=p)
        loss = values[index]
    return (
     index, loss)


class FullyTargetedAttacker:

    def __init__(self, model, critiria, optimizer, data, src_vocab, outfile, max_iterations=1000, max_count=10, threshold=0.9):
        self.model = model
        self.model.zero_grad()
        self.critiria = critiria
        self.optimizer = optimizer
        self.data = data
        self.src_vocab = src_vocab
        self.outfile = outfile
        self.max_iterations = max_iterations
        self.max_count = max_count
        self.threshold = threshold

    def run(self):
        src, src_length = self.data.src
        tgt = self.data.tgt
        intervel = 100
        len_vocab = len(self.src_vocab)
        n_vocab = len_vocab
        s_size = list(src.shape)
        s_size[1] = n_vocab
        modified_src_lengths = torch.zeros(n_vocab, dtype=(torch.long)).to(src_length.device) + src_length[0].item()
        ttt = torch.zeros(s_size).type(src.type()).to(src.device)
        ttt_tgt = torch.zeros(tgt.shape[0], n_vocab, tgt.shape[2]).type(tgt.type()).to(tgt.device)
        for i in range(n_vocab):
            ttt_tgt[:, i, :].copy_(tgt.squeeze(2))

        max_iterations = 5
        '''
        if self.model.opt.init_zeros:
            full_stop = src[(-1)]
            src = torch.zeros_like(src)
            src[-1] = full_stop
            src = src.to(tgt.device)
            self.data.src = (src, src_length)
            max_iterations = src.shape[0]
        '''
        min_loss = 100
        iter = 0
        output, attn = self.model.model(src, tgt, src_length)
        org_loss, _ = self.critiria(self.data, output, attn)
        org_loss = org_loss.mean()
        org_loss = org_loss.item()
        print('org_loss:', org_loss)
        for iter in range(max_iterations):
            flag = False
            indices = []
            while len(indices) != len(src):
                self.data.src = (
                 src, src_length)
                self.data.tgt = tgt
                for ind in self.attr_min_grad((self.data), k=(-1)):
                    if ind not in indices:
                        indices.append(ind)
                        break

                j = indices[(-1)]
                self.model.ae.re_init((src[j].item()), j, device=(src.device))
                mask = self.model.ae.mask
                n_vocab = mask.shape[0]
                for i in range(n_vocab):
                    ttt[:, i, :].copy_(src.squeeze(2))

                ttt[j, :n_vocab, 0] = mask
                modified_embedding = self.model.embedding_weight[ttt.squeeze(2)]
                bag_output = []
                bag_attn = []
                bag_loss = []
                for start in range(0, n_vocab, intervel):
                    end = min(start + intervel, n_vocab)
                    self.data.src = (ttt[:, start:end, :], modified_src_lengths[start:end])
                    self.data.tgt = ttt_tgt[:, start:end, :]
                    output, attn = self.model.model(modified_embedding[:, start:end, :], ttt_tgt[:, start:end, :], modified_src_lengths[start:end])
                    loss, _ = self.critiria(self.data, output, attn)
                    loss = loss.view(-1, output.shape[1], 1)
                    bag_loss.append(loss)

                loss = torch.cat(bag_loss, dim=1)
                loss_for_each_word = loss.mean(dim=0).mean(dim=1)
                sample_index, loss = sample(loss_for_each_word, how='min')
                new_index = mask[sample_index]
                loss = loss.item()
                tmp_index = src[j].item()
                if min_loss > loss:
                    if tmp_index != new_index:
                        src[j] = new_index
                        min_loss = max(loss, org_loss)
                        flag = True
                print('index:', j, 'loss:', loss, 'min_loss:', min_loss)

            if flag == False:
                break

        adv_src = src
        adv_src_text = ' '.join([self.src_vocab[j] for j in adv_src.squeeze()]) + '\n'
        self.outfile.write(adv_src_text)
        logger.info('MIN_LOSS({})\tMAX_ITERATIONS({})'.format(min_loss, iter))
        logger.info('SENT: {}\n'.format(adv_src_text))


class LinearCombinerHotFlip(nn.Module):

    def __init__(self, model, src, vocab, index, device='cpu'):
        super(LinearCombinerHotFlip, self).__init__()
        self.vocab = vocab
        self.index = index
        self.embed_weights = model.encoder.embeddings.word_lut.weight.data
        one_hot = []
        for i in src.squeeze():
            tmp_embed = [
             0] * len(vocab)
            tmp_embed[i] = 1
            one_hot.append(tmp_embed)

        one_hot = np.array(one_hot)
        self.coefficient = torch.from_numpy(one_hot).type(torch.FloatTensor).to(device)
        self.coefficient.requires_grad = True
        self.register_parameter('coeff', Parameter(self.coefficient))
        self._good_vocab = [self.vocab.index(i) for i in torch.load('good_vocab_de_en.pt')[0]]
        self.good_vocab = sorted(list(set(self._good_vocab).difference(src.squeeze().tolist())))
        print(len(self._good_vocab), len(src), len(self.good_vocab))

    def re_init(self, src, index, device='cpu'):
        self.index = index
        one_hot = []
        for i in src.squeeze():
            tmp_embed = [
             0] * len(self.vocab)
            tmp_embed[i] = 1
            one_hot.append(tmp_embed)

        one_hot = np.array(one_hot)
        self.coefficient = torch.from_numpy(one_hot).type(torch.FloatTensor).to(device)
        self.coefficient.requires_grad = True
        self.coeff = nn.Parameter(self.coefficient)

    def forward(self, src, tgt, src_lengths, to_translate=False):
        sentence_embedding = torch.matmul(self.coeff, self.embed_weights)
        return (
         sentence_embedding.unsqueeze(1), tgt, src_lengths)


class HotFlipAttacker:
    r"""'\n        HotFlip based attack. This is an reduced version of original HotFlip\n        model which uses only substitution operation.\n    '"""

    def __init__(self, model, critiria, optimizer, data, src_vocab, outfile, max_iterations=1000, max_count=10, threshold=0.9):
        self.model = model
        self.model.zero_grad()
        self.critiria = critiria
        self.optimizer = optimizer
        self.data = data
        self.src_vocab = src_vocab
        self.outfile = outfile
        self.max_iterations = max_iterations
        self.max_count = max_count
        self.threshold = threshold

    def run(self, return_index=False):
        src, src_length = self.data.src
        tgt = self.data.tgt
        output, attn = self.model(src, tgt, src_length)
        loss, _ = self.critiria(self.data, output, attn)
        loss.backward()
        grad = self.model.ae.coeff.grad
        min_flips = []
        for i in range(src.shape[0]):
            processed_grad = grad[i] - grad[i][src[i].item()]
            sorted_index = processed_grad.topk(k=(processed_grad.shape[0]), largest=False)[1]
            new_word = None
            ref_word = self.src_vocab[src[i].item()]
            if ref_word in word2vec_model.vocab:
                for wi in sorted_index:
                    if word2vec_model.filter_vocab(ref_word, [self.src_vocab[wi]]).shape[0] != 0:
                        new_word = wi
                        min_flips.append((processed_grad[wi], wi))
                        break

            elif sorted_index[0] == src[i].item():
                new_word = sorted_index[1]
                min_flips.append((processed_grad[1], new_word))
            else:
                new_word = sorted_index[0]
                min_flips.append((processed_grad[0], new_word))

        if self.model.ae.index is None:
            min_grad_word = min(min_flips, key=(lambda x: x[0]))
            min_grad_index = min_flips.index(min_grad_word)
            new_word = min_grad_word[1]
        else:
            min_grad_index = self.model.ae.index
            new_word = min_flips[min_grad_index][1]
        if return_index:
            return new_word
        adv_src = src
        adv_src[min_grad_index] = new_word
        adv_src_text = ' '.join([self.src_vocab[j] for j in adv_src.squeeze()]) + '\n'
        self.outfile.write(adv_src_text)
        logger.info('MIN_GRAD({})\tSELECTED_INDEX({})'.format(min_flips[min_grad_index][0], new_word))
        logger.info('SENT: {}\n'.format(adv_src_text))


class PosHotflipAttacker:

    def __init__(self, model, critiria, optimizer, data, src_vocab, outfile, max_iterations=1000, max_count=10, threshold=0.9):
        self.model = model
        self.model.zero_grad()
        self.critiria = critiria
        self.optimizer = optimizer
        self.data = data
        self.src_vocab = src_vocab
        self.outfile = outfile
        self.max_iterations = max_iterations
        self.max_count = max_count
        self.threshold = threshold

    def run(self, return_index=False):
        src, src_length = self.data.src
        tgt = self.data.tgt
        src_embedding = self.model.embedding_weight[src.squeeze(1)]
        ground_truth = self.model.model.encoder(src, src_length)
        target = ground_truth[1][self.model.ae.index:]
        _, output, _ = self.model.encoder(src, src_length)
        loss = self.critiria(output[self.model.ae.index:], target)
        loss.backward()
        model_wrapper = self.model.ae
        grad = model_wrapper.weights.grad
        index = model_wrapper.index
        sorted_index = grad.topk(k=(grad.shape[0]), largest=False)[1]
        new_word = None
        ref_word = self.src_vocab[src[index].item()]
        if ref_word in word2vec_model.vocab:
            for wi in sorted_index:
                if word2vec_model.filter_vocab(ref_word, [self.src_vocab[wi]]).shape[0] != 0:
                    new_word = wi
                    break

        else:
            if sorted_index[0] == src[index].item():
                new_word = sorted_index[1]
            else:
                new_word = sorted_index[0]
        adv_src = src
        adv_src[index] = model_wrapper.mask[new_word]
        adv_src_text = ' '.join([self.src_vocab[j] for j in adv_src.squeeze()]) + '\n'
        self.outfile.write(adv_src_text)
        logger.info('GRAD({})\tSELECTED_INDEX({})'.format(grad[new_word], new_word))
        logger.info('SENT: {}\n'.format(adv_src_text))


class FTHotFlipAttacker:
    def __init__(self, model, critiria, optimizer, data, src_vocab, outfile, max_iterations=1000, max_count=10, threshold=0.9):
        self.model = model
        self.model.zero_grad()
        self.critiria = critiria
        self.optimizer = optimizer
        self.data = data
        self.src_vocab = src_vocab
        self.outfile = outfile
        self.max_iterations = max_iterations
        self.max_count = max_count
        self.threshold = threshold

    def run(self, return_index=False):
        src, src_length = self.data.src
        tgt = self.data.tgt
        output, attn = self.model.model(src, tgt, src_length)
        org_loss, _ = self.critiria(self.data, output, attn)
        org_loss = org_loss.item()
        print('org_loss:', org_loss)
        prev_loss = org_loss
        max_iterations = 5
        index_changed = []
        min_loss = 100
        iter = 0
        adv_src_text = ' '.join([self.src_vocab[j] for j in src.squeeze()]) + '\n'
        logger.info('ORG_SENT: {}\n'.format(adv_src_text))
        for iter in range(max_iterations):
            flag = False
            indices = []
            while len(indices) != len(src):
                for ind in self.attr_min_grad((self.data), k=(-1)):
                    if ind not in indices:
                        indices.append(ind)
                        break

                j = indices[(-1)]
                self.data.src = (
                 src, src_length)
                self.model.ae.re_init(src, j, device=(src.device))
                tmp_index = src[j].item()
                output, attn = self.model.model(src, tgt, src_length)
                proj_loss, _ = self.critiria(self.data, output, attn)
                output, attn = self.model(src, tgt, src_length)
                loss, _ = self.critiria(self.data, output, attn)
                loss.backward()
                grad = self.model.ae.coeff.grad[self.model.ae.index][self.model.ae.good_vocab]
                new_word = self.model.ae.good_vocab[grad.argmin()]
                src[j] = new_word
                output, attn = self.model.model(src, tgt, src_length)
                loss, _ = self.critiria(self.data, output, attn)
                do_replace = True
                if proj_loss < loss:
                    if j in index_changed:
                        do_replace = False
                if min_loss > loss and new_word != tmp_index and do_replace:
                    flag = True
                    if j not in index_changed:
                        index_changed.append(j)
                    min_loss = max(loss.item(), org_loss)
                else:
                    src[j] = tmp_index
                print(j, 'loss({}) proj_loss({}) min_loss({}) org_loss({})'.format(loss.item(), proj_loss.item(), min_loss, org_loss))

            if flag == False:
                break

        adv_src = src
        adv_src_text = ' '.join([self.src_vocab[j] for j in adv_src.squeeze()]) + '\n'
        self.outfile.write(adv_src_text)
        print(index_changed)
        logger.info('SENT: {}\n'.format(adv_src_text))
# okay decompiling __pycache__/new_attacks.cpython-36.pyc
