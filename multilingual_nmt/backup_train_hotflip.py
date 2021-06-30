# encoding: utf-8
from __future__ import unicode_literals, print_function

import json
import os
import io
import itertools
import numpy as np
import random
from time import time
import torch
from torch import nn
import pickle
import shutil
import math
import six
from torch.autograd import Variable

import evaluator
from backup_models import MultiTaskNMT, Transformer, Shaped, LangShare
from exp_moving_avg import ExponentialMovingAverage
import optimizer as optim
from torchtext import data
import utils
from config import get_train_args
from fp16_utils import FP16_Optimizer, FP16_Module
import preprocess

def min_grad_method_multiple(model, in_arrays, k=4, device='cpu'):
    '''
        Implements the mid-grad method to find the most vulnarable
        position in the input sentence by using the gradient of the
        loss function wrt the word embeddings.

        model    := NMTModel
        data     := batch of size 1 containing src, src_lengths, tgt
        critiria := the loss function

        Returns:
            Calculated vulnarable position.
    '''

    model.zero_grad()
    for p in model.parameters():
        p.requires_grad = True

    model.weights = None
    ex_block = model.make_input_embedding(model.embed_word,
                                         in_arrays[0], weights=None,is_source=True)
    ex_block.retain_grad()
    loss, stat = model(*in_arrays, ex_block=ex_block)
    loss.backward()

    grad = ex_block.grad.squeeze()    
    # Need not replace full-stop. Thus ignoring the last norm value.
    if k == -1:
        k = grad.shape[0] - 1
    index = grad.norm(dim=1)[:-1].topk(k=k,largest=False)[1]
    index = index[index != 0]
    model.zero_grad()

    # TODO XXX CHANGE back
    # model.eval()
    for p in model.parameters():
        p.requires_grad = False
    #print(index,grad.norm(dim=1)[:-1],grad.shape[0])
    return (index - grad.shape[0] + 1).cpu().numpy()

def hotflip(model, in_arrays, index, vocab, device='cpu'):
    '''
        Implements the mid-grad method to find the most vulnarable
        position in the input sentence by using the gradient of the
        loss function wrt the word embeddings.

        model    := NMTModel
        data     := batch of size 1 containing src, src_lengths, tgt
        critiria := the loss function

        Returns:
            Calculated vulnarable position.
    '''

    model.zero_grad()
    for p in model.parameters():
        p.requires_grad = True
    model.weights = None
    ex_block = model.make_input_embedding(model.embed_word,
                                         in_arrays[0], weights=None,is_source=True)
    ex_block.retain_grad()
    loss, stat = model(*in_arrays, ex_block=ex_block)
    loss.backward()

    grad = ex_block.grad.squeeze()
    result = torch.matmul(model.embed_word.weight[vocab],grad[index])
    req_word = vocab[torch.argmin(result)]
    return req_word


def init_weights(m):
    if type(m) == nn.Linear:
        input_dim = m.weight.size(1)
        # LeCun Initialization
        m.weight.data.uniform_(-math.sqrt(3.0 / input_dim),
                                math.sqrt(3.0 / input_dim))
        # My custom initialization
        # m.weight.data.uniform_(-3. / input_dim, 3. / input_dim)

        # Xavier Initialization
        # output_dim = m.weight.size(0)
        # m.weight.data.uniform_(-math.sqrt(6.0 / (input_dim + output_dim)),
        #                         math.sqrt(6.0 / (input_dim + output_dim)))

        if m.bias is not None:
            m.bias.data.fill_(0.)


def save_checkpoint(state, is_best, model_path_, best_model_path_):
    torch.save(state, model_path_)
    if is_best:
        shutil.copyfile(model_path_, best_model_path_)


global max_src_in_batch, max_tgt_in_batch


def batch_size_fn(new, count, sofar):
    global max_src_in_batch, max_tgt_in_batch
    if count == 1:
        max_src_in_batch = 0
        max_tgt_in_batch = 0
    max_src_in_batch = max(max_src_in_batch, len(new[0]) + 2)
    max_tgt_in_batch = max(max_tgt_in_batch, len(new[1]) + 1)
    src_elements = count * max_src_in_batch
    tgt_elements = count * max_tgt_in_batch
    return max(src_elements, tgt_elements)


def save_output(hypotheses, vocab, outf):
    # Save the Hypothesis to output file
    with io.open(outf, 'w') as fp:
        for sent in hypotheses:
            words = [vocab[y] for y in sent]
            fp.write(' '.join(words) + '\n')


def tally_parameters(model):
    n_params = sum([p.nelement() for p in model.parameters()])
    print('* number of parameters: %d' % n_params)
    enc = 0
    dec = 0
    for name, param in model.named_parameters():
        if 'encoder' in name:
            enc += param.nelement()
        elif 'decoder' in name:
            dec += param.nelement()
    print('encoder: ', enc)
    print('decoder: ', dec)


def report_func(epoch, batch, num_batches, start_time, report_stats,
                report_every):
    """
    This is the user-defined batch-level training progress
    report function.
    Args:
        epoch(int): current epoch count.
        batch(int): current batch count.
        num_batches(int): total number of batches.
        start_time(float): last report time.
        lr(float): current learning rate.
        report_stats(Statistics): old Statistics instance.
    Returns:
        report_stats(Statistics): updated Statistics instance.
    """
    if batch % report_every == -1 % report_every:
        report_stats.output(epoch, batch + 1, num_batches, start_time)
        report_stats = utils.Statistics()

    return report_stats


class CalculateBleu(object):
    def __init__(self, model, test_data, key, batch=50, max_decode_len=50,
                 beam_size=1, alpha=0.6, max_sent=None):
        self.model = model
        self.test_data = test_data
        self.key = key
        self.batch = batch
        self.device = -1
        self.max_decode_length = max_decode_len
        self.beam_size = beam_size
        self.alpha = alpha
        self.max_sent = max_sent

    def __call__(self):
        self.model.eval()
        references = []
        hypotheses = []
        for i in range(0, len(self.test_data), self.batch):
            sources, targets = zip(*self.test_data[i:i + self.batch])
            references.extend(t.tolist() for t in targets)
            x_block = utils.source_pad_concat_convert(sources,
                                                      device=None)
            x_block = Variable(torch.LongTensor(x_block).type(utils.LONG_TYPE),
                               requires_grad=False)
            ys = self.model.translate(x_block,
                                      self.max_decode_length,
                                      beam=self.beam_size,
                                      alpha=self.alpha)
            hypotheses.extend(ys)
            if self.max_sent is not None and \
                    ((i + 1) > self.max_sent):
                break

            # Log Progress
            if self.max_sent is not None:
                den = self.max_sent
            else:
                den = len(self.test_data)
            print("> Completed: [ %d / %d ]" % (i, den), end='\r')

        bleu = evaluator.BLEUEvaluator().evaluate(references, hypotheses)
        print('BLEU:', bleu.score_str())
        print('')
        return bleu.bleu, hypotheses


def main():
    best_score = 0
    args = get_train_args()
    print(json.dumps(args.__dict__, indent=4))

    # Set seed value
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    if args.gpu:
        torch.cuda.manual_seed_all(args.seed)

    # Reading the int indexed text dataset
    # train_data = np.load(os.path.join(args.input, args.data + ".train.npy"), allow_pickle=True)
    # train_data = train_data.tolist()
    # dev_data = np.load(os.path.join(args.input, args.data + ".valid.npy"), allow_pickle=True)
    # dev_data = dev_data.tolist()
    # test_data = np.load(os.path.join(args.input, args.data + ".test.npy"), allow_pickle=True)
    # test_data = test_data.tolist()

    # Reading the vocab file
    with open(os.path.join(args.input, args.data + '.vocab.pickle'), 'rb') as f:
        id2w = pickle.load(f)

    w2id = {word: index for index, word in id2w.items()}
    
    # load the required pruned vocab here
    #vocab = [i    for i in id2w]
    _, org_vocab = torch.load(os.path.join(args.input, 'good_words_in_vocab.pt'))
    source_path = args.src
    target_path = args.pred
    args.tok = False
    args.max_seq_len = 70

    # Train Dataset
    source_data = preprocess.make_dataset(source_path, w2id, args.tok)
    target_data = preprocess.make_dataset(target_path, w2id, args.tok)
    assert len(source_data) == len(target_data)
    train_data = [(s, t) for s, t in six.moves.zip(source_data, target_data)
                  if 0 < len(s) < args.max_seq_len
                  and 0 < len(t) < args.max_seq_len]

    print('Size of data for attack:', len(train_data))
    args.id2w = id2w
    args.n_vocab = len(id2w)

    # Define Model
    model = eval(args.model)(args)
    model.apply(init_weights)
    
    tally_parameters(model)
    if args.gpu >= 0:
        model.cuda(args.gpu)
    #print(model)

    optimizer = optim.TransformerAdamTrainer(model, args)
    ema = ExponentialMovingAverage(decay=0.999)
    ema.register(model.state_dict())

    if args.fp16:
        model = FP16_Module(model)
        optimizer = FP16_Optimizer(optimizer,
                                   static_loss_scale=args.static_loss_scale,
                                   dynamic_loss_scale=args.dynamic_loss_scale,
                                   dynamic_loss_args={'init_scale': 2 ** 16},
                                   verbose=False)

    checkpoint = torch.load(args.best_model_file)
    print("=> loaded checkpoint '{}' (epoch {}, best score {})".
          format(args.best_model_file,
                 checkpoint['epoch'],
                 checkpoint['best_score']))

    state_dict = checkpoint['state_dict']
    if args.label_smoothing == 0:
        weight = torch.ones(args.n_vocab)
        weight[model.padding_idx] = 0
        state_dict['criterion.weight'] = weight
        state_dict.pop('one_hot')
    model.load_state_dict(state_dict)

    # put the model in train mode. But change the requires_grad var to False
    # for all parameters except weights
    for param in model.parameters():
        param.requires_grad = False
    model.eval()

    # optionally resume from a checkpoint
    # if args.resume:
    #     if os.path.isfile(args.model_file):
    #         print("=> loading checkpoint '{}'".format(args.model_file))
    #         checkpoint = torch.load(args.model_file)
    #         args.start_epoch = checkpoint['epoch']
    #         best_score = checkpoint['best_score']
    #         model.load_state_dict(checkpoint['state_dict'])
    #         optimizer.load_state_dict(checkpoint['optimizer'])
    #         print("=> loaded checkpoint '{}' (epoch {})".
    #               format(args.model_file, checkpoint['epoch']))
    #     else:
    #         print("=> no checkpoint found at '{}'".format(args.model_file))

    # train_data = dev_data
    src_data, trg_data = list(zip(*train_data))
    total_src_words = len(list(itertools.chain.from_iterable(src_data)))
    total_trg_words = len(list(itertools.chain.from_iterable(trg_data)))
    iter_per_epoch = (total_src_words + total_trg_words) // (2 * args.wbatchsize)
    print('Approximate number of iter/epoch =', iter_per_epoch)
    time_s = time()

    num_grad_steps = 0
    n_vocab = len(org_vocab)
    for epoch in range(1):
        train_iter = data.iterator.pool(train_data,
                                        args.batchsize,
                                        #args.wbatchsize,
                                        key=lambda x: (len(x[0]), len(x[1])))

        # def batch_iterator(train_iter):
        #     i = 0
        #     for train_batch in train_iter:
        #         for batch in train_batch:
        #             yield i, [batch]
        #             i += 1

        index = -2
        output_file = args.out_file
        output_fp = io.open(output_file, 'w', 1)
        for num_steps, train_batch in enumerate(train_iter):
        # for num_steps, train_batch in batch_iterator(train_iter):
            assert(len(train_batch[0][0])<=40)
            assert(len(train_batch[0][0])>5)
            if len(train_batch[0][0])>40:
                continue
            # print('>>>>>', len(list(train_batch)))
            in_arrays = utils.seq2seq_pad_concat_convert(train_batch, -1)
            print(len(in_arrays), in_arrays[0].shape, in_arrays[1].shape, in_arrays[2].shape)
            
            # loss, stat = model(*in_arrays, index=-2)

            model.weights = None
            loss, stat = model(*in_arrays)
            print('original loss:', loss.item())
            org_loss = loss.item()

            # resetting weights paramteter for eah sentence
            # model.weights = torch.nn.Parameter(torch.ones(args.n_vocab).cuda(), requires_grad=True)
            #model.weights = torch.nn.Parameter(torch.ones(len(vocab)).cuda(), requires_grad=True)
            #params = filter(lambda p: p.requires_grad, model.parameters())
            #optimizer = torch.optim.Adam(params,
            #                              lr=args.learning_rate,
            #                              betas=(args.optimizer_adam_beta1,
            #                                     args.optimizer_adam_beta2),
            #                              eps=args.optimizer_adam_epsilon)
            # optimizer = optim.TransformerAdamTrainer(model, args)
            #optimizer.zero_grad()
            min_loss = 100
            #zeros = 3*np.ones(len(train_batch[0][0])).astype(np.int)
            #train_batch[0] = (zeros, train_batch[0][1])
            vocab = list(set(org_vocab).difference(in_arrays[0].squeeze().tolist()))
            print(len(vocab),len(org_vocab))
            
            #rand_sent = np.random.choice(vocab,size=len(train_batch[0][0]))
            #train_batch[0] = (rand_sent, train_batch[0][1])
            #random.shuffle(org_vocab)
            #vocab = org_vocab
            index_changed = []
            for iter in range(5):
                flag = False

                index_visited = []
                while True:
                    in_arrays = utils.seq2seq_pad_concat_convert(train_batch, -1)
                    index_order = min_grad_method_multiple(model, in_arrays, k=-1)
                    tmp_flag = False
                    for index in index_order:
                        if index not in index_visited:
                            index_visited.append(index)
                            tmp_flag = True
                            break
                    index = index_visited[-1]
                    if tmp_flag == False:
                        break
                    model.weights = None
                    proj_loss,stat = model(*in_arrays)
                    print("Proj loss:",proj_loss.item())
                    old_word = train_batch[0][0][index]
                    new_word = hotflip(model, in_arrays,index=index-1, vocab=vocab)
                    train_batch[0][0][index] = new_word
                    in_arrays = utils.seq2seq_pad_concat_convert(train_batch, -1)
                    loss,stat = model(*in_arrays)
                    train_batch[0][0][index] = old_word
                    in_arrays = utils.seq2seq_pad_concat_convert(train_batch, -1)
                    proj_loss,stat = model(*in_arrays)
                    do_replace = True
                    if proj_loss < loss and index in index_changed:
                        do_replace = False
                    if min_loss > loss and new_word!=old_word and do_replace:
                        flag = True
                        min_loss = max(loss.item(), org_loss)
                        if index not in index_changed:
                            index_changed.append(index)
                        print(index_changed)
                        #min_loss = loss.item()
                        src = train_batch[0][0]
                        src[index] = new_word

                    train_batch[0] = (src,train_batch[0][1])
                    #print('len_src:', len(src), in_arrays[0].shape)
                    print('loss({})\tmin_loss({})\tindex({})'.format(loss.item(), min_loss, index))
                if flag == False:
                    break
            #new_word = model.weights.argmax()
            #src = train_batch[0][0]
            #src[index] = new_word
            src_text = ' '.join([id2w[i]    for i in src]) + '\n'
            output_fp.write(src_text)
            print(train_batch, src_text)
        output_fp.close()
            
if __name__ == '__main__':
    main()
