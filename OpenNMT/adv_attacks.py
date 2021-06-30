# uncompyle6 version 3.3.4
# Python bytecode 3.6 (3379)
# Decompiled from: Python 3.6.3 |Anaconda, Inc.| (default, Nov 20 2017, 20:41:42) 
# [GCC 7.2.0]
# Embedded file name: /home/cvpr/abijith/OpenNMT-py_old/adv_attacks.py
# Compiled at: 2019-06-25 10:49:12
# Size of source mod 2**32: 52035 bytes
import sys, torch, torch.nn as nn
from torch.nn import Parameter
from itertools import repeat
from onmt.utils.logging import logger
from onmt.utils.misc import use_gpu
from onmt.utils.misc import split_corpus
from onmt.modules.util_class import Cast
import onmt.inputters as inputters
from onmt.model_builder import build_base_model
from onmt.utils.parse import ArgumentParser
from onmt.utils.optimizers import build_torch_optimizer
from onmt.utils.loss import build_loss_compute
from new_attacks import LinearCombinerHotFlip, HotFlipAttacker, PosHotflipAttacker
import numpy as np
from collections import defaultdict
tmp_file_name = 'good_vocab_per_sent_random.pt'
main_vocab_file = 'good_vocab_de_en.pt'

def min_grad_method(model, data, tgt_vocab, opt, device='cpu'):
    """
        Implements the mid-grad method to find the most vulnarable
        position in the input sentence by using the gradient of the
        loss function wrt the word embeddings.

        model    := NMTModel
        data     := batch of size 1 containing src, src_lengths, tgt
        critiria := the loss function

        Returns:
            Calculated vulnarable position.
    """
    src, src_lengths = data.src
    tgt = data.tgt
    critiria = build_loss_compute(model, tgt_vocab, opt,
      train=True)
    model.zero_grad()
    for p in model.parameters():
        p.requires_grad = True

    embeding = []
    for i in src.squeeze():
        ttt = model.encoder.embeddings.word_lut.weight[i]
        tmp_embed = torch.zeros_like(ttt)
        tmp_embed.copy_(ttt)
        embeding.append(tmp_embed)

    embeding = torch.stack(embeding).unsqueeze(1)
    embeding.retain_grad()
    output, attn = model(embeding, tgt, src_lengths)
    loss, _ = critiria(data, output, attn)
    loss.backward()
    grad = embeding.grad.squeeze()
    index = grad.norm(dim=1)[:-1].argmin().item()
    model.zero_grad()
    for p in model.parameters():
        p.requires_grad = False

    return index - len(src)


def min_grad_method_multiple(model, data, tgt_vocab, opt, k=4, device='cpu'):
    """
        Implements the mid-grad method to find the most vulnarable
        position in the input sentence by using the gradient of the
        loss function wrt the word embeddings.

        model    := NMTModel
        data     := batch of size 1 containing src, src_lengths, tgt
        critiria := the loss function

        Returns:
            Calculated vulnarable position.
    """
    src, src_lengths = data.src
    tgt = data.tgt
    critiria = build_loss_compute(model, tgt_vocab, opt,
      train=True)
    model.zero_grad()
    for p in model.parameters():
        p.requires_grad = True

    embeding = []
    for i in src.squeeze():
        ttt = model.encoder.embeddings.word_lut.weight[i]
        tmp_embed = torch.zeros_like(ttt)
        tmp_embed.copy_(ttt)
        embeding.append(tmp_embed)

    embeding = torch.stack(embeding).unsqueeze(1)
    embeding.retain_grad()
    output, attn = model(embeding, tgt, src_lengths)
    loss, _ = critiria(data, output, attn)
    loss.backward()
    grad = embeding.grad.squeeze()
    if k == -1:
        k = len(src)
    index = grad.norm(dim=1).topk(k=k, largest=False)[1]
    model.zero_grad()
    for p in model.parameters():
        p.requires_grad = False

    return (index - len(src)).cpu().numpy()


def make_min_grad(model, tgt_vocab, opt, device):

    def _min_grad_method_multiple(data, k=4):
        return min_grad_method_multiple(model, data, tgt_vocab, opt, k=k, device=device)

    return _min_grad_method_multiple


def make_rand_pos(data, k=4):
    length = data.src[0].shape[0]
    if k == -1:
        k = length
    np.random.seed(0)
    k = min(k, length)
    for i in np.random.permutation(length)[:k]:
        yield i


def save_loss_value(loss, filename=None):
    if not filename:
        with open('/tmp/filename.txt') as (f):
            filename = f.readline().strip()
    try:
        data = torch.load(filename)
    except:
        data = []

    data.append(loss)
    torch.save(data, filename)


def build_attacker(model, opt, fields, data, outfile, device='cpu'):
    """
        position(int) := position from the end of the sentence.
                         this is done so that its in line with our experiments
    """
    position = opt.position
    src_vocab = fields['src'].base_field
    tgt_vocab = fields['tgt'].base_field
    if opt.position == 'min_grad':
        position = min_grad_method(model, data, tgt_vocab, opt, device)
    else:
        if opt.position == 'random':
            position = np.random.permutation(data.src[0].shape[0] - 1)[0]
    if opt.attack_type == 'multisoftmax':
        if opt.position != 'min_grad':
            position = np.array([position])
        else:
            position = min_grad_method_multiple(model, data, tgt_vocab, opt, k=(-1), device=device)
        word_index = data.src[0][position]
        input_generator = MultiLinearCombiner(model, replace_word_index=word_index,
          index=position,
          vocab=(src_vocab.vocab.itos),
          device=device)
    else:
        if opt.attack_type in ('fthotflip', 'hotflip'):
            src, _ = data.src
            if opt.position != 'min_grad':
                position = None
            input_generator = LinearCombinerHotFlip(model, src, (src_vocab.vocab.itos), position,
              device=device)
        else:
            word_index = data.src[0][position].item()
            input_generator = LinearCombiner(model, replace_word_index=word_index,
              index=position,
              vocab=(src_vocab.vocab.itos),
              device=device,
              do_filter=True,
              src=(data.src[0]))
    attack_model = AdversarialExploit(input_generator, model, opt)
    for module in attack_model.modules():
        if isinstance(module, nn.Dropout):
            module.training = False
        else:
            if isinstance(module, nn.LSTM):
                module.dropout = 0

    if opt.attack_type in ('multisoftmax', 'fullytargeted', 'softmax', 'ftsoftmax',
                           'hotflip', 'fthotflip'):
        critiria = build_loss_compute(attack_model, (fields['tgt'].base_field),
          opt, train=False, reduce=(opt.attack_type in ('multisoftmax', 'ftsoftmax', 'softmax',
                                                   'hotflip', 'fthotflip')))
    else:
        if opt.attack_type in ('enc_softmax', 'pos_hotflip'):
            critiria = nn.MSELoss()
        else:
            if opt.attack_type == 'klattack':
                gen_func = nn.Softmax(dim=(-1))
                generator = nn.Sequential(nn.Linear(opt.dec_rnn_size, len(fields['tgt'].base_field.vocab)), Cast(torch.float32), gen_func)
                model.generator = generator
                for p in generator.parameters():
                    p.requires_grad = False

                critiria = nn.KLDivLoss()
            else:
                if opt.attack_type in ('bruteforce', ):
                    if torch.__version__.startswith('0.4'):
                        critiria = nn.MSELoss(reduce=False)
                    else:
                        critiria = nn.MSELoss(reduction='none')
    optimizer = build_torch_optimizer(attack_model, opt)
    args = [
     attack_model, critiria, optimizer, data,
     src_vocab.vocab.itos, outfile]
    attacker = (str2attacker[opt.attack_type])(*args)
    if opt.position == 'random':
        attacker.attr_min_grad = make_rand_pos
    else:
        attacker.attr_min_grad = make_min_grad(model, tgt_vocab, opt, device)
    return attacker


def load_attack_model(opt, model_path=None):
    if model_path is None:
        model_path = opt.models[0]
    checkpoint = torch.load(model_path, map_location=(lambda storage, loc: storage))
    model_opt = ArgumentParser.ckpt_model_opts(checkpoint['opt'])
    ArgumentParser.update_model_opts(model_opt)
    ArgumentParser.validate_model_opts(model_opt)
    ArgumentParser.validate_and_update_attack_opts(model_opt, opt)
    vocab = checkpoint['vocab']
    if inputters.old_style_vocab(vocab):
        fields = inputters.load_old_vocab(vocab,
          (opt.data_type), dynamic_dict=(model_opt.copy_attn))
    else:
        fields = vocab
    model = build_base_model(model_opt, fields, use_gpu(opt), checkpoint, opt.gpu)
    if opt.fp32:
        model.float()
    for param in model.parameters():
        param.requires_grad = False

    return (
     fields, model, model_opt)


def run_attacker(model, src, tgt, opt, fields):
    use_cuda = opt.gpu > -1
    dev = torch.device('cuda', opt.gpu) if use_cuda else torch.device('cpu')
    src_shards = src
    tgt_shards = tgt
    src_reader = inputters.str2reader[opt.data_type].from_opt(opt)
    tgt_reader = inputters.str2reader['text'].from_opt(opt)
    data = inputters.Dataset(fields,
      readers=([src_reader, tgt_reader] if tgt_shards else [src_reader]),
      data=([('src', src_shards), ('tgt', tgt_shards)] if tgt_shards else [
     (
      'src', src_shards)]),
      dirs=([opt.src_dir, None] if tgt_shards else [opt.src_dir]),
      sort_key=(inputters.str2sortkey['text']))
    data_iter = inputters.OrderedIterator(dataset=data,
      device=dev,
      batch_size=(opt.batch_size),
      train=False,
      sort=False,
      sort_within_batch=True,
      shuffle=False)
    outfile = open((opt.output_adv), 'a', buffering=1)
    count = 0
    for batch in data_iter:
        count += 1
        #if count == 101:
        #    break
        attacker = build_attacker(model, opt, fields, batch, outfile, device=dev)
        attacker.run()

    outfile.close()


class SoftmaxAttacker:
    r"""'\n        Softmax based attack. This attack makes use of information\n        from both encoder and decoder. First the postion is\n        identified using min_grad_method(). Then the replacement\n        word is searched by iteratively updating the weights for\n        each words.\n\n    '"""

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

    def get_projected_loss(self, src, tgt, src_length):
        model_wrapper = self.model.ae
        logits = model_wrapper.wt_average(model_wrapper.weights)
        flip_word = model_wrapper.mask[logits.argmax()]
        tmp_word = src[model_wrapper.index].item()
        src[model_wrapper.index] = flip_word
        sentence_embedding = model_wrapper.embed_weights[src.squeeze(1)]
        output, attn = self.model(sentence_embedding, tgt, src_length)
        loss, _ = self.critiria(self.data, output, attn)
        src[model_wrapper.index] = tmp_word
        return loss.item()

    def run(self, return_index=False):
        count = 0
        cont_value = -1
        src, src_length = self.data.src
        tgt = self.data.tgt
        self.optimizer = build_torch_optimizer(self.model, self.model.opt)
        model_wrapper = self.model.ae
        l = []
        pl = []
        i = 0
        for i in range(self.max_iterations):
            output, attn = self.model(src, tgt, src_length)
            loss, _ = self.critiria(self.data, output, attn)
            loss.backward()
            self.optimizer.step()
            logits = model_wrapper.wt_average(model_wrapper.weights)
            if logits.max() > self.threshold:
                count += 1
                if cont_value == -1:
                    cont_value = logits.argmax().item()
                else:
                    if count == self.max_count:
                        if cont_value == logits.argmax().item():
                            break
                        if cont_value != logits.argmax().item():
                            cont_value = -1
                            count = 0
            else:
                if cont_value != -1:
                    cont_value = -1
                    count = 0

        logits = model_wrapper.wt_average(model_wrapper.weights)
        flip_word = model_wrapper.mask[logits.argmax()]
        if return_index:
            return flip_word
        adv_src = src
        adv_src[model_wrapper.index] = flip_word
        adv_src_text = ' '.join([self.src_vocab[j] for j in adv_src.squeeze()]) + '\n'
        self.outfile.write(adv_src_text)
        logger.info('LOSS({})\tMAX_LOGIT({})\tSELECTED_INDEX({})'.format(loss.item(), logits.max().item(), model_wrapper.mask[logits.argmax()]))
        logger.info('SENT: {}\n'.format(adv_src_text))


class EncSoftmaxAttacker:

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
        count = 0
        cont_value = -1
        src, src_length = self.data.src
        tgt = self.data.tgt
        src_embedding = self.model.embedding_weight[src.squeeze(1)]
        ground_truth = self.model.model.encoder(src, src_length)
        target = ground_truth[1][self.model.ae.index:]
        model_wrapper = self.model.ae
        position = model_wrapper.index
        i = 0
        for i in range(self.max_iterations):
            self.model.zero_grad()
            _, output, _ = self.model.encoder(src, src_length)
            loss = self.critiria(output[position:], target)
            loss.backward()
            self.optimizer.step()
            logits = model_wrapper.wt_average(model_wrapper.weights)
            if logits.max() > self.threshold:
                count += 1
                if cont_value == -1:
                    cont_value = logits.argmax().item()
                else:
                    if count == self.max_count:
                        if cont_value == logits.argmax().item():
                            break
                        if cont_value != logits.argmax().item():
                            cont_value = -1
                            count = 0
            else:
                if cont_value != -1:
                    cont_value = -1
                    count = 0

        logits = model_wrapper.wt_average(model_wrapper.weights)
        flip_word = model_wrapper.mask[logits.argmax()]
        adv_src = src
        adv_src[model_wrapper.index] = flip_word
        adv_src_text = ' '.join([self.src_vocab[j] for j in adv_src.squeeze()]) + '\n'
        self.outfile.write(adv_src_text)
        logger.info('LOSS({})\tMAX_LOGIT({})\tSELECTED_INDEX({})\tITER({})'.format(loss.item(), logits.max().item(), model_wrapper.mask[logits.argmax()], i))
        logger.info('SENT: {}\n'.format(adv_src_text))
        critiria = nn.MSELoss(reduce=False)
        ground_truth = self.model.model.encoder(adv_src, src_length)
        output = ground_truth[1][self.model.ae.index:]
        loss = critiria(output, target)


class BruteForceAttacker:

    def __init__(self, model, critiria, optimizer, data, src_vocab, outfile, max_iterations=1000, max_count=10, threshold=0.9):
        self.model = model
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
        index = self.model.ae.index
        n_vocab = len(self.src_vocab)
        s_size = list(src.shape)
        s_size[1] = n_vocab
        modified_src_lengths = torch.zeros(n_vocab, dtype=(torch.long)) + src_length[0].item()
        modified_src = torch.zeros_like(src).type(src.type())
        modified_src.copy_(src)
        ttt = torch.zeros(s_size).type(src.type())
        for i in range(n_vocab):
            ttt[:, i, :].copy_(modified_src.squeeze(2))

        ttt[index, :, 0] = torch.arange(n_vocab)
        inp = self.model.embedding_weight[modified_src.squeeze(1)]
        ground_truth = self.model.model.encoder(inp, src_length)
        target_vec = ground_truth[1][index:]
        modified_target_vec = torch.zeros([-index, n_vocab, 500]).type(target_vec.type())
        for i in range(n_vocab):
            modified_target_vec[:, i, :] = target_vec[:, 0, :]

        modified_embedding = self.model.embedding_weight[ttt.squeeze(2)]
        bag_output = []
        intervel = 500
        for start in range(0, n_vocab, intervel):
            end = start + intervel
            _, output, _ = self.model.model.encoder(modified_embedding[:, start:end, :], modified_src_lengths[start:end])
            bag_output.append(output)

        output = torch.cat(bag_output, dim=1)
        loss = self.critiria(output[index:], modified_target_vec)
        loss_for_each_word = loss.mean(dim=0).mean(dim=1)
        _, sorted_index = loss_for_each_word.topk(k=n_vocab, largest=False)
        new_word = None
        ref_word_index = src[index].item()
        ref_word = self.src_vocab[ref_word_index]
        if ref_word in word2vec_model.vocab:
            for wi in sorted_index:
                if word2vec_model.filter_vocab(ref_word, [self.src_vocab[wi]]).shape[0] != 0:
                    new_word = wi
                    break

        else:
            if sorted_index[0] == ref_word_index:
                new_word = sorted_index[1]
            else:
                new_word = sorted_index[0]
        adv_src = src
        adv_src[index] = new_word
        adv_src_text = ' '.join([self.src_vocab[j] for j in adv_src.squeeze()]) + '\n'
        self.outfile.write(adv_src_text)
        logger.info('LOSS({})\tSELECTED_INDEX({})'.format(loss_for_each_word[new_word].item(), new_word))
        logger.info('SENT: {}\n'.format(adv_src_text))


class IterTargetedSoftmaxAttacker(SoftmaxAttacker, object):

    def __init__(self, model, critiria, optimizer, data, src_vocab, outfile, max_iterations=1000, max_count=10, threshold=0.9):
        super(IterTargetedSoftmaxAttacker, self).__init__(model, critiria, optimizer, data, src_vocab, outfile, max_iterations, max_count, threshold)

    def run(self):
        src, src_length = self.data.src
        tgt = self.data.tgt
        full_stop = src[(-1)]
        src = torch.zeros_like(src)
        src[-1] = full_stop
        src = src.to(tgt.device)
        min_loss = 100
        iter = 0
        for iter in range(20):
            flag = False
            for j in np.random.permutation(len(src)):
                self.data.src = (
                 src, src_length)
                self.model.ae.re_init((src[j].item()), j, device=(src.device))
                self.optimizer = build_torch_optimizer(self.model, self.model.opt)
                flip_word = super(IterTargetedSoftmaxAttacker, self).run(return_index=True)
                tmp_index = src[j].item()
                src[j] = flip_word
                self.data.src = (src, src_length)
                modified_embedding = self.model.embedding_weight[src.squeeze(2)]
                output, attn = self.model.model(modified_embedding, tgt, src_length)
                loss, _ = self.critiria(self.data, output, attn)
                if loss < min_loss:
                    min_loss = loss
                    flag = True
                else:
                    src[j] = tmp_index
                print('.', end='')
                logger.info('LOSS({})\tNUM_ITERATIONS({})'.format(min_loss.item(), iter))

            if flag == False:
                break

        print('')
        adv_src = src
        adv_src_text = ' '.join([self.src_vocab[j] for j in adv_src.squeeze()]) + '\n'
        self.outfile.write(adv_src_text)
        logger.info('LOSS({})\tNUM_ITERATIONS({})'.format(min_loss.item(), iter))
        logger.info('SENT: {}\n'.format(adv_src_text))


class LinearCombiner(nn.Module):

    def __init__(self, model, replace_word_index, index, vocab, device='cpu', do_filter=True, src=None):
        super(LinearCombiner, self).__init__()
        self.wt_average = nn.Softmax(dim=(-1))
        self.vocab = vocab
        self.do_filter = do_filter
        self.index = index
        self.embed_weights = model.encoder.embeddings.word_lut.weight.data.to(device)
        self._good_vocab = [self.vocab.index(i) for i in torch.load('good_vocab_de_en.pt')[0]]
        if src is not None:
            self.good_vocab = sorted(list(set(self._good_vocab).difference(src.squeeze().tolist())))
        else:
            self.good_vocab = sorted(self._good_vocab)
        print(len(self._good_vocab), len(src), len(self.good_vocab))
        self.mask = torch.LongTensor(self.good_vocab)
        self.embed_param = self.embed_weights[self.mask]
        self._weights = torch.ones(self.embed_param.shape[0]).to(device)
        self.register_parameter('weights', Parameter(self._weights))
        self.weights.requires_grad = True
        self.weights.to(device)

    def re_init(self, replace_word_index, index, device='cpu'):
        self.index = index
        self.embed_param = self.embed_weights[self.mask]
        self._weights = torch.ones(self.embed_param.shape[0]).to(device)
        self.weights = Parameter(self._weights)
        self.weights.requires_grad = True

    def forward(self, src, tgt, src_lengths):
        try:
            sentence_embedding = self.embed_weights[src.squeeze(1)]
            flag = True
        except:
            sentence_embedding = src
            flag = False

        lc = self.wt_average(self.weights)
        linear_combination = torch.mul(self.embed_param, lc.view(-1, 1)).sum(dim=0, keepdim=True)
        if flag:
            sentence_embedding[self.index] = linear_combination
        return (
         sentence_embedding, tgt, src_lengths)


class AdversarialExploit(nn.Module):

    def __init__(self, ae, model, opt=None):
        super(AdversarialExploit, self).__init__()
        self.ae = ae
        self.model = model
        self.generator = self.model.generator
        self.embedding_weight = self.model.encoder.embeddings.word_lut.weight.data
        self.opt = opt

    def encoder(self, src, src_length):
        src_embedding = self.ae(src, None, None)[0]
        return self.model.encoder(src_embedding, src_length)

    def forward(self, src, tgt, src_lengths):
        output = self.ae(src, tgt, src_lengths)
        return (self.model)(*output)


class MultiLinearCombiner(nn.Module):
    r"""'\n        Same as LinearCombiner but this class deals with generalised problem of\n        multiple words replacements. The min_grad_method gets modified to\n        return multiple positions. Here multiple weight vectors would be\n        present to deal with multiple positions. All thw weight vectors are\n        updates parallelly.\n    '"""

    def __init__(self, model, replace_word_index, index, vocab, device='cpu'):
        """
            index(list int): list of positions.
            replace_word_index(list int): list of word indices.
        """
        super(MultiLinearCombiner, self).__init__()
        self.wt_average = nn.Softmax(dim=(-1))
        self.vocab = vocab
        self.index = index
        self.embed_weights = model.encoder.embeddings.word_lut.weight.data
        self.mask = dict()
        self.embed_param = dict()
        for i, r in zip(self.index, replace_word_index):
            i = str(i)
            self.mask[i] = word2vec_model.filter_vocab(vocab[r], vocab)
            self.embed_param[i] = self.embed_weights[self.mask[i]]
            _weights = torch.ones(self.mask[i].shape[0]).to(device)
            self.register_parameter(i, Parameter(_weights))
            getattr(self, i).requires_grad = True

    def re_init(self, replace_word_index, index, device='cpu'):
        self.index = index
        self.mask = word2vec_model.filter_vocab(self.vocab[replace_word_index], self.vocab)
        self.embed_param = self.embed_weights[self.mask]
        self._weights = torch.ones(self.embed_param.shape[0]).to(device)
        self.weights = Parameter(self._weights)
        self.weights.requires_grad = True

    def forward(self, src, tgt, src_lengths):
        try:
            sentence_embedding = self.embed_weights[src.squeeze(1)]
        except:
            sentence_embedding = src

        lc = dict()
        linear_combination = dict()
        for index in map(str, self.index):
            lc[index] = self.wt_average(getattr(self, index))
            linear_combination[index] = torch.mul(self.embed_param[index], lc[index].view(-1, 1)).sum(dim=0, keepdim=True)
            sentence_embedding[int(index)] = linear_combination[index]

        return (
         sentence_embedding, tgt, src_lengths)

    def get_logits(self):
        """
            to get max and argmax of the weight vectors
        """
        lc = dict()
        p, m, am = [], [], []
        for index in map(str, self.index):
            lc[index] = self.wt_average(getattr(self, index))
            m.append(lc[index].max())
            am.append(self.mask[index][lc[index].argmax()])
            p.append(int(index))

        return (p, m, am)

    def remove_index(self, index):
        """
            Removes the provided index(int) from the self.index list to stop
            its proccessing. Assuming that the attacking model deals with
            rest all replacements.
        """
        self.index = self.index[(self.index != index)]

    def is_completed(self):
        """
            If all the indices are finished, termination condition is attained.
        """
        return self.index.shape[0] == 0


class MultiSoftmaxAttacker:
    r"""'\n        Softmax based attack on multiple positions. This attack makes use of\n        information from both encoder and decoder. First the postion is\n        identified using min_grad_method(). Then the replacement word is\n        searched by iteratively updating the weights for each words.\n\n    '"""

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

    def _run(self):
        count = defaultdict(lambda : 0)
        cont_value = defaultdict(lambda : -1)
        src, src_length = self.data.src
        tgt = self.data.tgt
        model_wrapper = self.model.ae
        for i in range(self.max_iterations):
            output, attn = self.model(src, tgt, src_length)
            loss, _ = self.critiria(self.data, output, attn)
            loss.backward()
            self.optimizer.step()
            positions, logits_max, logits_argmax = model_wrapper.get_logits()
            print(i, loss.item(), logits_max, logits_argmax)
            for position, index, max_val in zip(positions, logits_argmax, logits_max):
                if max_val > self.threshold:
                    count[position] += 1
                    if cont_value[position] == -1:
                        cont_value[position] = index
                    else:
                        if count[position] == self.max_count:
                            if cont_value[position] == index:
                                src[position] = index
                                model_wrapper.remove_index(position)
                            if cont_value[position] != index:
                                cont_value[position] = -1
                                count[position] = 0
                else:
                    if cont_value[position] != -1:
                        cont_value[position] = -1
                        count[position] = 0

            print(model_wrapper.is_completed(), model_wrapper.index, count[position], position)
            if model_wrapper.is_completed():
                break

        adv_src = src
        adv_src_text = ' '.join([self.src_vocab[j] for j in adv_src.squeeze()]) + '\n'
        self.outfile.write(adv_src_text)

    def run(self):
        count = defaultdict(lambda : 0)
        cont_value = defaultdict(lambda : -1)
        src, src_length = self.data.src
        tgt = self.data.tgt
        model_wrapper = self.model.ae
        for i in range(self.max_iterations):
            output, attn = self.model(src, tgt, src_length)
            loss, _ = self.critiria(self.data, output, attn)
            loss.backward()
            self.optimizer.step()
            positions, logits_max, logits_argmax = model_wrapper.get_logits()
            print(i, loss.item(), logits_max, logits_argmax)
            for position, index, max_val in zip(positions, logits_argmax, logits_max):
                if max_val > self.threshold:
                    count[position] += 1
                    if cont_value[position] == -1:
                        cont_value[position] = index
                    else:
                        if count[position] == self.max_count:
                            if cont_value[position] == index:
                                src[position] = index
                                model_wrapper.remove_index(position)
                            if cont_value[position] != index:
                                cont_value[position] = -1
                                count[position] = 0
                else:
                    if cont_value[position] != -1:
                        cont_value[position] = -1
                        count[position] = 0

            print(model_wrapper.is_completed(), model_wrapper.index, count[position], position)
            if model_wrapper.is_completed():
                break

        adv_src = src
        adv_src_text = ' '.join([self.src_vocab[j] for j in adv_src.squeeze()]) + '\n'
        self.outfile.write(adv_src_text)


class IterTargetedSoftmaxMinGradAttacker(SoftmaxAttacker, object):

    def __init__(self, model, critiria, optimizer, data, src_vocab, outfile, max_iterations=1000, max_count=10, threshold=0.9):
        super(IterTargetedSoftmaxMinGradAttacker, self).__init__(model, critiria, optimizer, data, src_vocab, outfile, max_iterations, max_count, threshold)

    def run(self):
        src, src_length = self.data.src
        tgt = self.data.tgt
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
        print(main_vocab_file)
        words, _ = torch.load(main_vocab_file)
        words = [self.model.ae.vocab.index(w) for w in words]
        good_vocab = torch.LongTensor(list(set(words).difference(src.squeeze().tolist())))
        print(len(set(words)), len(set(src.squeeze().tolist())), len(good_vocab))
        torch.save(good_vocab, tmp_file_name)
        modified_embedding = self.model.embedding_weight[src.squeeze(2)]
        output, attn = self.model.model(modified_embedding, tgt, src_length)
        org_loss, _ = self.critiria(self.data, output, attn)
        org_loss = org_loss.item()
        print('org_loss:', org_loss)
        index_changed = []
        min_loss = 100
        iter = 0
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
                self.model.ae.re_init((src[j].item()), j, device=(src.device))
                self.optimizer = build_torch_optimizer(self.model, self.model.opt)

                # calculates the loss using the original source
                modified_embedding = self.model.embedding_weight[src.squeeze(2)]
                output, attn = self.model.model(modified_embedding, tgt, src_length)
                proj_loss, _ = self.critiria(self.data, output, attn)
                
                # calculates the softmax_loss
                flip_word = super(IterTargetedSoftmaxMinGradAttacker, self).run(return_index=True)
                tmp_index = src[j].item()
                src[j] = flip_word
                self.data.src = (src, src_length)
                output, attn = self.model(src, tgt, src_length)
                loss, _ = self.critiria(self.data, output, attn)
                
                # calculates the projected loss after modification
                # (this is just for verification)
                modified_embedding = self.model.embedding_weight[src.squeeze(2)]
                output, attn = self.model.model(modified_embedding, tgt, src_length)
                new_loss, _ = self.critiria(self.data, output, attn)
                
                print('index:', j, 'loss:', loss.item(), 'min_loss:', min_loss, 'new_loss:', new_loss.item())
                print('proj_loss:', proj_loss.item())
                do_replace = True
                # this is basically comparing previous projected loss
                # and current loss
                if proj_loss < loss:
                    if j in index_changed:
                        do_replace = False
                if min_loss > loss and flip_word != tmp_index and do_replace:
                    flag = True
                    if j not in index_changed:
                        index_changed.append(j)
                    min_loss = max(loss.item(), org_loss)
                else:
                    src[j] = tmp_index
                print('.', end='')
                self.data.src = (src, src_length)

            if flag == False:
                break

        print('')
        print(index_changed)
        adv_src = src
        adv_src_text = ' '.join([self.src_vocab[j] for j in adv_src.squeeze()]) + '\n'
        self.outfile.write(adv_src_text)
        logger.info('LOSS({})\tNUM_ITERATIONS({})'.format(min_loss, iter))
        logger.info('SENT: {}\n'.format(adv_src_text))


class IterTargetedHotFlipMinGradAttacker(HotFlipAttacker, object):

    def __init__(self, model, critiria, optimizer, data, src_vocab, outfile, max_iterations=1000, max_count=10, threshold=0.9):
        super(IterTargetedHotFlipMinGradAttacker, self).__init__(model, critiria, optimizer, data, src_vocab, outfile, max_iterations, max_count, threshold)

    def run(self):
        src, src_length = self.data.src
        tgt = self.data.tgt
        min_loss = 100
        iter = 0
        for iter in range(20):
            flag = False
            indices = []
            while len(indices) != len(src) - 1:
                for ind in self.attr_min_grad((self.data), k=(-1)):
                    if ind not in indices:
                        indices.append(ind)
                        break

                j = indices[(-1)]
                self.data.src = (
                 src, src_length)
                self.model.ae.re_init(src, j, device=(src.device))
                self.optimizer = build_torch_optimizer(self.model, self.model.opt)
                flip_word = super(IterTargetedHotFlipMinGradAttacker, self).run(return_index=True)
                tmp_index = src[j].item()
                src[j] = flip_word
                self.data.src = (src, src_length)
                modified_embedding = self.model.embedding_weight[src.squeeze(2)]
                output, attn = self.model.model(modified_embedding, tgt, src_length)
                loss, _ = self.critiria(self.data, output, attn)
                print('index:', j, 'loss:', loss.item())
                if loss < min_loss:
                    min_loss = loss
                    flag = True
                else:
                    src[j] = tmp_index
                print('.', end='')
                self.data.src = (src, src_length)

            if flag == False:
                break

        print('')
        adv_src = src
        adv_src_text = ' '.join([self.src_vocab[j] for j in adv_src.squeeze()]) + '\n'
        self.outfile.write(adv_src_text)
        logger.info('LOSS({})\tNUM_ITERATIONS({})'.format(min_loss.item(), iter))
        logger.info('SENT: {}\n'.format(adv_src_text))


from new_attacks import FullyTargetedAttacker, FTHotFlipAttacker
str2attacker = {'softmax':SoftmaxAttacker, 
 'enc_softmax':EncSoftmaxAttacker, 
 'multisoftmax':MultiSoftmaxAttacker, 
 'hotflip':HotFlipAttacker, 
 'fthotflip':FTHotFlipAttacker, 
 'pos_hotflip':PosHotflipAttacker, 
 'bruteforce':BruteForceAttacker, 
 'fullytargeted':FullyTargetedAttacker, 
 'ft_hotflip':IterTargetedHotFlipMinGradAttacker, 
 'ftsoftmax':IterTargetedSoftmaxMinGradAttacker}
# okay decompiling __pycache__/adv_attacks.cpython-36.pyc
