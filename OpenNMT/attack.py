#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import unicode_literals
from itertools import repeat

from onmt.utils.logging import init_logger
from onmt.utils.misc import split_corpus

import onmt.opts as opts
from onmt.utils.parse import ArgumentParser


def main(opt):
    ArgumentParser.validate_translate_opts(opt)

    logger = init_logger(opt.log_file)

    from adv_attacks import build_attacker, run_attacker, load_attack_model
    fields, model, model_opt = load_attack_model(opt)

    src_shards = split_corpus(opt.src, opt.shard_size)
    tgt_shards = split_corpus(opt.tgt, opt.shard_size) \
        if opt.tgt is not None else repeat(None)
    shard_pairs = zip(src_shards, tgt_shards)

    logger.info('Attack type: {}'.format(opt.attack_type))
    logger.info('Position: {}'.format(opt.position))
    logger.info('Optimizer: {}'.format(opt.optim))

    for (src_shards, tgt_shards) in shard_pairs:
        run_attacker(model, src_shards, tgt_shards, opt, fields)

def _get_parser():
    parser = ArgumentParser(description='attack.py')

    opts.config_opts(parser)
    opts.translate_opts(parser)
    opts.attack_opts(parser)
    return parser


if __name__ == "__main__":
    parser = _get_parser()

    opt = parser.parse_args()
    main(opt)
