
from utils import *
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--large', dest = 'large', action='store_true')
parser.add_argument('--regular', dest = 'regular', action='store_true')
parser.add_argument('--masked', dest = 'masked', action='store_true')
parser.add_argument('--pairs', dest = 'pairs', action='store_true')

options, rest = parser.parse_known_args()

large = options.large
regular = options.regular
masked = options.masked
pairs = options.pairs

from prep import texts1, texts2
texts = {**texts1, **texts2}

from transformers import AutoTokenizer, AutoModelForSequenceClassification
repo = 'cross-encoder/nli-deberta-v3-' + ('large' if large else 'base')
tokenizer = AutoTokenizer.from_pretrained(repo)
model = AutoModelForSequenceClassification.from_pretrained(repo)

prefix = 'attention_features/' + ('large/' if large else '')

lims_reg_path = prefix + '1_lims_reg.txt'
lims_mask_path = prefix + '2_lims_mask.txt'
lims_pair_path = prefix + '3_lims_pair.txt'


def edges(n, cls=True):
    r = []
    for i in range(0-cls, n):
        for j in range(0-cls, n):
            r.append((i, j))
    return r


corpus_features = {}

def lnd(tokenizer, model, sent_list):
    inputs, lims = sl2i(tokenizer, sent_list)
    a = attentions(model, inputs)
    keys = edges(len(lims))
    a_feats = {}
    for i,j in keys:
        a_feats[(i,j)] = [[satt(l,k,lims,i,j) for k in range(l.shape[1])] for l in a]
    return lims, a_feats

if regular:
    print("\nlimits regular\n")
    ctr = 0
    for t in texts:
        ctr += 1
        print("  processing text  _ ", ctr,"/",len(texts), end='\r')
        corpus_features[t] = lnd(tokenizer, model, texts[t])

    with open(lims_reg_path, 'w') as f:
        f.write(str(corpus_features))


corpus_features = {}

def lmc(tokenizer, model, sent_list):
    inputs, lims = sl2i(tokenizer, sent_list)
    mi = masked_input(inputs, lims, mask_cls=True)
    a = attentions(model, mi)
    keys = edges(len(lims), False)
    a_feats = {}
    for i,j in keys:
        a_feats[(i,j)] = [[satt(l,k,lims,i,j) for k in range(l.shape[1])] for l in a]
    return lims, a_feats

if masked:
    print("\nlimits mask cls\n")
    ctr = 0
    for t in texts:
        ctr += 1
        print("  processing text  _ ", ctr,"/",len(texts), end='\r')
        corpus_features[t] = lmc(tokenizer, model, texts[t])

    with open(lims_mask_path, 'w') as f:
        f.write(str(corpus_features))


corpus_features = {}

def lpw(tokenizer, model, sent_list):
    inputs, lims = sl2i(tokenizer, sent_list)
    keys = pairs(len(lims))
    a_feats = {}
    u = set(range(len(lims)))
    for i,j in keys:
        mi = masked_input(inputs, lims, u-{i,j}, True)
        a = attentions(model, mi)
        a_feats[(i,j)] = [[satt(l,k,lims,i,j) for k in range(l.shape[1])] for l in a]
    return lims, a_feats

if pairs:
    print("\nlimit pairwise\n")
    ctr = 0
    for t in texts:
        ctr += 1
        print("  processing text  _ ", ctr,"/",len(texts), end='\r')
        corpus_features[t] = lpw(tokenizer, model, texts[t])

    with open(lims_pair_path, 'w') as f:
        f.write(str(corpus_features))


