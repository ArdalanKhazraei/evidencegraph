from prep import texts1, trees1, texts2, trees2
from NN_utils import node_feats, edge_feats
from parser_utils import get_static_folds, shuffle_folds

from sklearn.metrics import precision_recall_fscore_support, accuracy_score, classification_report, confusion_matrix
import numpy as np

import os, argparse

levels = ["cc", "at", "fu", "ro"]

parser = argparse.ArgumentParser()

parser.add_argument('--corpus', dest = 'corpus')
parser.add_argument('--large', dest = 'large', action='store_true')
parser.add_argument('--eval_mst', dest = 'eval_mst', action='store_true')
parser.add_argument('--mst_verbose', dest = 'mst_verbose', action='store_true')
parser.add_argument('--reports', dest = 'reports', action='store_true')
parser.add_argument('--eval_base', dest = 'eval_base', action='store_true')
parser.add_argument('--base_verbose', dest = 'base_verbose', action='store_true')

options, rest = parser.parse_known_args()

corpus= options.corpus
large=options.large

texts = eval("texts"+corpus) if corpus != 'c' else {**texts1, **texts2}
trees = eval("trees"+corpus) if corpus != 'c' else {**trees1, **trees2}

efeats = edge_feats(large)
nfeats = node_feats(corpus, large)

assert efeats.keys() >= nfeats.keys()

features = {t: {"edge": efeats[t], "node": nfeats[t]} for t in trees}

efl = {len(f) for t, d in features.items() for e, f in d["edge"].items()}
assert len(efl) == 1
edge_features_length = efl.pop()

print("edge_features_length =", edge_features_length)

nfl = {len(f) for t, d in features.items() for e, f in d["node"].items()}
assert len(nfl) == 1
node_features_length = nfl.pop()

print("node_features_length =", node_features_length)

from NN_classifier import NNClassifier

folds = list(get_static_folds()) if corpus=='1' else shuffle_folds(corpus)


if __name__ == '__main__':
    
    read_path = "trained/models/"+("combined" if corpus=='c' else "Part"+corpus)+"/pt_att4"+("_large/" if large else "/")
    write_path = "trained/weight_opt/"+("combined" if corpus=='c' else "Part"+corpus)+"/pt_att4"+("_large/" if large else "/")

    for train_tids, test_tids, i in folds:
        print("\n  fold  _",i+1,"/",len(folds),"\n")
        if os.path.isfile(write_path+"fold_"+str(i)) or not os.path.isfile(read_path+"fold_"+str(i)):
            continue
        clf = NNClassifier(edge_features_length, node_features_length, optimize_weighting="inner_cv", large=large)
        try:
            clf.load(read_path+"fold_"+str(i))
        except RuntimeError:
            continue
        train_feats = [features[t] for t in train_tids]
        train_gold = [trees[t] for t in train_tids]
        clf.train_metaclassifier(train_feats, train_gold)
        with open(write_path+"fold_"+str(i), 'w') as f:
            f.write(str(clf.weighting))
