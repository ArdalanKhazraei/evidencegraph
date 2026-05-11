
from prep import texts1, trees1, texts2, trees2
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


folds = list(get_static_folds()) if corpus=='1' else shuffle_folds(corpus)


if __name__ == '__main__':
    from evidencegraph.classifiers import EvidenceGraphClassifier
    from evidencegraph.features_text import init_language, TextFeatures
    from evidencegraph.argtree import SIMPLE_RELATION_SET

    features = init_language("en")
    features.feature_set = TextFeatures.F_SET_ALL_BUT_VECTORS
    
    mst_maF1s, mst_miF1s = {level: [] for level in levels}, {level: [] for level in levels}
    base_maF1s, base_miF1s = {level: [] for level in levels}, {level: [] for level in levels}
    
    read_path = "trained/models/"+("combined" if corpus=='c' else "Part"+corpus)+"/eg/"
    write_path = "trained/weight_opt/"+("combined" if corpus=='c' else "Part"+corpus)+"/eg/"

    for train_tids, test_tids, i in folds:
        print("\n  fold  _",i+1,"/",len(folds),"\n")
        if os.path.isfile(write_path+"fold_"+str(i)) or not os.path.isfile(read_path+"fold_"+str(i)):
            continue
        clf = EvidenceGraphClassifier(
            features.feature_function_segments,
            features.feature_function_segmentpairs,
            relation_set=SIMPLE_RELATION_SET
        )
        try:
            clf.load(read_path+"fold_"+str(i))
        except RuntimeError:
            continue
        train_txts = [texts[t] for t in train_tids]
        train_args = [trees[t] for t in train_tids]
        clf.train_metaclassifier(train_txts, train_args)
        with open(write_path+"fold_"+str(i), 'w') as f:
            f.write(str(clf.weighting))
