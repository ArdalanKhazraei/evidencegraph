
from edge_utils import *

from prep import texts1, trees1, texts2, trees2
from NN_utils import node_feats, edge_feats
from parser_utils import get_static_folds, shuffle_folds

import numpy as np
import os, argparse

levels = ["at", "nfu", "efu", "ero"]

parser = argparse.ArgumentParser()

parser.add_argument('--corpus', dest = 'corpus')
parser.add_argument('--large', dest = 'large', action='store_true')
parser.add_argument('--weight_opt', dest = 'weight_opt', action='store_true')
parser.add_argument('--eval_mst', dest = 'eval_mst', action='store_true')
parser.add_argument('--mst_verbose', dest = 'mst_verbose', action='store_true')
parser.add_argument('--reports', dest = 'reports', action='store_true')
parser.add_argument('--eval_base', dest = 'eval_base', action='store_true')
parser.add_argument('--base_verbose', dest = 'base_verbose', action='store_true')

options, rest = parser.parse_known_args()

corpus= options.corpus
large=options.large
weight_opt = options.weight_opt
eval_mst = options.eval_mst
mst_verbose = options.mst_verbose
reports = options.reports
eval_base = options.eval_base
base_verbose = options.base_verbose

texts = eval("texts"+corpus) if corpus != 'c' else {**texts1, **texts2}
trees = eval("trees"+corpus) if corpus != 'c' else {**trees1, **trees2}

efeats = edge_feats(large)
nfeats = node_feats(corpus, large)

assert efeats.keys() >= nfeats.keys()

features = {t: {"edge": efeats[t], "node": nfeats[t]} for t in trees}

efl = {len(f) for t, d in features.items() for e, f in d["edge"].items()}
assert len(efl) == 1
edge_features_length = efl.pop()

nfl = {len(f) for t, d in features.items() for e, f in d["node"].items()}
assert len(nfl) == 1
node_features_length = nfl.pop()

from NN_classifier import NNClassifier

folds = list(get_static_folds()) if corpus=='1' else shuffle_folds(corpus)

if __name__ == '__main__':
    base_maF1s, base_accs = {level: [] for level in levels}, {level: [] for level in levels}
    mst_maF1s, mst_accs = {level: [] for level in levels}, {level: [] for level in levels}
    wo_maF1s, wo_accs = {level: [] for level in levels}, {level: [] for level in levels}
    
    model_path = "trained/models/"+("combined" if corpus=='c' else "Part"+corpus)+"/pt_att4"+("_large/" if large else "/")
    weight_path = "trained/weight_opt/"+("combined" if corpus=='c' else "Part"+corpus)+"/pt_att4"+("_large/" if large else "/")
    
    for train_tids, test_tids, i in folds:
        if not os.path.isfile(model_path+"fold_"+str(i)):
            continue
        if not (eval_base or eval_mst) and (not weight_opt or not os.path.isfile(weight_path+"fold_"+str(i))):
            continue
        print("\n  fold  _",i+1,"/",len(folds),"\n")
        clf = NNClassifier(edge_features_length, node_features_length, large=large)
        try:
            clf.load(model_path+"fold_"+str(i))
        except RuntimeError:
            continue
        test_feats = [features[t] for t in test_tids]
        test_gold = [trees[t] for t in test_tids]
        
        if eval_base:
            if base_verbose:
                print("base")
            for level in levels:
                _, _, maF1, acc = get_Fs(clf, level, test_feats, test_gold, base=True)
                base_maF1s[level].append(maF1)
                base_accs[level].append(acc)
                if base_verbose:
                    print(level, end=' ')
                    print("macro F1", maF1, "accuracy", acc)
        
        if eval_mst:
            if mst_verbose or reports:
                print("mst")
            for level in levels:
                if mst_verbose or reports:
                    print(level, end=' ')
                golds, preds, maF1, acc = get_Fs(clf, level, test_feats, test_gold)
                mst_maF1s[level].append(maF1)
                mst_accs[level].append(acc)
                if mst_verbose:
                    print("macro F1", maF1, "accuracy", acc)
                if reports:
                    print(f"Model Accuracy: {accuracy_score(golds, preds):.4f}")
                    print(classification_report(golds, preds))
                    
        if weight_opt:
            if not os.path.isfile(weight_path+"fold_"+str(i)):
                continue
            with open(weight_path+"fold_"+str(i), 'r') as f:
                clf.weighting = eval(f.read())
            print("Loaded weights {}".format(weight_path+"fold_"+str(i)))
            if mst_verbose or reports:
                print("wo")
            for level in levels:
                if mst_verbose or reports:
                    print(level, end=' ')
                golds, preds, maF1, acc = get_Fs(clf, level, test_feats, test_gold)
                wo_maF1s[level].append(maF1)
                wo_accs[level].append(acc)
                if mst_verbose:
                    print("macro F1", maF1, "accuracy", acc)
                if reports:
                    print(f"Model Accuracy: {accuracy_score(golds, preds):.4f}")
                    print(classification_report(golds, preds))
    
    result = ""
    
    if eval_base:
        result += "base\n\n"
        result += "n = " + str(len(base_maF1s["at"])) + "\n"
        for l in levels:
            result += '\n' + l + '   '
            result += "macro avg " + str(np.mean(base_maF1s[l])) + u" \u00B1 " + str(np.std(base_maF1s[l])/np.sqrt(len(base_maF1s[l])-1))
            result += "  "
            result += "accuracy avg " + str(np.mean(base_accs[l])) + u" \u00B1 " + str(np.std(base_accs[l])/np.sqrt(len(base_accs[l])-1))
    
    if eval_mst:
        result += "\n\nmst\n\n"
        result += "n = " + str(len(mst_maF1s["at"])) + "\n"
        for l in levels:
            result += '\n' + l + '   '
            result += "macro avg " + str(np.mean(mst_maF1s[l])) + u" \u00B1 " + str(np.std(mst_maF1s[l])/np.sqrt(len(mst_maF1s[l])-1))
            result += "  "
            result += "accuracy avg " + str(np.mean(mst_accs[l])) + u" \u00B1 " + str(np.std(mst_accs[l])/np.sqrt(len(mst_accs[l])-1))
        
    if weight_opt:
        result += "\n\nwo\n\n"
        result += "n = " + str(len(wo_maF1s["at"])) + "\n"
        for l in levels:
            result += '\n' + l + '   '
            result += "macro avg " + str(np.mean(wo_maF1s[l])) + u" \u00B1 " + str(np.std(wo_maF1s[l])/np.sqrt(len(wo_maF1s[l])-1))
            result += "  "
            result += "accuracy avg " + str(np.mean(wo_accs[l])) + u" \u00B1 " + str(np.std(wo_accs[l])/np.sqrt(len(wo_accs[l])-1))

    print(result)
    print()
    
    with open(f'evaluations/edge/{"large" if large else "base"}_{corpus}', 'w') as f:
        f.write(result)
        
