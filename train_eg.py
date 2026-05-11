
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
eval_mst = options.eval_mst
mst_verbose = options.mst_verbose
reports = options.reports
eval_base = options.eval_base
base_verbose = options.base_verbose

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
    
    model_path = "trained/models/"+("combined" if corpus=='c' else "Part"+corpus)+"/eg/"

    for train_tids, test_tids, i in folds:
        print("\n  fold  _",i+1,"/",len(folds),"\n")
        clf = EvidenceGraphClassifier(
            features.feature_function_segments,
            features.feature_function_segmentpairs,
            relation_set=SIMPLE_RELATION_SET
        )
        try:
            clf.load(model_path+"fold_"+str(i))
        except RuntimeError:
            train_txts = [texts[t] for t in train_tids]
            train_args = [trees[t] for t in train_tids]
            clf.train(train_txts, train_args)
            clf.save(model_path+"fold_"+str(i))
        test_txts = [texts[i] for i in test_tids]
        test_args = [trees[i] for i in test_tids]
        
        if eval_mst:
            if mst_verbose or reports:
                print("mst")
            for level in levels:
                if mst_verbose or reports:
                    print(level, end=' ')
                mst_preds = []
                mst_golds = []
                for j in range(len(test_txts)):
                    mst = clf.predict(test_txts[j])
                    mst_preds += eval("mst.get_"+level+"_vector()")
                    mst_golds += eval("test_args[j].get_"+level+"_vector()")
                _, _, mst_maF1, _ = precision_recall_fscore_support(mst_golds, mst_preds, average="macro")
                _, _, mst_miF1, _ = precision_recall_fscore_support(mst_golds, mst_preds, average="micro")
                mst_maF1s[level].append(mst_maF1)
                mst_miF1s[level].append(mst_miF1)
                if mst_verbose:
                    print("macro", mst_maF1, "micro", mst_miF1)
                if reports:
                    print(f"Model Accuracy: {accuracy_score(mst_golds[level], mst_preds[level]):.4f}")
                    print(classification_report(mst_golds[level], mst_preds[level]))
        
        if eval_base:
            if base_verbose:
                print("base")
            for level in levels:
                base_maF1, base_miF1 = clf.ensemble[level].test(test_txts, test_args)
                base_maF1s[level].append(base_maF1)
                base_miF1s[level].append(base_miF1)
                if base_verbose:
                    print(level, end=' ')
                    print("macro", base_maF1, "micro", base_miF1)
    
    if eval_base:
        print("\nbase\n")
        print ("n =", len(base_miF1s["at"]))
        for l in levels:
            print(l, end = ' ') 
            print("macro avg", np.mean(base_maF1s[l]), u"\u00B1", np.std(base_maF1s[l])/np.sqrt(len(base_maF1s[l])-1), "micro avg", np.mean(base_miF1s[l]), u"\u00B1", np.std(base_miF1s[l])/np.sqrt(len(base_miF1s[l])-1))
    
    if eval_mst:
        print("\nmst\n")
        print ("n =", len(mst_miF1s["at"]))
        for l in levels:
            print(l, end = ' ') 
            print("macro avg", np.mean(mst_maF1s[l]), u"\u00B1", np.std(mst_maF1s[l])/np.sqrt(len(mst_maF1s[l])-1), "micro avg", np.mean(mst_miF1s[l]), u"\u00B1", np.std(mst_miF1s[l])/np.sqrt(len(mst_miF1s[l])-1))
        
