
from prep import texts1, trees1, texts2, trees2
from parser_utils import get_static_folds, shuffle_folds

from sklearn.metrics import precision_recall_fscore_support, accuracy_score, classification_report, confusion_matrix
import numpy as np

import os, argparse

levels = ["cc", "at", "fu", "ro"]

parser = argparse.ArgumentParser()

parser.add_argument('--corpus', dest = 'corpus')
parser.add_argument('--weight_opt', dest = 'weight_opt', action='store_true')
parser.add_argument('--eval_mst', dest = 'eval_mst', action='store_true')
parser.add_argument('--mst_verbose', dest = 'mst_verbose', action='store_true')
parser.add_argument('--reports', dest = 'reports', action='store_true')
parser.add_argument('--eval_base', dest = 'eval_base', action='store_true')
parser.add_argument('--base_verbose', dest = 'base_verbose', action='store_true')

options, rest = parser.parse_known_args()

corpus= options.corpus
weight_opt = options.weight_opt
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
    
    base_maF1s, base_miF1s = {level: [] for level in levels}, {level: [] for level in levels}
    mst_maF1s, mst_miF1s = {level: [] for level in levels}, {level: [] for level in levels}
    wo_maF1s, wo_miF1s = {level: [] for level in levels}, {level: [] for level in levels}
    
    model_path = "trained/models/"+("combined" if corpus=='c' else "Part"+corpus)+"/eg/"
    weight_path = "trained/weight_opt/"+("combined" if corpus=='c' else "Part"+corpus)+"/eg/"

    for train_tids, test_tids, i in folds:
        if not os.path.isfile(model_path+"fold_"+str(i)):
            continue
        if not (eval_base or eval_mst) and (not weight_opt or not os.path.isfile(weight_path+"fold_"+str(i))):
            continue
        print("\n  fold  _",i+1,"/",len(folds),"\n")
        clf = EvidenceGraphClassifier(
            features.feature_function_segments,
            features.feature_function_segmentpairs,
            relation_set=SIMPLE_RELATION_SET
        )
        try:
            clf.load(model_path+"fold_"+str(i))
        except RuntimeError:
            continue
        test_txts = [texts[i] for i in test_tids]
        test_args = [trees[i] for i in test_tids]
        
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
                    print(f"Model Accuracy: {accuracy_score(mst_golds, mst_preds):.4f}")
                    print(classification_report(mst_golds, mst_preds))
                    
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
                mst_preds = []
                mst_golds = []
                for j in range(len(test_txts)):
                    mst = clf.predict(test_txts[j])
                    mst_preds += eval("mst.get_"+level+"_vector()")
                    mst_golds += eval("test_args[j].get_"+level+"_vector()")
                _, _, wo_maF1, _ = precision_recall_fscore_support(mst_golds, mst_preds, average="macro")
                _, _, wo_miF1, _ = precision_recall_fscore_support(mst_golds, mst_preds, average="micro")
                wo_maF1s[level].append(wo_maF1)
                wo_miF1s[level].append(wo_miF1)
                if mst_verbose:
                    print("macro", wo_maF1, "micro", wo_miF1)
                if reports:
                    print(f"Model Accuracy: {accuracy_score(wo_golds, wo_preds):.4f}")
                    print(classification_report(wo_golds, wo_preds))
    
    result = ""
    
    if eval_base:
        result += "base\n\n"
        result += "n = " + str(len(base_miF1s["at"])) + "\n"
        for l in levels:
            result += '\n' + l + '   '
            result += "macro avg " + str(np.mean(base_maF1s[l])) + u" \u00B1 " + str(np.std(base_maF1s[l])/np.sqrt(len(base_maF1s[l])-1))
            result += "  "
            result += "micro avg " + str(np.mean(base_miF1s[l])) + u" \u00B1 " + str(np.std(base_miF1s[l])/np.sqrt(len(base_miF1s[l])-1))
    
    if eval_mst:
        result += "\n\nmst\n\n"
        result += "n = " + str(len(mst_miF1s["at"])) + "\n"
        for l in levels:
            result += '\n' + l + '   '
            result += "macro avg " + str(np.mean(mst_maF1s[l])) + u" \u00B1 " + str(np.std(mst_maF1s[l])/np.sqrt(len(mst_maF1s[l])-1))
            result += "  "
            result += "micro avg " + str(np.mean(mst_miF1s[l])) + u" \u00B1 " + str(np.std(mst_miF1s[l])/np.sqrt(len(mst_miF1s[l])-1))
        
    if weight_opt:
        result += "\n\nwo\n\n"
        result += "n = " + str(len(wo_miF1s["at"])) + "\n"
        for l in levels:
            result += '\n' + l + '   '
            result += "macro avg " + str(np.mean(wo_maF1s[l])) + u" \u00B1 " + str(np.std(wo_maF1s[l])/np.sqrt(len(wo_maF1s[l])-1))
            result += "  "
            result += "micro avg " + str(np.mean(wo_miF1s[l])) + u" \u00B1 " + str(np.std(wo_miF1s[l])/np.sqrt(len(wo_miF1s[l])-1))
    
    print(result)
    
    with open(f'evaluations/eg_{corpus}', 'w') as f:
        f.write(result)
            
