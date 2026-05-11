
import os, argparse

from NN_utils import node_feats, edge_feats, labels, dataset

from sklearn.metrics import precision_recall_fscore_support, accuracy_score, classification_report, confusion_matrix
import numpy as np

from itertools import permutations

levels = ["cc", "at", "fu", "ro"]

parser = argparse.ArgumentParser()

parser.add_argument('--corpus', dest = 'corpus')
parser.add_argument('--large', dest = 'large', action='store_true')
parser.add_argument('--level', dest = 'level')

options, rest = parser.parse_known_args()

corpus= options.corpus
large=options.large
level = options.level

assert level in levels

n_out = 3 if level == "fu" else 2

if large:
    n_hiddens = [600, 100, 20] if level == "att" else [1000, 300, 60] if level == "fu" else [1000, 300, 40]
else:
    n_hiddens = [80, 40, 20] if level == "att" else [300, 60] if level == "fu" else [300, 40]

from NN_model import model, batch_onehot

root_dir = "trained/layer/" + level +"/"

L = 12 if level == "at" else 13

if __name__ == '__main__':
    for layer in range(L):
        
        model_path = root_dir + str(layer) + "/"
        if os.path.isdir(model_path) and os.path.isfile(model_path+"evals.txt"):
            continue
        
        print("layer", layer)

        feats_dict = edge_feats(large, layer) if level == "at" else node_feats(corpus, large, layer)
        gold = labels(lambda g: dict(zip( sorted( permutations(g.nodes(), 2) if level == "at" else g.nodes() ), eval( "g.get_"+level+"_vector()" )  ) ) )
        
        feature_lengths = {len(f) for t, d in feats_dict.items() for e, f in d.items()}
        assert len(feature_lengths) == 1
        feat_len = feature_lengths.pop()
        
        print(feat_len)
        
        txt = ""
        maF1s = []
        miF1s = []
        
        print("model path =", model_path)
        if not os.path.isdir(model_path):
            os.mkdir(model_path)
            print("created model directory")
        else:
            print("model directory previously existed")

        for i in range(50):
            s = "\n  fold  _" + str(i+1)  + "/50\n" # print("  _",i+1,"/",len(folds), end="\r")
            print(s)
            txt += s
            X_train, y_train, X_test, y_test = dataset(feats_dict, gold, corpus, i)
            clf = model(feat_len, n_hiddens, n_out)
            try:
                clf.load(model_path+"fold_"+str(i))
            except RuntimeError:
                dr = 5 if level == "at" else 5
                lr = 0.01 if level == "at" else 0.01
                epochs = 30 # 20 if level == "at" else 20
                clf.train(X_train, y_train, epochs, lr=lr, dr=dr)
                clf.save(model_path+"fold_"+str(i))
            preds = clf.predict(X_test).detach().numpy()
            preds = np.argmax(preds, -1)
            _, _, maF1, _ = precision_recall_fscore_support(y_test, preds, average="macro")
            _, _, miF1, _ = precision_recall_fscore_support(y_test, preds, average="micro")
            s = "macro " + str(maF1) + " micro " + str(miF1) + "\n"
            print(s)
            txt += s
            maF1s.append(maF1)
            miF1s.append(miF1)
        txt += '\n\n'
        s = "macro avg " + str(np.mean(maF1s)) + u" \u00B1 " + str(np.std(maF1s)/np.sqrt(len(maF1s)-1)) + " micro avg " + str(np.mean(miF1s)) + u" \u00B1 " + str(np.std(miF1s)/np.sqrt(len(miF1s)-1)) + "\n"
        print(s)
        txt += s
        with open(model_path+"evals.txt", 'w') as f:
            f.write(txt)
