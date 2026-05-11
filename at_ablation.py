
import os

from NN_utils import edge_feats, labels, dataset

from sklearn.metrics import precision_recall_fscore_support, accuracy_score, classification_report, confusion_matrix
import numpy as np

from itertools import permutations

large=False
corpus = "c"
level = "at"
assert large == False and corpus == "c" and level == "at"

n_out = 2
n_hiddens = [40, 20, 10]

from NN_model import model, batch_onehot

root_dir = "trained/at_ablation/"

if __name__ == '__main__':
    for factor in {2, 4, 8}:
        print("factor", factor)

        feats_dict = edge_feats(factor=factor)
        gold = labels(lambda g: dict(zip( sorted( permutations(g.nodes(), 2) ), eval( "g.get_"+level+"_vector()" )  ) ) )
        
        feature_lengths = {len(f) for t, d in feats_dict.items() for e, f in d.items()}
        assert len(feature_lengths) == 1
        feat_len = feature_lengths.pop()

        print(feat_len)
        
        txt = ""
        maF1s = []
        miF1s = []
        
        model_path = root_dir + str(factor) + "/"
        print("model path =", model_path)
        if not os.path.isdir(model_path):
            os.mkdir(model_path)
            print("created model path")
        else:
            print("model path previously existed")
        
        for i in range(50):
            s = "\n  fold  _" + str(i+1)  + "/50\n" # print("  _",i+1,"/",len(folds), end="\r")
            print(s)
            txt += s
            X_train, y_train, X_test, y_test = dataset(feats_dict, gold, corpus, i)
            clf = model(feat_len, n_hiddens, n_out)
            try:
                clf.load(model_path+"fold_"+str(i))
            except RuntimeError:
                dr = 5
                lr = 0.01
                epochs = 30
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
