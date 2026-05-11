
import numpy as np
import torch
from prep import trees1, trees2
trees = {**trees1, **trees2}


from evidencegraph.utils import foldsof

######################################################################### genuine utils #########################################################################

def stats(array_list):
    means = np.mean(array_list, axis=0)
    stds = np.std(array_list, axis=0)
    return means, stds

def normalize(array_list, means, stds):
    return torch.tensor((array_list-means)/stds, dtype=torch.float32)

def onehot3(fu):
    r = [0]*3
    r[fu] = 1
    return torch.tensor(r, dtype=torch.float32)

def onehot2(b):
    r = [1-b, 0+b]
    return torch.tensor(r, dtype=torch.float32)

def batch_onehot(fu, n):
    onehot = eval("onehot"+str(n))
    return torch.cat([onehot(f).unsqueeze(0) for f in fu])

######################################################################### node features #########################################################################
    
def sent_feats(fd):
    return {n: torch.cat((torch.sum(fd[n], dim=0), torch.mean(fd[n], dim=0))).tolist() for n in fd if type(n)==int}
    
def layer_feats(cf, l):
    return {t: sent_feats(cf[t][l]) for t in cf}
    
def text_layers(mt, lr):
    rng = range(len(mt)) if lr == "all" else range(*lr)
    return {n: [i for l in rng for i in sent_feats(mt[l])[n]] for n in mt[0] if type(n)==int}
    
def total_feats(cf, lr):
    return {t: text_layers(cf[t], lr) for t in cf}

def node_feats(corpus, large=False, layer="all"):
    from compile_hidden_features import corpus_hiddens
    
    nfeats = (
        layer_feats(corpus_hiddens(corpus, large), layer) 
        if type(layer) == int 
        else total_feats(corpus_hiddens(corpus, large), layer)
    )
    
    assert all( list(nfeats[t].keys()) == list( range(1, 1+len(nfeats[t].keys()) ) ) for t in nfeats )
    print("node keys assert passed")
    
    return nfeats


######################################################################### edge features #########################################################################

def keys(mt):
    return [k for k in mt.keys() if k[0] >= 0 and k[1] >= 0 and k[0] != k[1]]

def att_feats(sl, tf, e, l, h, f):
    n = len(sl)
    il = sl[e[0]][1] - sl[e[0]][0]
    jl = sl[e[1]][1] - sl[e[1]][0]
    i, j = e
    r = [tf[(i,j)][l][h], tf[(j,i)][l][h]]
    assert f in {2, 4, 8}
    if f > 2:
        r += [tf[(i,j)][l][h]/il, tf[(j,i)][l][h]/jl]
        if f > 4:
            r += [n*tf[(i,j)][l][h], n*tf[(j,i)][l][h], n*tf[(i,j)][l][h]/il, n*tf[(j,i)][l][h]/jl]
    #r += [tf[(i,-1)][l][h], tf[(j,-1)][l][h], tf[(i,-1)][l][h]/il, tf[(j,-1)][l][h]/jl, n*tf[(i,-1)][l][h], n*tf[(j,-1)][l][h], n*tf[(i,-1)][l][h]/il, n*tf[(j,-1)][l][h]/jl]
    #r += [tf[(i,i)][l][h], tf[(j,j)][l][h], tf[(i,i)][l][h]/il, tf[(j,j)][l][h]/jl, n*tf[(i,i)][l][h], n*tf[(j,j)][l][h], n*tf[(i,i)][l][h]/il, n*tf[(j,j)][l][h]/jl]
    return r

def edge_feats(large=False, layer="all", factor=4):
    with open('attention_features/' + ('large/' if large else '') + '1_lims_reg.txt', 'r') as f:
        cfeats = eval(f.read())
    
    shapes = {np.shape(v) for t, f in cfeats.items() for v in f[1].values()}
    assert len(shapes)==1
    L, H = shapes.pop()
    
    rng = range(L) if layer == "all" else range(layer, layer+1) if type(layer) == int else range(*layer)
    feats_dict = {
        t: {(k[0]+1,k[1]+1): 
        [v for l in rng for h in range(H) for v in att_feats(*f, k, l, h, factor)] 
        for k in keys(f[1])} for t, f in cfeats.items()
    }
    
    print("\ninput dimensions of edge features:", set(len(v) for i in feats_dict.values() for v in i.values()))
    
    return feats_dict

######################################################################### edge features #########################################################################

def labels(tree_feat):
    return {t: tree_feat(trees[t]) for t in trees}

#######################################################################################################################################



def dataset(feats_dict, gold, corpus, i):
    assert feats_dict.keys() == gold.keys() == trees.keys()
    assert all(gold[t].keys() == feats_dict[t].keys() for t in trees)
    
    from evidencegraph.folds import get_static_folds
    from parser_utils import shuffle_folds
    folds = list(get_static_folds()) if corpus==1 else shuffle_folds(corpus)
    
    train_tids, test_tids, idx = folds[i]
    
    X_train = [feats_dict[t][k] for t in train_tids for k in sorted(gold[t])]
    y_train = [gold[t][k] for t in train_tids for k in sorted(gold[t])]
    
    X_test = [feats_dict[t][k] for t in test_tids for k in sorted(gold[t])]
    y_test = [gold[t][k] for t in test_tids for k in sorted(gold[t])]
    
    assert len(X_train) == len(y_train)
    assert len(X_test) == len(y_test)
    
    return X_train, y_train, X_test, y_test


#######################################################################################################################################

def devset(feats_dict, gold, corpus, i, n, j):
    assert feats_dict.keys() == gold.keys() == trees.keys()
    assert all(gold[t].keys() == feats_dict[t].keys() for t in trees)
    
    from evidencegraph.folds import get_static_folds
    from parser_folds_eval import shuffle_folds
    folds = list(get_static_folds()) if corpus==1 else shuffle_folds(corpus)
    
    train_tids, test_tids, idx = folds[i]
    
    (train_tids, _), (dev_tids, _) = list(foldsof(train_tids, train_tids, n))[j]
    
    X_train = [feats_dict[t][k] for t in train_tids for k in sorted(gold[t])]
    y_train = [gold[t][k] for t in train_tids for k in sorted(gold[t])]
    
    X_dev = [feats_dict[t][k] for t in dev_tids for k in sorted(gold[t])]
    y_dev = [gold[t][k] for t in dev_tids for k in sorted(gold[t])]
    
    X_test = [feats_dict[t][k] for t in test_tids for k in sorted(gold[t])]
    y_test = [gold[t][k] for t in test_tids for k in sorted(gold[t])]
    
    assert len(X_train) == len(y_train)
    assert len(X_dev) == len(y_dev)
    assert len(X_test) == len(y_test)
    
    return X_train, y_train, X_dev, y_dev, X_test, y_test


######################################################################### eval function #########################################################################


from sklearn.metrics import precision_recall_fscore_support, accuracy_score, classification_report, confusion_matrix

import torch
from torch import nn
from torch.nn import functional as F

def evals(gold, pred, verbose=True, report=True, con_mat=False, loss=False, weight=None):
    r = {}
    if loss:
        n = pred.shape[1]
        assert n in {2, 3}
        r['loss'] = F.cross_entropy(pred, batch_onehot(gold, n), weight=weight, reduction='mean').item()
        if verbose:
            print("loss:", r['loss'])
    pred = torch.argmax(pred, -1)
    r['acc'] = accuracy_score(gold, pred)
    _, _, r['maF1'], _ = precision_recall_fscore_support(gold, pred, average="macro")
    _, _, r['miF1'], _ = precision_recall_fscore_support(gold, pred, average="micro")
    if verbose:
        print("accuracy:", r['acc'])
        print("micro:", r['miF1'], "macro", r['maF1'])
    if report:
        print(classification_report(gold, pred))
    if con_mat:
        print(confusion_matrix(gold, pred))
    return r

