

from NN_classifier import NNClassifier

import os, joblib

def get_dim(nn):
    linlayers = {int(name[2:]): module.in_features for name, module in nn.named_modules() if len(name) > 2 and name[:2] == "fc"}
    return linlayers[1]

def get_ls(models):
    nets = {model[0]: model[1][0] for model in models}
    ls = {lvl: get_dim(nn) for lvl, nn in nets.items()}
    assert set(ls.keys()) == set(levels)
    nodels = {d for l, d in ls.items() if l != "at"}
    assert len(nodels) == 1
    nodel = nodels.pop()
    edgel = ls["at"]
    return edgel, nodel

def load(path, verbose=True):
    if not os.path.isfile(path) or not os.access(path, os.R_OK):
        raise RuntimeError("Can't load model from file {:s}".format(path))
    models = joblib.load(path)
    edgel, nodel = get_ls(models)
    clf = NNClassifier(edgel, nodel, large=large)
    for lvl, triple in models:
        m = clf.ensemble[lvl].model
        m.nn, m.means, m.stds = triple
    if verbose:
        print("Loaded model {}".format(path))
    return clf

