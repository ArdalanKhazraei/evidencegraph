
 # utils

def onehot(i, n):
    r = [0]*n
    r[i] = 1
    return r

def uniformity(array, fct):
    fs = set(fct(item) for item in array)
    assert len(fs) == 1
    return fs.pop()

def confusion_matrix(golds, preds):
    n = uniformity((golds, preds), len)
    c = uniformity(preds, len)
    m = [[0]*c for _ in range(c)]
    for i in range(n):
        for j in range(c):
            m[golds[i]][j] += preds[i][j]
    return n, c, m

def proba_scores(golds, preds):
    n, l, cm = confusion_matrix(golds, preds)
    golds = [onehot(gold, l) if type(gold)==int else gold for gold in golds]
    gp = [sum(gold[i] for gold in golds) for i in range(l)]
    pp = [sum(pred[i] for pred in preds) for i in range(l)]
    bp = [sum(golds[j][i]*preds[j][i] for j in range(n)) for i in range(l)]
    p = [bp[i]/pp[i] if pp[i] else 1 for i in range(l)]
    r = [bp[i]/gp[i] if gp[i] else 1 for i in range(l)]
    f1 = [2*r[i]*p[i]/(r[i]+p[i]) if r[i]+p[i] else 0 for i in range(l)]
    macro = sum(f1)/l
    acc = sum(bp)/n
    return p, r, f1, macro, acc, cm

def class_scores(golds, preds):
    c = max(max(golds), max(preds)) + 1
    preds = [onehot(pred, c) for pred in preds]
    return proba_scores(golds, preds)

def scores(golds, preds):
    t = uniformity(preds, type)
    if t == int:
        return class_scores(golds, preds)
    elif hasattr(t, '__iter__'):
        return proba_scores(golds, preds)
    else:
        raise TypeError("undefined prediction type")

 # base

def att_scores(clf, txt):
    n = len(txt) if type(txt)==list else len(txt["node"])
    N = list(range(1, n+1))
    E = [(p, q) for p in N for q in N if q!=p]
    return dict(zip(E, clf.ensemble["at"].predict(txt, "proba")))

def node_fu_edge_scores(clf, txt):
    n = len(txt) if type(txt)==list else len(txt["node"])
    nd = dict(zip(range(1, 1+n), clf.ensemble["fu"].predict(txt, "proba")))
    return {(m, n): nd[m] for m in nd for n in nd if n!=m}

def edge_ro_scores(clf, txt):
    n = len(txt) if type(txt)==list else len(txt["node"])
    nd = dict(zip(range(1, 1+n), clf.ensemble["ro"].predict(txt, "proba")))
    return {(m, n): [nd[m][0]*nd[n][0]+nd[m][1]*nd[n][1], nd[m][0]*nd[n][1]+nd[m][1]*nd[n][0]] for m in nd for n in nd if n!=m}

 # mst

def att_dict(argtree):
    N = sorted(argtree.nodes())
    E = [(p, q) for p in N for q in N if q!=p]
    return dict(zip(E, argtree.get_at_vector()))

def node_fu_edge_dict(argtree):
    fu_dict = dict(zip(sorted(argtree.nodes()), argtree.get_fu_vector()))
    return {(m, n): fu_dict[m] for m in fu_dict for n in fu_dict if n!=m}

def edge_fu_edge_dict(argtree):
    fd = node_fu_edge_dict(argtree)
    ad = att_dict(argtree)
    assert fd.keys()==ad.keys()
    return {e: ad[e] and fd[e] for e in ad}

def dict_edge_fu(argtree):
    N = sorted(argtree.nodes())
    d = {(p, q): 0 for p in N for q in N if q!=p}
    for t in argtree.get_triples():
        d[tuple(t[:2])] = {'sup': 1, 'att': 2}[t[2]]
    return d

def edge_ro_dict(argtree):
    ro_dict = dict(zip(sorted(argtree.nodes()), argtree.get_ro_vector()))
    return {(m, n): (ro_dict[m]+ro_dict[n])%2 for m in ro_dict for n in ro_dict if n!=m}

 # F scores

def base_pred(clf, lvl, txt):
    if lvl == "at":
        return att_scores(clf, txt)
    elif lvl[-2:] == "fu":
        return node_fu_edge_scores(clf, txt)
    elif lvl == "ero":
        return edge_ro_scores(clf, txt)
    else:
        raise ValueError('lvl either "at", "nfu", "efu", or "ero"')

def mst2dict(clf, lvl, mst):
    prefix = {'at': 'att', 'nfu': 'node_fu_edge', 'efu': 'edge_fu_edge', 'ero': 'edge_ro'}
    return eval(prefix[lvl]+'_dict(mst)')

def mst_pred(clf, lvl, txt):
    mst = clf.predict(txt)
    return mst2dict(clf, lvl, mst)

def get_Fs(clf, lvl, texts_list, golds_list, base=False):
    preds = []
    golds = []
    for i in range(len(texts_list)):
        gold = mst2dict(clf, lvl, golds_list[i])
        get_pred = base_pred if base else mst_pred
        pred = get_pred(clf, lvl, texts_list[i])
        assert pred.keys() == gold.keys()
        golds += [gold[e] for e in sorted(gold)]
        preds += [pred[e] for e in sorted(gold)]
    _, _, _, maF1, acc, _ = scores(golds, preds)
    return golds, preds, maF1, acc



