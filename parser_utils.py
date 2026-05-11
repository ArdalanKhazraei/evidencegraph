
from prep import texts1, trees1, texts2, trees2

from evidencegraph.folds import get_static_folds

def shuffle_folds(c):
    with open('folds/shuffle_folds_'+c+'.txt', 'r') as f:
        folds = eval(f.read())
    texts = eval("texts"+c) if c != 'c' else {**texts1, **texts2}
    tids = sorted(texts)
    splits = []
    for i, test_tids in enumerate(folds):
        assert set(test_tids).issubset(tids)
        train_tids = [tid for tid in tids if tid not in test_tids]
        splits.append((train_tids, test_tids, i))
    return splits

