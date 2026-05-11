
from prep import texts1, texts2
import os, random

path2 = 'folds/shuffle_folds_2.txt'
pathc = 'folds/shuffle_folds_c.txt'

def shuffle_folds(array, num):
    l = list(array)
    random.shuffle(l)
    f = [l[(i*len(l))//num:((i+1)*len(l))//num] for i in range(num)]
    random.shuffle(f)
    return f

def iter_shuffle(array, num, iterations):
    l = []
    for _ in range(iterations):
        l += shuffle_folds(array, num)
    return l

if not os.path.isfile(path2):
    with open(path2, 'w') as f:
        f.write(str(iter_shuffle(texts2, 5, 10)))

if not os.path.isfile(pathc):
    with open(pathc, 'w') as f:
        f.write(str(iter_shuffle({**texts1, **texts2}, 5, 10)))
