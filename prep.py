
from evidencegraph.corpus import GraphCorpus
from evidencegraph.argtree import SIMPLE_RELATION_SET

path1 = "data/corpus/arg-microtexts-master/corpus/en/"
path2 = "data/corpus/arg-microtexts-part2-master/corpus/"

corpus = GraphCorpus()

corpus.load(path1)
texts1, trees1 = corpus.segments_trees("adu", SIMPLE_RELATION_SET)

corpus = GraphCorpus()

corpus.load(path2)
texts2, trees2 = corpus.segments_trees("adu", SIMPLE_RELATION_SET)
