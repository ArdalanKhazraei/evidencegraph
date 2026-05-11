
Evidence graphs for parsing argumentation structure
===================================================

[![Build Status](https://github.com/peldszus/evidencegraph/actions/workflows/workflow.yaml/badge.svg?branch=master)](https://github.com/peldszus/evidencegraph/actions)
[![codecov](https://codecov.io/gh/peldszus/evidencegraph/branch/master/graph/badge.svg)](https://codecov.io/gh/peldszus/evidencegraph)
[![GitHub](https://img.shields.io/github/license/peldszus/evidencegraph)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/ambv/black)



## Foreword

The present repository is a fork of the [evidencegraph argumentation parser](https://github.com/peldszus/evidencegraph) repository to present the extension I developed for my master's thesis.

It contains the original evidence graph parser as well as a version that uses features from a pretrained model instead of the surface linguistic features of the original, as well as neural classifiers instead of SGD lin-log regression classifiers as in the original.

The pretrained models used for the features were the [base](https://huggingface.co/cross-encoder/nli-deberta-v3-base) and [large](https://huggingface.co/cross-encoder/nli-deberta-v3-large) versions of NLI-DeBERTa-v3.

Below we first repeat instructions from the original repository, and subsequently our own steps for setting up the environment and using the pretrained model version.

# README of the original evidence graphs



## About

[This](https://github.com/peldszus/evidencegraph)  repository holds the code of the Evidence Graph model, a model for parsing the argumentation structure of text.

It basically is a re-implementation of the model presented first in [(1)](#references). Most work was done 2016-2017. It was used in the experiments of [(2)](#references), [(3)](#references) and [(4)](#references).


## Prerequisites

This code runs in Python 3.8. It is recommended to install it in a separate virtual environment. Here are installation instructions for an Ubuntu 18.04 linux:

```sh
# basics
sudo apt install python3.8-dev
# for lxml
sudo apt install libxml2-dev libxslt1-dev
# for matplotlib
sudo apt install libpng-dev libfreetype6-dev
# for graph plotting
sudo apt install graphviz
```


## Setup environment

Install all required python libaries in the environment and download the language models required by the spacy library.

    make install-requirements
    make download-spacy-data-de
    make download-spacy-data-en

Furthermore, several microtext corpora required for the experiments can be downloaded with:

    make download-corpora


## Test

Make sure all the tests pass.

    make test


## Run a minimal experiment

Run a (shortened and simplified) minimal experiment, to see that everything is working:

    env/bin/python src/experiments/run_minimal.py --corpus m112en

You should (see last lines of the output) get an average macro F1 of the *base classifiers* similar to:  
  (cc ~= 0.82, ro ~= 0.75, fu ~= 0.74, at ~= 0.72).

Evaluate the results, which have been written to `data/`:

    env/bin/python src/experiments/eval_minimal.py --corpus m112en

You should (see first lines of the output) get an average macro F1 for the *decoded results* similar to:  
  (cc ~= 0.86, ro ~= 0.74, fu ~= 0.76, at ~= 0.71).


## Replicate published results

Adjust run_minimal.py:
* Remove the line `folds = folds[:5]` in order to run all 50 train/test splits.
* In the experimental conditions, set `optimize` to `True` so that the local model's hyperparameters are optimized.
* In the experimental conditions, set `optimize_weights` to `True` so that the global model's hyperparameters are optimized.

For more details, see the actual experiment definitions in `src/experiments`.

Note that the results published in the papers were obtained using the Python 2 version of this code base. With the migration to Python 3 and various updated dependencies, the scores differ slightly. To reproduce the exact published scores, you will need to run version v0.4.0 of this code base.


## Reusing / extending components of the library

### Use the same features for a new language

Load a spacy nlp for the desired language and pass it together with a connective lexicon to the TextFeatures.

```python
from evidencegraph.features_text import TextFeatures
from evidencegraph.classifiers import EvidenceGraphClassifier

my_features = TextFeatures(
    nlp=spacy.load("klingon"),
    connectives={}, # add a connective lexicon here
    feature_set=TextFeatures.F_SET_ALL_BUT_VECTORS
)
clf = EvidenceGraphClassifier(
    my_features.feature_function_segments,
    my_features.feature_function_segmentpairs
)
```

### Use a custom base classifier

Derive a custom base classifier class (stick to the interface) and pass this class to the EvidenceGraphClassifier.

```python
from evidencegraph.classifiers import BaseClassifier

class MyBaseClassifier(BaseClassifier):
    # do something different here
    pass

clf = EvidenceGraphClassifier(
    my_features.feature_function_segments,
    my_features.feature_function_segmentpairs,
    base_classifier_class=MyBaseClassifier
)
```

### Load a custom corpus

Simply load a folder containing argument graph xml files into a GraphCorpus.

```python
from evidencegraph.corpus import GraphCorpus

corpus = GraphCorpus()
corpus.load("path/to/my/folder")
texts, trees = corpus.segments_trees()
```


## References

1) [Joint prediction in MST-style discourse parsing for argumentation mining](https://aclweb.org/anthology/D/D15/D15-1110.pdf)  
   Andreas Peldszus, Manfred Stede.  
   In: Proceedings of the 2015 Conference on Empirical Methods in Natural Language  Processing (EMNLP), Portugal, Lisbon, September 2015.

2) [Automatic recognition of argumentation structure in short monological texts](https://publishup.uni-potsdam.de/files/42144/diss_peldszus.pdf)  
   Andreas Peldszus.  
   Ph.D. thesis, Universität Potsdam, 2018.

3) [Comparing decoding mechanisms for parsing argumentative structures](https://content.iospress.com/download/argument-and-computation/aac033?id=argument-and-computation%2Faac033)  
   Stergos Afantenos, Andreas Peldszus, Manfred Stede.  
   In: Argument & Computation, Volume 9, Issue 3, 2018, Pages 177-192.

4) [More or less controlled elicitation of argumentative text: Enlarging a microtext corpus via crowdsourcing](http://www.aclweb.org/anthology/W/W18/W18-5218.pdf)  
   Maria Skeppstedt, Andreas Peldszus, Manfred Stede.  
   In: Proceedings of the 5th Workshop on Argument Mining. EMNLP 2018, Belgium, Brussels, November 2018.

# Instructions for the neural pretrained-features version

## Installations

One thing omitted from the original README's Prerequisites section is probably that one may also create a virtual environment by running the command below which is also present in the original Makefile, but for which we have more specifically identified 3.8 as the python version, as the user probably uses a newer version of python3 as their default.

	make virtualenv

Additional packages have been added to the requirements file, so running `make install-requirements` in the above instruction for the installation of the regular evidencegraph parser should have already covered the requirements for the extension version.

## Folder structure/Directory tree

I have simply added the main scripts that run my model in the top-level directory of the repository. Below you will find the additional directories I have added to the original evidencegraph parser's repository. The function and role of each subdirectory will be described in the corresponding instruction below. All subdirectories to which the scripts will add files not present in this repository contain a `.gitignore` file.

```
. 
 |
 |_ folds/
 |    |_ shuffle_folds_2.txt
 |    |_ shuffle_folds_c.txt
 |
 |_ evaluations/
 |    |
 |    |_ edge/
 |
 |_ attention_features/
 |    |
 |    |_ large/
 |
 |_ trained/
      |
      |_ at_ablation/
      |
      |_ layer/
      |    |_ cc/
      |    |_ at/
      |    |_ fu/
      |    |_ ro/
      |
      |_ models/
      |    |
      |    |_ combined/
      |    |    |_ eg/
      |    |    |_ pt_att4/
      |    |    |_ pt_att4_large/
      |    |
      |    |_ Part1/
      |    |    |_ eg/
      |    |    |_ pt_att4/
      |    |    |_ pt_att4_large/
      |    |
      |    |_ Part2/
      |         |_ eg/
      |         |_ pt_att4/
      |         |_ pt_att4_large/
      |
      |_ weight_opt/
           |
           |_ combined/
           |    |_ eg/
           |    |_ pt_att4/
           |    |_ pt_att4_large/
           |
           |_ Part1/
           |    |_ eg/
           |    |_ pt_att4/
           |    |_ pt_att4_large/
           |
           |_ Part2/
                |_ eg/
                |_ pt_att4/
                |_ pt_att4_large/
```

## Setup

### Train-test folds for Part 2 of the corpus and the combined corpus

The original evidencegraph parser provides train-test folds only for [Part1](https://github.com/peldszus/arg-microtexts) of the arg-microtexts corpus. I therefore generated my own folds for [Part2](https://github.com/discourse-lab/arg-microtexts-part2) of the corpus, along with the corpus obtained from combining both.

The folds used for the experiments of my thesis have been included in the `folds/` directory located in the top-level of this repository, under the names `folds/shuffle_folds_2.txt` and `folds/shuffle_folds_c.txt` respectively.

That means, the user intending to replicate my thesis results with the same folds need not and should not generate new folds. However, the script I used to generate my own folds have also been provided in `generate_folds.py` in the top-level directory of this repository.

__Note:__ A perfect replication of my results is of course not possible due to the stochastic nature of neural nets.

### Generating your own folds

If the goal is not to repeat my experiments with the same folds, the following commands may be used to generate random folds for Part 2 of the arg-microtexts corpus, and the comined corpus of parts 1 and 2 after deleting the already existing folds.

	make generate-folds

This will not replace the folds that have already been placed in the `folds/` directory, so if you want to generate new ones, you first have to delete the folds inside the `folds/` folder before running the above command for them to be replaced.

### Compile attention features

Running the neural parser requires the attention features to already be stored in the `attention_features/` folder. Of course the files for them are too big to upload onto the repository. They are compiled by running the two commands for the base and large versions:

	make compile-attention-features
	make compile-attention-features-large

## Performance Experiments

### Training

The following commands are used to train the [original evidencegraph parser](https://github.com/peldszus/evidencegraph), the base version of my neural extension, and the large version of the neural extension to the parser on the combined arg-microtexts corpus containing both parts 1 and 2:

	make train-eg
<!-- -->
	make train-neural
<!-- -->
	make train-neural-large

These will save models for each fold in the `trained/models/combined/` directory.

To restrict to part 1 of the arg-microtexts corpus instead, adding `CORPUS=1` to these commands as so

	make train-eg CORPUS=1
<!-- -->
	make train-neural CORPUS=1
<!-- -->
	make train-neural-large CORPUS=1

will save models to `trained/models/Part1/`

Similarly for part 2 of the corpus to save to `trained/models/Part2/`:

	make train-eg CORPUS=2
<!-- -->
	make train-neural CORPUS=2
<!-- -->
	make train-neural-large CORPUS=2

### Weight optimization

The following commands are used to optimize the weights of models stored in  `trained/models/combined/` and store the optimized weights in `trained/weight_opt/combined` 

	make optimize-weights-eg
<!-- -->
	make optimize-weights-neural
<!-- -->
	make optimize-weights-neural-large

To restrict to part 1 of the arg-microtexts corpus instead, add `CORPUS=1` to these commands as so

	make optimize-weights-eg CORPUS=1
<!-- -->
	make optimize-weights-neural CORPUS=1
<!-- -->
	make optimize-weights-neural-large CORPUS=1

which reads saved models from `trained/models/Part1/` and stores weights to `trained/weight_opt/Part1/`.

Similarly for part 2 of the corpus reading from `trained/models/Part2/` and storing weights to `trained/weight_opt/Part2/`:

	make optimize-weights-eg CORPUS=2
<!-- -->
	make optimize-weights-neural CORPUS=2
<!-- -->
	make optimize-weights-neural-large CORPUS=2

### Evaluation

Run the commands below to carry out evaluations on the models and save performance scores of the base classifiers, the minimum spanning tree, and the weight-optimized minimum spanning tree on each of the four levels (attachment `at` , central claim `cc`, function `fu`, and role `ro`) to a single text file in `evaluations/` for each combination of parser $\in$  {original evidencegraph `eg`, base neural `base`, large neural `large`}, and corpus $\in$ {`c`(combined), `1`, and `2`}.

Combined corpus:

	make eval-eg
<!-- -->
	make eval-neural
<!-- -->
	make eval-neural-large

Part 1 of the corpus:

	make eval-eg CORPUS=1
<!-- -->
	make eval-neural CORPUS=1
<!-- -->
	make eval-neural-large CORPUS=1

Part 2 of the corpus:

	make eval-eg CORPUS=2
<!-- -->
	make eval-neural CORPUS=2
<!-- -->
	make eval-neural-large CORPUS=2


## Ablations

### Layer ablations

Run the following commands to perform the layer ablation experiments described in my thesis for any given `$(LEVEL)` which creates for each layer number `$(LAYER)` of the pretrained model a directory `trained/layer/$(LEVEL)/$(LAYER)/`, stores single-layer models and a text file containing the performance metrics right there in the same folder.

For a single level `$(level)` $\in$ {`at`, `cc`, `fu`, `ro`}:

	make layer-ablation-single LEVEL=$(LEVEL)

And to run for all levels:

	make layer-ablation-all

which basically runs

	make layer-ablation-single LEVEL=at;
	make layer-ablation-single LEVEL=cc;
	make layer-ablation-single LEVEL=fu;
	make layer-ablation-single LEVEL=ro;

### Attention features ablation

Run the following command to perform attention ablation, and store results for factors `{2, 4, 8}` in directory `trained/at_ablation/`:

	make attention-ablation

## Edge scores comparison

The following commands carry out evaluations of edge scores defined in my thesis on the models and save performance scores of the base classifiers, the minimum spanning tree, and the weight-optimized minimum spanning tree on each of the four metrics (attachment `at` , node-wise function edge scores `nfu`, edge-wise function edge scores `efu`, and edge role scores `ero`) to a single text file in directory `evaluations/edge/` for each combination of parser $\in$  {original evidencegraph `eg`, base neural `base`, large neural `large`}, and corpus $\in$ {`c`(combined), `1`, and `2`}.

Combined corpus:

	make edge-eg
<!-- -->
	make edge-neural
<!-- -->
	make edge-neural-large

Part 1 of the corpus:

	make edge-eg CORPUS=1
<!-- -->
	make edge-neural CORPUS=1
<!-- -->
	make edge-neural-large CORPUS=1

Part 2 of the corpus:

	make edge-eg CORPUS=2
<!-- -->
	make edge-neural CORPUS=2
<!-- -->
	make edge-neural-large CORPUS=2


