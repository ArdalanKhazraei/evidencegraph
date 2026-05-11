
import os
from functools import partial
from itertools import permutations

import joblib
from numpy import mean, zeros, argmax
from sklearn.metrics import precision_recall_fscore_support
import torch

from evidencegraph.argtree import SIMPLE_RELATION_SET
from evidencegraph.decode import find_mst as find_mst
from evidencegraph.evidence_graph import EvidenceGraph
from evidencegraph.search import EvolutionarySearch
from evidencegraph.utils import foldsof

from NN_model import model


def label_function_cc(argtree):
    return argtree.get_cc_vector()


def label_function_ro(argtree):
    return argtree.get_ro_vector()


def label_function_fu(argtree):
    return argtree.get_fu_vector()


def label_function_at(argtree):
    return argtree.get_at_vector()


class BaseNNClassifier:

    def __init__(self, n_in, n_hiddens, n_out, feature_function, label_function, ns=0.1, epochs=20):
        self.feature_function = feature_function
        self.label_function = label_function
        self.model = model(n_in, n_hiddens, n_out, ns)
        self.epochs = epochs

    def train(self, in_data, gold_data):
        assert len(in_data) == len(gold_data)
        features = [
            feature
            for datum in in_data
            for feature in self.extract_features(datum)
        ]
        labels = [
            label
            for datum in gold_data
            for label in self.extract_labels(datum)
        ]
        self._train(features, labels)

    def train_optimize(self, in_data, gold_data, verbose=True):
        raise NotImplementedError

    def _train(self, features, labels, lr=0.01, dr=5):
        self.model.train(features, labels, num_epochs=self.epochs, lr=lr, dr=dr)

    def predict_collection(self, in_data, prediction_type="class"):
        features = [
            feature
            for datum in in_data
            for feature in self.extract_features(datum)
        ]
        return self._predict(features, prediction_type=prediction_type)

    def predict(self, in_datum, prediction_type="class"):
        features = self.extract_features(in_datum)
        return self._predict(features, prediction_type=prediction_type)

    def _predict(self, features, prediction_type="class"):
        if prediction_type == "proba":
            return self.model.predict(features).detach().numpy()
        elif prediction_type == "decision":
            return NotImplemented
        else:
            return argmax(self.model.predict(features).detach().numpy(), -1)

    def test(self, in_data, gold_data):
        predicted_labels = self.predict_collection(in_data)
        labels = [
            label
            for datum in gold_data
            for label in self.extract_labels(datum)
        ]
        return self._score(labels, predicted_labels)

    def _score(self, gold, pred):
        _, _, macro_f1, _ = precision_recall_fscore_support(
            gold, pred, average="macro", pos_label=None, warn_for=()
        )
        _, _, micro_f1, _ = precision_recall_fscore_support(
            gold, pred, average="micro", pos_label=None, warn_for=()
        )
        return macro_f1, micro_f1

    def extract_features(self, in_datum):
        return self.feature_function(in_datum)

    def extract_labels(self, gold_datum):
        return self.label_function(gold_datum)

class NNClassifier:
    def __init__(
        self,
        edge_features_length,
        node_features_length,
        optimize_weighting=False,
        relation_set=SIMPLE_RELATION_SET,
        large=False
    ):
        self.relation_set = relation_set
        self.edge_features_length = edge_features_length
        self.node_features_length = node_features_length
        levels = ["cc", "fu", "ro", "at"]
        sv = lambda i : [i[j] for j in sorted(i)]
        self.ensemble = {
            lvl : BaseNNClassifier(
                    edge_features_length if lvl=="at" else node_features_length,
                    (([600, 100, 20] if lvl == "att" else [1000, 300, 60] if lvl == "fu" else [1000, 300, 40])
                    if large else
                    ([80, 40, 20] if lvl == "att" else [300, 60] if lvl == "fu" else [300, 40])),
                    3 if lvl == "fu" else 2,
                    (lambda i: sv(i["edge"])) if lvl=="at" else (lambda i : sv(i["node"])),
                    eval("label_function_"+str(lvl)),
                    epochs = 30 if large else 20
                )
                for lvl in levels
            }
        self.optimize_weighting = optimize_weighting
        self.weighting = {level: 0.25 for level in self.ensemble}
        self.large = large

    def train(self, input_trees, output_trees):
        # train base classifiers
        for lvl, clf in self.ensemble.items():
            clf.train(input_trees, output_trees)
        # train meta classifier
        if self.optimize_weighting:
            self.train_metaclassifier(input_trees, output_trees)

    def train_metaclassifier(self, input_trees, output_trees):
        if self.optimize_weighting == "inner_cv":
            # predict all items in trainingset as unseen via inner CV
            egs = []
            for (train_X, train_y), (test_X, _) in foldsof(
                input_trees, output_trees
            ):
                egclf = NNClassifier(
                    edge_features_length=self.edge_features_length,
                    node_features_length=self.node_features_length,
                    optimize_weighting=False,
                    relation_set=self.relation_set,
                    large=self.large,
                )
                egclf.train(train_X, train_y)
                fold_egs = [egclf._predict_evidence_graph(t) for t in test_X]
                del egclf
                egs.extend(fold_egs)
        else:
            egs = [self._predict_evidence_graph(t) for t in input_trees]

        def weighting_dict(w1, w2, w3, w4):
            return {"cc": w1, "ro": w2, "fu": w3, "at": w4}

        def score_weighting(w1, w2, w3, w4, items=None):
            weighting = weighting_dict(w1, w2, w3, w4)
            scores = []
            for eg, gold in items:
                mst = self._decode(eg, weighting=weighting)
                scores.append(self.score(mst, gold))
            return mean(scores)

        callback = partial(score_weighting, items=list(zip(egs, output_trees)))
        search = EvolutionarySearch(callback, n_to_start_with=20)
        search.search(verbose=True)
        search.report()
        self.weighting = weighting_dict(*search.get_best())

    def predict_collection(self, input_trees):
        for tree in input_trees:
            yield self.predict(tree)

    def _predict(self, input_tree, prediction_type="proba"):
        # predict with base classifiers
        predictions = {}
        for level, clf in self.ensemble.items():
            predictions[level] = clf.predict(
                input_tree, prediction_type=prediction_type
            )

        def itemize(level, preds):
            if level == "at":
                return dict(zip(input_tree["edge"], preds))
            else:
                return dict(zip(input_tree["node"], preds))

        itemized_predictions = {
            level: itemize(level, preds)
            for level, preds in predictions.items()
        }
        return itemized_predictions

    def predict_decisions(self, input_tree):
        return self._predict(input_tree, prediction_type="decision")

    def predict(self, input_tree):
        eg = self._predict_evidence_graph(input_tree)
        mst = self._decode(eg)
        return mst

    def _predict_evidence_graph(self, input_tree):
        itemized_predictions = self._predict(
            input_tree, prediction_type="proba"
        )
        eg = self._build_evidence_graph(itemized_predictions)
        return eg

    def _decode(self, evidence_graph, weighting=None):
        if weighting is None:
            weighting = self.weighting
        weg = evidence_graph.get_weighted_evidence_graph(weights=weighting)
        mst = find_mst(weg)
        mst.relation_set = self.relation_set
        return mst

    def score(self, input_tree, output_tree):
        pred = input_tree.get_vector()
        gold = output_tree.get_vector()
        score = 1.0
        for level in self.ensemble:
            _, _, macro_f1, _ = precision_recall_fscore_support(
                gold[level],
                pred[level],
                average="macro",
                pos_label=None,
                warn_for=(),
            )
            score *= macro_f1
        return score

    def _build_evidence_graph(self, itemized_predictions):
        eg = EvidenceGraph(weight_ids=["cc", "ro", "fu", "at"])
        map_fu_to_vec = self.relation_set.map_function_to_vector
        for (source, target), p_at in itemized_predictions["at"].items():
            p_cc = itemized_predictions["cc"][source]
            p_ro_source = itemized_predictions["ro"][source]
            p_ro_target = itemized_predictions["ro"][target]
            p_fu = itemized_predictions["fu"][source]
            for func_type in map_fu_to_vec:
                if func_type in ["cc", "unknown"]:
                    continue
                # probability of attachment
                at_weight = p_at[1]
                # probability of not being the central claim
                cc_weight = p_cc[0]
                # probability of role switch
                if func_type in self.relation_set.functions_inverting_role:
                    ro_weight = (
                        p_ro_source[0] * p_ro_target[1]
                        + p_ro_source[1] * p_ro_target[0]
                    )
                else:
                    ro_weight = (
                        p_ro_source[0] * p_ro_target[0]
                        + p_ro_source[1] * p_ro_target[1]
                    )
                # probability of edge type
                try:
                    fu_weight = p_fu[map_fu_to_vec[func_type]]
                except IndexError:
                    print(p_fu)
                    print(map_fu_to_vec)
                    print(func_type)
                eg.add_edge(
                    source,
                    target,
                    type=func_type,
                    cc=cc_weight,
                    ro=ro_weight,
                    fu=fu_weight,
                    at=at_weight,
                )
        return eg

    def save(self, path, verbose=True):
        """Save ensemble of base classifiers."""
        models = [
            (lvl, (m.nn, m.means, m.stds)) for lvl in sorted(self.ensemble) for m in {self.ensemble[lvl].model}
        ]
        joblib.dump(models, path, compress=1)
        if verbose:
            print("Saved model {}".format(path))

    def load(self, path, verbose=True):
        """Load an object (typically a classifier model) using joblib."""
        if not os.path.isfile(path) or not os.access(path, os.R_OK):
            raise RuntimeError("Can't load model from file {:s}".format(path))
        models = joblib.load(path)
        for lvl, triple in models:
            m = self.ensemble[lvl].model
            m.nn, m.means, m.stds = triple
        if verbose:
            print("Loaded model {}".format(path))
