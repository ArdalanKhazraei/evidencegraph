.PHONY: all virtualenv install-requirements download-spacy-data-de download-spacy-data-en test run-minimal-de run-minimal-en eval-minimal-de eval-minimal-en

VIRTUALENV_DIR=./env
CORPUS_DIR=./data/corpus

virtualenv:
	if [ ! -e ${VIRTUALENV_DIR}/bin/pip ]; then python3.8 -m venv ${VIRTUALENV_DIR}; fi

install-requirements: virtualenv
	${VIRTUALENV_DIR}/bin/pip install --upgrade pip
	${VIRTUALENV_DIR}/bin/pip install -r requirements.txt
	${VIRTUALENV_DIR}/bin/python setup.py develop
	${VIRTUALENV_DIR}/bin/pre-commit install

download-corpora:
	mkdir -p ${CORPUS_DIR}
	curl -o /tmp/arg-microtexts-1.zip -LO https://github.com/peldszus/arg-microtexts/archive/master.zip
	unzip -qq /tmp/arg-microtexts-1.zip -d ${CORPUS_DIR}
	curl -o /tmp/arg-microtexts-1-multi.zip -LO https://github.com/peldszus/arg-microtexts-multilayer/archive/master.zip
	unzip -qq /tmp/arg-microtexts-1-multi.zip -d ${CORPUS_DIR}
	curl -o /tmp/arg-microtexts-2.zip -LO https://github.com/discourse-lab/arg-microtexts-part2/archive/master.zip
	unzip -qq /tmp/arg-microtexts-2.zip -d ${CORPUS_DIR}

download-spacy-data-de:
	${VIRTUALENV_DIR}/bin/python -m spacy download de_core_news_lg

download-spacy-data-en:
	${VIRTUALENV_DIR}/bin/python -m spacy download en_core_web_lg

test:
	${VIRTUALENV_DIR}/bin/py.test -v --cov=src/evidencegraph --cov-report xml src test

run-minimal-en:
	stdbuf -o 0 ${VIRTUALENV_DIR}/bin/python src/experiments/run_minimal.py -c m112en | tee "data/m112en-test-adu-simple-noop|equal.log"

run-minimal-de:
	stdbuf -o 0 ${VIRTUALENV_DIR}/bin/python src/experiments/run_minimal.py -c m112de | tee "data/m112de-test-adu-simple-noop|equal.log"

eval-minimal-en:
	stdbuf -o 0 ${VIRTUALENV_DIR}/bin/python src/experiments/eval_minimal.py -c m112en | tee data/m112en-test-evaluation.log

eval-minimal-de:
	stdbuf -o 0 ${VIRTUALENV_DIR}/bin/python src/experiments/eval_minimal.py -c m112de | tee data/m112de-test-evaluation.log
	


############################################################################################################################################################

CORPUS = c

generate-folds:
	stdbuf -o 0 ${VIRTUALENV_DIR}/bin/python generate_folds.py

compile-attention-features:
	stdbuf -o 0 ${VIRTUALENV_DIR}/bin/python compile_attention_features.py --regular

compile-attention-features-large:
	stdbuf -o 0 ${VIRTUALENV_DIR}/bin/python compile_attention_features.py --large --regular

train-eg:
	stdbuf -o 0 ${VIRTUALENV_DIR}/bin/python train_eg.py --corpus $(CORPUS)
	
train-neural:
	stdbuf -o 0 ${VIRTUALENV_DIR}/bin/python train_neural.py --corpus $(CORPUS)
	
train-neural-large:
	stdbuf -o 0 ${VIRTUALENV_DIR}/bin/python train_neural.py --large --corpus $(CORPUS)
	
optimize-weights-eg:
	stdbuf -o 0 ${VIRTUALENV_DIR}/bin/python weight_opt_eg.py --corpus $(CORPUS)
	
optimize-weights-neural:
	stdbuf -o 0 ${VIRTUALENV_DIR}/bin/python weight_opt_neural.py --corpus $(CORPUS)
	
optimize-weights-neural-large:
	stdbuf -o 0 ${VIRTUALENV_DIR}/bin/python weight_opt_neural.py --large --corpus $(CORPUS)

eval-eg:
	stdbuf -o 0 ${VIRTUALENV_DIR}/bin/python eval_eg.py --corpus $(CORPUS) --eval_base --eval_mst --weight_opt

eval-neural:
	stdbuf -o 0 ${VIRTUALENV_DIR}/bin/python eval_neural.py --corpus $(CORPUS) --eval_base --eval_mst --weight_opt

eval-neural-large:
	stdbuf -o 0 ${VIRTUALENV_DIR}/bin/python eval_neural.py --large --corpus $(CORPUS) --eval_base --eval_mst --weight_opt




layer-ablation-single:
	stdbuf -o 0 ${VIRTUALENV_DIR}/bin/python layer.py --corpus $(CORPUS) --level $(LEVEL)

layer-ablation-all:
	$(MAKE) layer-ablation-single LEVEL=at;
	$(MAKE) layer-ablation-single LEVEL=cc;
	$(MAKE) layer-ablation-single LEVEL=fu;
	$(MAKE) layer-ablation-single LEVEL=ro;

attention-ablation:
	stdbuf -o 0 ${VIRTUALENV_DIR}/bin/python at_ablation.py

edge-eval-eg:
	stdbuf -o 0 ${VIRTUALENV_DIR}/bin/python edge_eg.py --corpus $(CORPUS) --eval_base --eval_mst --weight_opt

edge-eval-neural:
	stdbuf -o 0 ${VIRTUALENV_DIR}/bin/python edge_neural.py --corpus $(CORPUS) --eval_base --eval_mst --weight_opt

edge-eval-neural-large:
	stdbuf -o 0 ${VIRTUALENV_DIR}/bin/python edge_neural.py --corpus $(CORPUS) --large --eval_base --eval_mst --weight_opt







