
from utils import *

from prep import texts1, texts2


def hidden_features(tokenizer, model, sent_list):
    inputs, lims = sl2i(tokenizer, sent_list)
    h = hiddens(model, inputs)
    h_feats = []
    for l in h:
        h_feats.append({})
        h_feats[-1]['cls'] = l[0][0]
        for i in range(len(lims)):
            h_feats[-1][i+1] = l[0][lims[i][0]:lims[i][1]]
    return h_feats

def corpus_hiddens(corpus, large):
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    repo = 'cross-encoder/nli-deberta-v3-' + ('large' if large else 'base')
    tokenizer = AutoTokenizer.from_pretrained(repo)
    model = AutoModelForSequenceClassification.from_pretrained(repo)

    result = {}

    ctr = 0

    print("\ncompiling features from hidden layer embeddings\n")
    
    texts = eval("texts"+corpus) if corpus != 'c' else {**texts1, **texts2}

    for t in texts:
        ctr += 1
        print("  processing text  _ ", ctr,"/",len(texts), end='\r')
        result[t] = hidden_features(tokenizer, model, texts[t])
    
    print()
    print()
    return result


