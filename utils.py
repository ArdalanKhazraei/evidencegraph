import torch

def sl2i(tokenizer, sent_list):
    sep = tokenizer.sep_token
    sid = tokenizer.convert_tokens_to_ids(sep) # remove this!!
    sep = ' ' + sep + ' '
    txt = sep.join(sent_list)
    inputs = tokenizer(txt, return_tensors='pt')
    sent_lims = [0] + [i for i, t in enumerate(inputs['input_ids'][0]) if t == sid] # fix this!!
    lims = [(sent_lims[i-1]+1,sent_lims[i]+1) for i in range(1,len(sent_lims))]
    assert len(lims) == len(sent_list)
    return inputs, lims

def mask_s2v(inputs, lims, mask=set(), mask_cls=False):
    assert all(i in range(len(lims)) for i in mask)
    attention_mask = torch.ones(1, len(inputs['input_ids'][0]))
    if mask_cls:
        attention_mask[0, 0] = 0
    for s in mask:
        attention_mask[0, lims[s][0]:lims[s][1]] = 0
    return attention_mask

def masked_input(inputs, lims, mask=set(), mask_cls=False):
    masked_inputs = {k: inputs[k] for k in inputs}
    masked_inputs['attention_mask'] = mask_s2v(inputs, lims, mask, mask_cls)
    return masked_inputs

def msl2i(tokenizer, sent_list, mask=set(), mask_cls=False):
    inputs, lims = sl2i(tokenizer, sent_list)
    return masked_input(inputs, lims, mask, mask_cls)

def layers(model, inputs):
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True, output_attentions=True)
    h = outputs.hidden_states
    a = outputs.attentions
    return h, a

def attentions(model, inputs):
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)
    a = outputs.attentions
    return a

def hiddens(model, inputs):
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    h = outputs.hidden_states
    return h

def satt(al, hi, lims, i, j):
    if i < 0:
        if j < 0:
            return torch.sum(al[0,hi,0,0]).item()
        else:
            return torch.sum(al[0,hi,0,lims[j][0]:lims[j][1]]).item()
    else:
        
        if j < 0:
            return torch.sum(al[0,hi,lims[i][0]:lims[i][1],0]).item()
        else:
            return torch.sum(al[0,hi,lims[i][0]:lims[i][1],lims[j][0]:lims[j][1]]).item()

def semb(hl, lims, i):
    return torch.mean(hl[0,lims[i][0]:lims[i][1],:], dim=-2)

def pairs(n):
    r = []
    for i in range(n):
        for j in range(n):
            if j==i:
                continue
            r.append((i,j))
    return r

def single(model, inputs, lims, i):
    u = set(range(len(lims)))
    mi = masked_input(inputs, lims, u-{i})
    hi = hiddens(model, mi)
    return [semb(l,lims,i) for l in hi]

def double(model, inputs, lims, i, j):
    u = set(range(len(lims)))
    mij = masked_input(inputs, lims, u-{i,j})
    hij = hiddens(model, mij)
    return [semb(l,lims,i) for l in hij], [semb(l,lims,j) for l in hij]

def slemb(model, inputs, lims):
    r = {}
    h = hiddens(model, inputs)
    for i in range(len(lims)):
        r[((), i)] = [semb(l,lims,i) for l in h]
        r[((i,), i)] = single(model,inputs,lims,i)
        for j in range(i+1, len(lims)):
            siji, sijj = double(model,inputs,lims,i,j)
            r[(i,j),i] = siji
            r[(i,j),j] = sijj
            r[(j,i),i] = siji
            r[(j,i),j] = sijj
    return r

def features(tokenizer, model, sent_list):
    inputs, lims = sl2i(tokenizer, sent_list)
    h, a = layers(model, inputs)
    keys = pairs(len(lims))
    a_feats = {}
    for i,j in keys:
        a_feats[(i,j)] = [[satt(l,k,lims,i,j) for k in range(l.shape[1])] for l in a]
        a_feats[(i,j)] += [[satt(l,k,lims,j,i) for k in range(l.shape[1])] for l in a]
    h_feats = {}
    s = slemb(model, inputs, lims)
    hr = range(1,len(h))
    for p in keys:
        i,j = p
        di = [0]+ [s[(),i][l]-s[(i,),i][l] for l in hr]
        dj = [0]+ [s[(),j][l]-s[(j,),j][l] for l in hr]
        #diji = [0]+ [s[(),i][l]-s[(i,j),i][l] for l in hr]
        #dijj = [0]+ [s[(),j][l]-s[(i,j),j][l] for l in hr]
        diji = [0]+ [s[(i,j),i][l]-s[(i,),i][l] for l in hr]
        dijj = [0]+ [s[(i,j),j][l]-s[(j,),j][l] for l in hr]
        h_feats[p] = []
        h_feats[p].append([torch.dot(s[(),i][l],s[(),j][l]) for l in hr])
        h_feats[p].append([torch.dot(s[(),i][l],di[l]) for l in hr])
        h_feats[p].append([torch.dot(s[(),j][l],dj[l]) for l in hr])
        h_feats[p].append([torch.dot(s[(),i][l],dj[l]) for l in hr])
        h_feats[p].append([torch.dot(s[(),j][l],di[l]) for l in hr])
        h_feats[p].append([torch.dot(di[l],dj[l]) for l in hr])
        h_feats[p].append([torch.dot(s[(),i][l],diji[l]) for l in hr])
        h_feats[p].append([torch.dot(s[(),j][l],dijj[l]) for l in hr])
        h_feats[p].append([torch.dot(s[(),i][l],dijj[l]) for l in hr])
        h_feats[p].append([torch.dot(s[(),j][l],diji[l]) for l in hr])
        h_feats[p].append([torch.dot(diji[l],dijj[l]) for l in hr])
        h_feats[p] = [[t.item() for t in tup] for tup in (zip(*h_feats[p]))]
        h_feats[p] = [[torch.dot(s[(),i][0], s[(),j][0]).item()]] + h_feats[p]
    return a_feats, h_feats

