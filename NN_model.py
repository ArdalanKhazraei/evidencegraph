
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.modules.module")

import os, joblib
import torch, numpy as np
from torch import nn
from NN_utils import evals, stats, normalize, batch_onehot


class FF(nn.Module):
    def __init__(self, n_in, n_hiddens, n_out, ns=0.1):
        super(FF, self).__init__()
        self.relu = nn.LeakyReLU(negative_slope=ns)
        if n_hiddens:
            self.fc1 = nn.Linear(n_in, n_hiddens[0])
            for i, n_hidden in enumerate(n_hiddens[:-1]):
                exec("self.fc"+str(i+2)+" = nn.Linear(n_hiddens[i], n_hiddens[i+1])")
            exec("self.fc"+str(len(n_hiddens)+1)+" = nn.Linear(n_hiddens[-1], n_out)")
        else:
            self.fc1 = nn.Linear(n_in, n_out)
        self.sm = nn.Softmax(dim=-1)
    
    def forward(self, x):
        for i, fc in enumerate(list(self.modules())[2:-1]):
            if i:
                x = self.relu(x)
            x = fc(x)
        return self.sm(x)

class model():
    def __init__(self, n_in, n_hiddens, n_out, ns=0.1):
        self.nn = FF(n_in, n_hiddens, n_out, ns)
        self.ns = [n_in] + n_hiddens + [n_out]
        self.oh = lambda fu: batchonehot(fu, n_out)
        self.means = None
        self.stds = None
    
    def train(self, X_train, y_train, num_epochs=20, lr=0.01, dr=5, convergence_tracking=False, X_eval=None, y_eval=None):
        assert type(X_train)== list and type(y_train) == list
        assert {type(f) for f in X_train} == {list}
        assert {type(l) for l in y_train} == {int}
        
        self.means, self.stds = stats(X_train)
        
        X_train = normalize(X_train, self.means, self.stds)
        y_train = torch.tensor(y_train)
        
        i0 = [i for i in range(len(y_train)) if y_train[i]==0]
        i1 = [i for i in range(len(y_train)) if y_train[i]==1]
        i2 = [i for i in range(len(y_train)) if y_train[i]==2]
        
        n0, n1, n2 = len(i0), len(i1), len(i2)
        n_out = self.ns[-1]
        assert n2 and n_out==3 or not n2 and n_out==2, "n2=" + str(n2) + " n_out=" + str(n_out)
        
        n = sum((n0,n1,n2))
        
        weights = [n/n0,n/n1,n/n2] if n2 else [n/n0,n/n1]
        self.weights = torch.tensor(weights)
        
        optimizer = torch.optim.SGD(self.nn.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda n: 2/np.sqrt(1 + n/dr))
        
        criterion = nn.CrossEntropyLoss(weight=self.weights, reduction='sum')
        
        es = {"train": [], "test": []} if convergence_tracking else None
        for epoch in range(1, 1+num_epochs):
            train_loss = 0
            print("  _ epoch", epoch, end='\r' if not convergence_tracking else '\n')
            batch = 10
            num_batches = len(X_train)//batch
            if len(X_train)%batch:
                num_batches += 1
            for _ in range(1):
                for i in range(num_batches):
                    optimizer.zero_grad()
                    out = self.nn(X_train[batch*i: batch*(i+1)].to(torch.float32))
                    loss = criterion(out, batch_onehot(y_train[batch*i: batch*(i+1)], n_out ))
                    train_loss += loss.item()/float(len(X_train))
                    loss.backward()
                    optimizer.step()
                    scheduler.step()
            if convergence_tracking:
                print("train metrics:")
                y_pred = self.predict(X_train)
                es["train"].append(evals(y_train, y_pred, con_mat=True, loss=True, weight=self.weights))
                print("eval metrics")
                y_pred = self.predict(X_eval)
                es["test"].append(evals(y_eval, y_pred, con_mat=True, loss=True, weight=self.weights))
        print()
        return es
    
    def predict(self, X_test):
        X_test = normalize(X_test, self.means, self.stds)
        return self.nn(X_test)
    
    def save(self, path, verbose=True):
        joblib.dump((self.nn, self.ns, self.means, self.stds), path, compress=1)
        if verbose:
            print("Saved model {}".format(path))
    
    def load(self, path, verbose=True):
        if not os.path.isfile(path) or not os.access(path, os.R_OK):
            raise RuntimeError("Can't load model from file {:s}".format(path))
        self.nn, self.ns, self.means, self.stds = joblib.load(path)
        self.oh = lambda fu: batchonehot(fu, self.ns[-1])
        if verbose:
            print("Loaded model {}".format(path))
        
        
        
