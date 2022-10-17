"""XYZ"""

import os
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# TODO:
# - Define:
# - Symbols
# - Expressions
# - Model (policy)
# - Constraints
# - Reward
# - Optimize constants
# - Evaluation via strings or functions?
# - Policy gradient
# - Entropy
# - Get data
# - Device


class Symbol(object):
    """XYZ"""

    def __init__(self, name, arity):

        self.name = name
        self.arity = arity

    def __str__(self):
        return self.name


class Expression(object):
    """XYZ"""

    def __init__(self):
        
        self.symbols = []
        self.likelihood = 1.0

        self.open_syms = 1
        self.par_sym = None
        self.sib_sym = None

    def __getitem__(self, val):
        return self.symbols[val]

    def __len__(self):
        return len(self.symbols)

    def __str__(self):
        return self.translate(self.symbols)[0]

    def print_symbols(self):
        return [str(s) for s in self.symbols]

    def translate(self, syms):

        sym = syms[0]
        syms = syms[1:]

        mid = sym.name
        ar = sym.arity

        if ar == 0:
            return mid, syms
        elif ar == 1:
            right, syms = self.translate(syms)
            return f"{mid}({right})", syms
        elif ar == 2:
            left, syms = self.translate(syms)
            right, syms = self.translate(syms)
            return f"({left}{mid}{right})", syms
        else:
            raise NotImplementedError()

    def add(self, sym, prob):

        self.symbols.append(sym)
        self.likelihood *= prob
        
        self.open_syms += sym.arity - 1
        self.update_status()

        return self.get_status()

    def update_status(self):

        if self.open_syms == 0:
            self.par_sym = None
            self.sib_sym = None

        else:
            sym = self.symbols[-1]
            if sym.arity > 0:
                self.par_sym = sym
                self.sib_sym = None

            else:
                counter = -1
                idx = len(self.symbols) - 1
                
                while counter != 0:
                    idx -= 1
                    counter += self.symbols[idx].arity - 1

                self.par_sym = self.symbols[idx]
                self.sib_sym = self.symbols[idx+1]
            
    def get_status(self):
        return self.open_syms, self.par_sym, self.sib_sym


class Constraints(object):
    """XYZ"""

    def __init__(self, min_len=None, max_len=None):

        self.min_len = min_len
        self.max_len = max_len
               
    def __call__(self, pred, expr, pool):

        if self.min_len is not None:

            open_syms, _, _ = expr.get_status()

            if open_syms == 1 and len(expr) < self.min_len:

                for s, sym in enumerate(pool):
                    if sym.arity == 0: pred[s] = 0.0

        if self.max_len is not None:

            open_syms, _, _ = expr.get_status()

            max_arity = self.max_len - len(expr) - open_syms

            for s, sym in enumerate(pool):
                if sym.arity > max_arity: pred[s] = 0.0

        pred = pred / pred.sum()
                
        return pred


# class Reward(object):
#     """XYZ"""

#     def __init__(self, objective="NRMSE"):

#         self.objective = objective
            

class DSO(nn.Module):
    """XYZ"""

    def __init__(self, pool, hid_size, constraints=None):
        super().__init__()

        self.pool = pool

        self.in_size = 2 * len(self.pool)                       # TODO: parents cannot be terminal symbols - change get_input
        self.hid_size = hid_size
        self.out_size = len(self.pool)
        
        self.rnn = nn.LSTMCell(self.in_size, hid_size) 
        
        self.linear = nn.Linear(hid_size, self.out_size)

        self.constraints = constraints
          
    def get_input(self, par_sym=None, sib_sym=None):

        in_data = torch.zeros(self.in_size)                     # TODO: explicit empty symbol encoding?

        if par_sym is not None:
            in_data[self.pool.index(par_sym)] = 1.0

        if sib_sym is not None:
            in_data[len(self.pool) + self.pool.index(sib_sym)] = 1.0

        return in_data

    def get_probabilities(self, in_data, state=None):

        h, c = self.rnn(in_data, state)
        out = self.linear(h)
        probs = F.softmax(out, dim=0)

        return probs, (h, c)

    def get_symbol(self, probs):
        
        sym_idx = torch.multinomial(probs, 1)

        return self.pool[sym_idx], probs[sym_idx]

    def forward(self):

        expr = Expression()
        open_syms, par_sym, sib_sym = expr.get_status()
        state = None

        while open_syms:
            in_data = self.get_input(par_sym, sib_sym)
            probs, state = self.get_probabilities(in_data, state)

            if self.constraints:
                probs = self.constraints(probs, expr, self.pool)
            
            sym, prob = self.get_symbol(probs)
            open_syms, par_sym, sib_sym = expr.add(sym, prob)

        return expr

        # make batches work?!


if __name__ == "__main__":

    torch.manual_seed(0)

    # define hyperparameters
    hid_size = 8
    min_len = 4
    max_len = 20

    epochs = 5
    batch_size = 2

    # define symbol pool
    plus = Symbol("+", 2)
    minus = Symbol("-", 2)
    times = Symbol("*", 2)

    sin = Symbol("sin", 1)
    cos = Symbol("cos", 1)

    x_sym = Symbol("x", 0)
    y_sym = Symbol("y", 0)

    pool = [plus, minus, times, sin, cos, x_sym, y_sym]

    # define constraints
    constraints = Constraints(min_len=min_len, max_len=max_len)

    # create model
    # TODO: hyperparam dict, loading, saving
    model = DSO(pool=pool, hid_size=hid_size, constraints=constraints)

    # load data
    data_path = "data"
    data_name = "Nguyen-1"
    data_ext = "csv"

    data = np.loadtxt(os.path.join(data_path, f"{data_name}.{data_ext}"), delimiter=',')

    X_train = torch.Tensor(data[:,:-1])
    y_train = torch.Tensor(data[:,-1])

    # define optimizer

    # define reward function

    # // RNN architectural hyperparameters.
    # "cell" : "lstm",
    # "num_layers" : 1,
    # "num_units" : 32,
    # "initializer" : "zeros",

    # // Optimizer hyperparameters.
    # "learning_rate" : 0.001,
    # "optimizer" : "adam",

    # // Entropy regularizer hyperparameters.
    # "entropy_weight" : 0.005,
    # "entropy_gamma" : 1.0,

    # run training
    for epoch in range(epochs):

        batch = [model() for _ in range(batch_size)]

        for expr in batch:
            print(expr.print_symbols())
            print(expr)