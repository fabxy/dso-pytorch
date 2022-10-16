"""XYZ"""

import os
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

# torch.manual_seed(0)


class Symbol(object):
    """XYZ"""

    def __init__(self, name, arity):

        self.name = name
        self.arity = arity

    def __repr__(self):                 # TODO: Do we want this?
        return self.name


class Expression(object):
    """XYZ"""

    def __init__(self):
        
        self.symbols = []
        self.likelihood = 1.0
        self.open_syms = 1

    def add(self, sym, prob):

        self.symbols.append(sym)
        self.likelihood *= prob
        self.open_syms += sym.arity - 1

    def get_status(self):

        if self.open_syms == 0:
            par_sym = None
            sib_sym = None

        else:
            sym = self.symbols[-1]
            if sym.arity > 0:
                par_sym = sym
                sib_sym = None
            
            else:
                c = -1
                i = len(self.symbols) - 1
                
                while c != 0:
                    i -= 1
                    c += self.symbols[i].arity - 1

                par_sym = self.symbols[i]
                sib_sym = self.symbols[i+1]
            
        return self.open_syms, par_sym, sib_sym


class DSOnet(nn.Module):
    """XYZ"""

    def __init__(self, pool, hid_size):
        super().__init__()

        self.pool = pool

        self.in_size = 2 * len(self.pool)                       # TODO: parents cannot be terminal symbols - change get_input
        self.hid_size = hid_size
        self.out_size = len(self.pool)
        
        self.rnn = nn.LSTMCell(self.in_size, hid_size) 
        
        self.linear = nn.Linear(hid_size, self.out_size)

        # max sequence length
        # hidden size same as pool length?

    
    def get_input(self, par_sym=None, sib_sym=None):

        in_data = torch.zeros(self.in_size)                     # TODO: explicit empty symbol encoding?

        if par_sym is not None:
            in_data[self.pool.index(par_sym)] = 1.0

        if sib_sym is not None:
            in_data[len(self.pool) + self.pool.index(sib_sym)] = 1.0

        return in_data


    def get_symbol(self, in_data, state=None):

        h, c = self.rnn(in_data, state)

        probs = F.softmax(self.linear(h), dim=0)

        sym_idx = torch.multinomial(probs, 1)

        return self.pool[sym_idx], probs[sym_idx], (h, c)


    def forward(self):

        expr = Expression()

        in_data = self.get_input()
        sym, prob, state = self.get_symbol(in_data)

        expr.add(sym, prob)
        open_syms, par_sym, sib_sym = expr.get_status()

        while open_syms:

            in_data = self.get_input(par_sym, sib_sym)
            sym, prob, state = self.get_symbol(in_data, state)

            expr.add(sym, prob)
            open_syms, par_sym, sib_sym = expr.get_status()

        return expr

        # initialize RNN cell state to zero
        # run cell
        # get probs
        # apply constraints
        # apply softmax
        # sample symbol
        # add to expression
        # finish or get next parent and sibling

        # make batches work?!


if __name__ == "__main__":

    plus = Symbol("+", 2)
    minus = Symbol("-", 2)
    times = Symbol("*", 2)

    x_sym = Symbol("x", 0)
    y_sym = Symbol("y", 0)

    pool = [plus, minus, times, x_sym, y_sym]

    model = DSOnet(pool, 6)

    expr = model()

    print([s.name for s in expr.symbols])