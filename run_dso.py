import os
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from tqdm import trange

class Symbol(object):
    """XYZ"""

    def __init__(self, name, arity, function=None):

        self.name = name
        self.arity = arity
        self.fn = function

    def __str__(self):
        return self.name


class Expression(object):
    """XYZ"""

    def __init__(self):
        
        self.symbols = []
        self.likelihood = 1.0
        self.entropy = 0.0

        self.open_syms = 1
        self.par_sym = None
        self.sib_sym = None

    def __getitem__(self, val):
        return self.symbols[val]

    def __len__(self):
        return len(self.symbols)

    def __str__(self):
        return self.translate(self.symbols)[0]

    def __call__(self, data):
        return self.evaluate(self.symbols, data)[0]

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

    def evaluate(self, syms, data):

        sym = syms[0]
        syms = syms[1:]

        ar = sym.arity
        fn = sym.fn

        if ar == 0:
            return data[sym.name], syms
        elif ar == 1:
            arg1, syms = self.evaluate(syms, data)
            return fn(arg1), syms
        elif ar == 2:
            arg1, syms = self.evaluate(syms, data)
            arg2, syms = self.evaluate(syms, data)
            return fn(arg1, arg2), syms
        else:
            raise NotImplementedError()

    def add(self, sym, prob, step_entropy):

        self.symbols.append(sym)
        self.likelihood *= prob
        self.entropy += step_entropy
        
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
               
    def __call__(self, probs, expr, pool):

        constr = torch.ones_like(probs)

        if self.min_len is not None:

            open_syms, _, _ = expr.get_status()

            if open_syms == 1 and len(expr) < self.min_len:

                for s, sym in enumerate(pool):
                    if sym.arity == 0: constr[s] = 0.0

        if self.max_len is not None:

            open_syms, _, _ = expr.get_status()

            max_arity = self.max_len - len(expr) - open_syms

            for s, sym in enumerate(pool):
                if sym.arity > max_arity: constr[s] = 0.0

        # apply constraints
        probs = probs * constr

        # normalize
        probs = probs / probs.sum()
                
        return probs


class Reward(object):
    """XYZ"""

    def __init__(self, in_data, target_data, objective="NRMSE", in_vars=None, target_var=None, eval_type="str"):

        # get variable names
        if in_vars is None:
            self.in_vars = [f"x{i}" for i in range(in_data.shape[1])]
        else:
            self.in_vars = in_vars

        if target_var is None:
            self.target_var = "y"
        else:
            self.target_var

        # create data dict
        self.data = {self.in_vars[i]: in_data[:,i] for i in range(in_data.shape[1])}
        self.data[self.target_var] = target_data

        # get objective and set evaluation type
        self.objective = objective
        self.eval_type = eval_type

    def __call__(self, batch):
        if isinstance(batch, Expression):
            return self.evaluate(batch)
        else:
            return [self.evaluate(expr) for expr in batch]

    def evaluate(self, expr):

        if self.eval_type == "str":
            if self.objective == "NRMSE":
                eval_fun = f"1 / np.std({self.target_var}) * np.sqrt(np.mean(({expr}-{self.target_var})**2))"
            else:
                eval_fun = self.objective

            reward = eval(eval_fun, {'np': np}, self.data)
            reward = 1 / (1 + reward)

        else:
            if self.objective == "NRMSE":
                reward = 1 / np.std(self.data[self.target_var]) * np.sqrt(np.mean((expr(self.data) - self.data[self.target_var])**2))
                reward = 1 / (1 + reward)
            else:
                raise NotImplementedError()

        return reward


class PolicyGradient():
    """XYZ"""

    def __init__(self, algorithm="REINFORCE", entropy_lambda=None):

        self.algorithm = algorithm
        self.entropy_lambda = entropy_lambda

    def __call__(self, batch, rewards):
        return self.evaluate(batch, rewards)

    def evaluate(self, batch, rewards):

        if self.algorithm == "REINFORCE":
            likelihoods = torch.cat([expr.likelihood.unsqueeze(0) for expr in batch])
            rewards = torch.Tensor(rewards)
            loss = -1 * (rewards * torch.log(likelihoods)).mean()
        else:
            raise NotImplementedError()

        if self.entropy_lambda:
            entropies = torch.cat([expr.entropy.unsqueeze(0) for expr in batch])                 # TODO: entropy of expression as sum or mean of individual entropies?
            loss += -1 * self.entropy_lambda * entropies.mean()

        return loss


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

        in_encoding = torch.zeros(self.in_size)                     # TODO: explicit empty symbol encoding?

        if par_sym is not None:
            in_encoding[self.pool.index(par_sym)] = 1.0

        if sib_sym is not None:
            in_encoding[len(self.pool) + self.pool.index(sib_sym)] = 1.0

        return in_encoding

    def get_probabilities(self, in_encoding, state=None):

        h, c = self.rnn(in_encoding, state)
        out = self.linear(h)
        probs = F.softmax(out, dim=0)

        return probs, (h, c)

    def get_symbol(self, probs):
        
        sym_idx = torch.multinomial(probs, 1)[0]

        return self.pool[sym_idx], probs[sym_idx]

    def get_entropy(self, probs):

        log_safety = (probs.detach() == 0.0) * 1.0

        return -1 * (probs * torch.log(probs + log_safety)).sum()

    def forward(self):

        expr = Expression()
        open_syms, par_sym, sib_sym = expr.get_status()
        state = None

        while open_syms:
            in_encoding = self.get_input(par_sym, sib_sym)
            probs, state = self.get_probabilities(in_encoding, state)

            if self.constraints:
                probs = self.constraints(probs, expr, self.pool)
            
            sym, prob = self.get_symbol(probs)
            step_entropy = self.get_entropy(probs)
            open_syms, par_sym, sib_sym = expr.add(sym, prob, step_entropy)

        return expr

        # make batches work?!


if __name__ == "__main__":

    torch.manual_seed(0)

    # define hyperparameters
    hid_size = 32
    min_len = 4
    max_len = 20

    epochs = 50
    batch_size = 200
    learning_rate = 1e-3
    entropy_lambda = 5e-3

    # define symbol pool
    plus = Symbol("+", 2, function=np.add)
    minus = Symbol("-", 2, function=np.subtract)
    times = Symbol("*", 2, function=np.multiply)

    sin = Symbol("np.sin", 1, function=np.sin)
    cos = Symbol("np.cos", 1, function=np.cos)

    x_sym = Symbol("x0", 0)

    pool = [plus, minus, times, sin, cos, x_sym]

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

    X_train = data[:,:-1]
    y_train = data[:,-1]

    # define reward function
    reward = Reward(X_train, y_train, objective="NRMSE", eval_type="fn")

    # define policy gradient
    policy_grad = PolicyGradient("REINFORCE", entropy_lambda=entropy_lambda)

    # define optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # run training
    losses = []
    max_reward = 0.0
    best_expr = None

    t = trange(epochs, desc="Epoch")
    for epoch in t:

        optimizer.zero_grad()

        batch = [model() for _ in range(batch_size)]

        rewards = reward(batch)

        loss = policy_grad(batch, rewards)

        loss.backward()

        optimizer.step()

        losses.append(loss.item())
        
        t.set_postfix({"loss": f"{loss.item():.2f}"})

        r = np.max(rewards)
        if r > max_reward:
            max_reward = r
            best_expr = batch[rewards.index(r)]

    print(max_reward)
    print(best_expr)