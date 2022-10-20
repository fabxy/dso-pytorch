# Deep Symbolic Optimization in PyTorch

The paper "Deep Symbolic Regression: Recovering Mathematical Expressions from Data via Risk-Seeking Policy Gradients" by [Petersen et al. (2021)](https://arxiv.org/abs/1912.04871) is great.

The [available code](https://github.com/brendenpetersen/deep-symbolic-optimization) is long and in TensorFlow.

We don't like TensorFlow and we don't want to read a lot of code.

This project attempts to implement the above stated paper in PyTorch.

This is WORK-IN-PROGRESS until stated otherwise.

Open TODOs:

* Implement different policy gradients

* Sample batches in parallel

* Explore entropy regularization further

* Implement more constraints

* Optimize constants

* Improve one-hot-encoding (parent nodes, empty symbol)

* Train on GPU