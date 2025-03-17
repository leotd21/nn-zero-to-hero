import sys
import os
import torch

from micrograd.engine import Value
from micrograd.utils import draw_dot
from micrograd.nn import Neuron, Layer, MLP

def test_micrograd_land():
    print("Running micrograd_test.py")

    # initialize values
    x1 = Value(2.0, label="x1")
    x2 = Value(0.0, label="x2")
    
    # weights
    w1 = Value(-3.0, label="w1")
    w2 = Value(1.0, label="w2") 
    # bias
    b = Value(6.8813735870195432, label="b")
    # x1w1 + x2w2 + b
    x1w1 = x1 * w1; x1w1.label = "x1w1"
    x2w2 = x2 * w2; x2w2.label = "x2w2"
    x1w1x2w2 = x1w1 + x2w2; x1w1x2w2.label = "x1w1 + x2w2"
    n = x1w1x2w2 + b; n.label = "n"
    # activation
    # o = n.tanh(); o.label = "o"
    e = (2*n).exp(); e.label = "e"
    o = (e - 1) / (e + 1); o.label = "o"
    o.backward()
    # target
    # Generate and display the graph
    dot = draw_dot(o)
    dot.render('output/grad_viz', format='png', cleanup=True)  # Save as PNG


def test_torch_land():
    x1 = torch.Tensor([2.0]).double(); x1.requires_grad = True
    x2 = torch.Tensor([0.0]).double(); x2.requires_grad = True
    w1 = torch.Tensor([-3.0]).double(); w1.requires_grad = True
    w2 = torch.Tensor([1.0]).double(); w2.requires_grad = True
    b = torch.Tensor([6.8813735870195432]).double(); b.requires_grad = True
    n = x1*w1 + x2*w2 + b
    o = torch.tanh(n)
    o.backward()

    print("-----------------")
    # print(o.grad.item())
    # print(n.grad.item())
    print(w1.grad.item())
    print(w2.grad.item())
    print(b.grad.item())
    print(x1.grad.item())
    print(x2.grad.item())
    print("-----------------")

def test_mlp():
    # init network
    n = MLP(3, [4, 4, 1])

    xs = [
        [2.0, 3.0, -1.0],
        [3.0, -1.0, 0.5],
        [0.5, 1.0, 1.0],
        [1.0, 1.0, -1.0]
    ]
    ys = [1.0, -1.0, -1.0, 1.0] # desired targets

    # training loop
    for k in range(200):
        # forward pass
        ypred = [n(x) for x in xs]
        loss = sum((yout - ygt)**2 for ygt, yout in zip(ys, ypred))

        # backward pass
        # zero all gradients
        for p in n.parameters():
            p.grad = 0.0
        loss.backward()

        # update weights
        lr = 0.01   # learning rate
        for p in n.parameters():
            p.data -= lr * p.grad

        print(k, loss.data)

    print([y.data for y in ypred])

if __name__ == '__main__':
    # test_micrograd_land()
    # test_torch_land()
    test_mlp()
