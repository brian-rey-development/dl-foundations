# 10. Next steps

Everything in this project scales up. Nothing here is a toy version of a different idea; it is the
same idea at small size. This guide maps what you built to what comes next.

## What you have

| Built here | Same thing in PyTorch |
|------------|----------------------|
| `Dense` + `dense_forward` | `nn.Linear` |
| `Activation` | `nn.ReLU`, `nn.Sigmoid`, `torch.tanh` |
| `forward` over a tuple of layers | `nn.Sequential` |
| `binary_cross_entropy` | `nn.BCELoss` (or `BCEWithLogitsLoss`, fused with sigmoid) |
| `backward` + `DenseCache` | `loss.backward()` and the autograd graph |
| `sgd_step` | `torch.optim.SGD` |
| `iterate_batches` | `DataLoader(shuffle=True)` |
| `gradient_check` | `torch.autograd.gradcheck` |
| `save_artifact` / `load_artifact` | `torch.save(model.state_dict())` / `load_state_dict` |

When you open PyTorch for the first time, every one of those lines will map to something you wrote.

## What is missing, in the order it usually matters

**Adam.** Replace `sgd_step` with an optimizer that keeps a running mean and variance of past gradients
per parameter. Converges faster on almost everything. About 15 lines in the functional style used here:
the step function takes and returns an optimizer state.

**Early stopping.** Track the validation loss in `train` and return the network from the best epoch, not
the last. Directly addresses the overfitting figure in guide 07.

**Regularization.** L2 (add $\lambda \sum w^2$ to the loss, which becomes `- lr * lambda * w` in the
update) and dropout (randomly zero activations during training). Both fight overfitting.

**Softmax and multi-class.** Sigmoid gives one probability. Softmax gives a distribution over K classes.
The loss becomes categorical cross-entropy. The gradient at the output is still $p - y$.

**Autograd.** Write a tiny `Tensor` class that records operations and computes gradients with the chain
rule, so you never write `dense_backward` by hand again. Andrej Karpathy's micrograd does this in about
100 lines and is the best next project after this one.

**Convolutions, attention.** Different layer types for images and sequences. Same forward, backward,
loss, optimizer around them.

## Resources

- *Neural Networks and Deep Learning*, Michael Nielsen. Free online. Chapters 1 and 2 are this project with
  more prose and MNIST instead of moons.
- *Deep Learning*, Goodfellow, Bengio and Courville. Chapter 6 (feedforward networks) and 8 (optimization)
  are the reference for everything here.
- Karpathy, *Neural Networks: Zero to Hero*, video series. The first video builds micrograd.
- CS231n course notes on backpropagation and on gradient checking. Short, dense, the source for the
  relative-error thresholds in guide 08.
- *Mathematics for Machine Learning*, Deisenroth, Faisal and Ong. Free online. Chapter 5 if you want
  the vector calculus behind the shapes in guide 05 to feel obvious rather than memorized.
