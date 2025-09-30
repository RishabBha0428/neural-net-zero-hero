# neural-net-zero-hero
All code artifacts as part of the Karpathy video series
# Lesson 1: micrograd and backprop
## Summary
Neural networks are mathematical expressions with input as data and weights. Forward pass of predictions, then evaluate loss to evaluate predictions with ground truth, lower the loss the better, compute the backward pass of the loss to get the gradient, then we know how to tune the parameters to update locally, do this many times for gradient descent to minimize loss

Do batching in practice for millions of examples

Learning rate decay: fine-detail stabilization
## Why Backpropagation
Neural networks can be viewed as complex mathematical functions. Backpropagation is the algorithm that efficiently computes the derivative of the loss function with respect to each parameter in the network. This allows the model to adjust its parameters in the direction of the steepest descent of the loss, enabling learning through gradient descent.

## Most Common Mistakes Training Neural Nets
1. You didn't try to overfit a single batch first
2. you forgot to toggle train/eval mode for the net
3. you forgot to zero_grad() before backward()
	1. grads accumulate for each backward pass, need to reset, makes convergence too fast
4. passed softmax outputs to loss that expects raw logits
