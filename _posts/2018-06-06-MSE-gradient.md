---
layout: post
title:  "Step-by-step gradient computation of the Mean Squared Error (MSE)"
comments: true
tags:
  - calculus
---

## Equations

$$
y_{prediction}(x_0, x_1, ..., x_{m-1}) = w_0x_0 + w_1x_1 + ... + w_{m-1}x_{m-1}
$$

$$
MSE = \frac{1}{2n} \sum_{i=0}^{n-1}{(y_{actual} - (w_0x_0 + w_1x_1 + ... + w_{m-1}x_{m-1}))}^2
$$

Gradient of the MSE:

$$
\nabla MSE(w_0, w_1, ..., w_{m-1}) = \frac{\partial_{MSE}}{\partial w_0}, \frac{\partial_{MSE}}{\partial w_1}, ..., \frac{\partial_{MSE}}{\partial w_{m-1}}
$$

Let's take the first partial derivative as an example.

$$
u = (y_{actual} - (w_0x_0 + w_1x_1 + ... + w_{m-1}x_{m-1}))
$$

$$
MSE = \frac{1}{2n} \sum_{i=0}^{n-1}{u}^2
$$

Using the chain rule:

$$
\frac{\partial_{MSE}}{\partial w_0} = \frac{\partial_{MSE}}{\partial u} * \frac{\partial u}{\partial w_0}
$$

The constants are cancelled out.

$$
\frac{\partial_{MSE}}{\partial u} = \frac{1}{n} \sum_{i=0}^{n-1}{u}
$$

Only $$x_0$$ will remain from $$u = (y_{actual} - (w_0x_0 + w_1x_1 + ... + w_{m-1}x_{m-1}))$$ since all other variables will be treated as a constant except for $$w_0$$.

$$
\frac{\partial u}{\partial w_0} = x_0
$$

Back to the other equation:

$$
\frac{\partial_{MSE}}{\partial w_0} = \frac{\partial_{MSE}}{\partial u} * \frac{\partial u}{\partial w_0}
$$

$$
\frac{\partial_{MSE}}{\partial w_0} = \frac{1}{n} \sum_{i=0}^{n-1}{(u)} * x_0
$$

Substitute $$u$$.

$$
\frac{\partial_{MSE}}{\partial w_0} = \frac{1}{n} \sum_{i=0}^{n-1}{(y_{actual} - (w_0x_0 + w_1x_1 + ... + w_{m-1}x_{m-1}))} * x_0
$$

Substitute the $$y_{prediction}$$ function.

$$
\frac{\partial_{MSE}}{\partial w_0} = \frac{1}{n} \sum_{i=0}^{n-1}{(y_{actual} - y_{prediction})} * x_0
$$

Then do this for all the weights:

$$
\frac{\partial_{MSE}}{\partial w_1} = \frac{1}{n} \sum_{i=0}^{n-1}{(y_{actual} - y_{prediction})} * x_1
$$

$$
...
$$

$$
\frac{\partial_{MSE}}{\partial w_{m-1}} = \frac{1}{n} \sum_{i=0}^{n-1}{(y_{actual} - y_{prediction})} * x_{m-1}
$$
