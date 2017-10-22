
# coding: utf-8

# # Table of Contents
#  <p><div class="lev1 toc-item"><a href="#Binary-Classification" data-toc-modified-id="Binary-Classification-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Binary Classification</a></div><div class="lev2 toc-item"><a href="#Set-up" data-toc-modified-id="Set-up-1.1"><span class="toc-item-num">1.1&nbsp;&nbsp;</span>Set-up</a></div><div class="lev2 toc-item"><a href="#Cost-Function" data-toc-modified-id="Cost-Function-1.2"><span class="toc-item-num">1.2&nbsp;&nbsp;</span>Cost Function</a></div><div class="lev2 toc-item"><a href="#Optimization" data-toc-modified-id="Optimization-1.3"><span class="toc-item-num">1.3&nbsp;&nbsp;</span>Optimization</a></div><div class="lev3 toc-item"><a href="#Gradient-descent" data-toc-modified-id="Gradient-descent-1.3.1"><span class="toc-item-num">1.3.1&nbsp;&nbsp;</span>Gradient descent</a></div><div class="lev2 toc-item"><a href="#Forward-and-Backward-propagation" data-toc-modified-id="Forward-and-Backward-propagation-1.4"><span class="toc-item-num">1.4&nbsp;&nbsp;</span>Forward and Backward propagation</a></div><div class="lev2 toc-item"><a href="#Logistic-Reg-as-a-Neural-Net" data-toc-modified-id="Logistic-Reg-as-a-Neural-Net-1.5"><span class="toc-item-num">1.5&nbsp;&nbsp;</span>Logistic Reg as a Neural Net</a></div><div class="lev2 toc-item"><a href="#Forward-propagation-in-python" data-toc-modified-id="Forward-propagation-in-python-1.6"><span class="toc-item-num">1.6&nbsp;&nbsp;</span>Forward propagation in python</a></div><div class="lev2 toc-item"><a href="#Backward-propagation-in-python" data-toc-modified-id="Backward-propagation-in-python-1.7"><span class="toc-item-num">1.7&nbsp;&nbsp;</span>Backward propagation in python</a></div>

# # Binary Classification
# 
# ## Set-up
# 
# Given $\{ (x_1, y_1), ..., (x_m, y_m) \}$, where $y \in \{0, 1\}$, we are trying to approximate $P(y | x) = \hat{y}$.
# 
# Let's define $$X_{(n_{X}, m)}= \begin{bmatrix}
# x_{1,1}&...&x_{m,1}\\
# x_{1,2}&...&x_{m,2}\\
# ...&...&...\\
# x_{1, n_X}&...&x_{m, n_X}\\
# \end{bmatrix}$$
# 
# And $$y_{(1, m)} = \begin{bmatrix} y_1 & y_2 & ... & y_m \\ \end{bmatrix}$$
# 
# That is, for individual vector of features, we stack them as columns and we call the result the overall $X$ and $Y$.
# 
# Given this, we parametrize our hypothesis as follows:
# 
# $$\hat{y} = \sigma(w^T x + b)$$
# 
# Where $$\sigma(z) = \frac{1}{1+e^{-z}}$$
# 
# ## Cost Function
# 
# From the maximum likelihood, we get the following loss function (single training example):
# 
# $$ L(y_i, \hat{y_i}) = - (y log(\hat{y}) + (1-y) log(1 - \hat{y})) $$
# 
# And the cost function (average loss over training):
# 
# $$ J(w, b) = \frac{1}{m} \sum_{i = 1}^m L(y_i, \hat{y_i}) $$
# 
# ## Optimization
# 
# Ok, so we have posted a hypothesis and have a cost function to measure the availability of each possible pair of parameteres. How to find the optimal pair? Gradient descent.
# 
# ### Gradient descent
# 
# Gradient descent is simple hill climbing. Start somewhere in the mountain, follow the step where the function decreases the most, i.e., gradient's way. Do so till you get to a minimum. 
# 
# In pseudo-code:
# 
# Repeat { 
#     $$w := w - \alpha \frac{\partial C}{\partial w}$$
#     $$b := b - \alpha \frac{\partial C}{\partial b}$$
# }
# 
# Andrew Ng didn't define a stopping criteria. 
# 
# ## Forward and Backward propagation
# 
# Neural Nets are, quite simply, the automation of variable transformation through a multi-step transformation of the input variables. The magic, of course, comes from the fact that the transformations are done to fit the training set. 
# 
# Thus, to compute a prediction with a Neural Network, we must compute multiple intermediate variables from the back to the to top. However, the effect of the initial variable now is muted by the multiple steps in the intermediate variables. Thus, derivative computation is not straightforward. The effect of the input is reflected thorought the network in the computation of all the intermediate variables. 
# 
# Solution? Backward propagation. Start with the immediate last hidden network and the corresponding hidden variable. Compute the derivative with respect to it. Repeat. Repeat till you have computed the derivative with respect to an input variable. The Chain rule simply tells you how to organize these derivatives to compute the derivative of the output variable with respect to the input. 
# 
# ## Logistic Reg as a Neural Net
# 
# Remember: a neural net is simply a transformation-first framework. Just say that the transformation is as follows: 
# 
# $$ z = w^T*X + b $$
# $$ a = \sigma (z)$$
# 
# $$ \frac{da}{dz} = \frac{d \sigma}{dz} $$
# $$ \frac{da}{dw} = X \frac{da}{dz}^T $$
# $$ \frac{da}{db} = \frac{da}{dz} $$
# 

# ## Forward propagation in python

# In[93]:

import numpy as np 
np.random.seed(10)
import math
# Given X, w, and b.
X = np.random.random(size = (3, 10000))
w = np.random.random(size = (3, 1))
b = 0.03
def sigmoid(z):
    return 1/(1 + np.exp(-z))
# Find out true y
Y = sigmoid(np.dot(w.T, X) + b)
Y = np.where(Y <= 0.5, 0, 1)


# ## Backward propagation in python

# In[97]:

alpha = 0.03
iterations = 2000
w_hat = np.zeros_like(w)
b_hat = 0
for x in range(iterations):
    # Forward propagation 
    Z = np.dot(w_hat.T, X) + b_hat
    A = sigmoid(Z)
    cost = -1/m * np.sum(Y * np.log(A) + (1 - Y)* np.log(1-A)) 
    #Backward propagation
    dZ = A - Y
    dw = 1/10000 * (np.dot(X, dZ.T))
    db = 1/10000 * (np.sum(dZ))
    w_hat = w_hat - alpha * dw
    b_hat = b_hat - alpha * db
Y_hat = sigmoid(np.dot(w_hat.T, X) + b_hat)
print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_hat - Y)) * 100))


# In[ ]:



