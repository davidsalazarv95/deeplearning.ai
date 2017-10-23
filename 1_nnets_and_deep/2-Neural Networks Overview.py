
# coding: utf-8

# # Table of Contents
#  <p><div class="lev1 toc-item"><a href="#Neural-Networks" data-toc-modified-id="Neural-Networks-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Neural Networks</a></div><div class="lev2 toc-item"><a href="#Transformation-layer" data-toc-modified-id="Transformation-layer-1.1"><span class="toc-item-num">1.1&nbsp;&nbsp;</span>Transformation layer</a></div><div class="lev2 toc-item"><a href="#1-Hidden-Layer-NNet" data-toc-modified-id="1-Hidden-Layer-NNet-1.2"><span class="toc-item-num">1.2&nbsp;&nbsp;</span>1 Hidden Layer NNet</a></div><div class="lev2 toc-item"><a href="#Activation-Functions" data-toc-modified-id="Activation-Functions-1.3"><span class="toc-item-num">1.3&nbsp;&nbsp;</span>Activation Functions</a></div><div class="lev3 toc-item"><a href="#Rectified-Linear-Unit-(ReLu)" data-toc-modified-id="Rectified-Linear-Unit-(ReLu)-1.3.1"><span class="toc-item-num">1.3.1&nbsp;&nbsp;</span>Rectified Linear Unit (ReLu)</a></div><div class="lev3 toc-item"><a href="#Linear-Activation" data-toc-modified-id="Linear-Activation-1.3.2"><span class="toc-item-num">1.3.2&nbsp;&nbsp;</span>Linear Activation</a></div><div class="lev2 toc-item"><a href="#Bakcward-Propagation" data-toc-modified-id="Bakcward-Propagation-1.4"><span class="toc-item-num">1.4&nbsp;&nbsp;</span>Bakcward Propagation</a></div><div class="lev2 toc-item"><a href="#Random-Initialization" data-toc-modified-id="Random-Initialization-1.5"><span class="toc-item-num">1.5&nbsp;&nbsp;</span>Random Initialization</a></div><div class="lev2 toc-item"><a href="#Conclusion" data-toc-modified-id="Conclusion-1.6"><span class="toc-item-num">1.6&nbsp;&nbsp;</span>Conclusion</a></div>

# # Neural Networks
# 
# The key takeaway from neural networks: transformation layers and a prediction layer. Transformations are decided based on the data. Thanks to multiple transformations, neural networks are universal approximators. 
# 
# ## Transformation layer
# 
# Remember that in machine learning, we assume the "true" function resembles our own. However, within neural networks, we must make the assumption that the transformation layer(s) are not observable, and thus are called hidden layer(s). 
# 
# The hidden layers all work the same. For each neuron in the current hidden layer, compute a linear combination of the results (activations) of the last year and activate this result with a given function. Then, the current neuron has been activated; when done with all neurons in the layer, do the same with the next layer. And so on. 
# 
# Thus, each neuron has associated with it the corresponding parameters it uses in the linear combination and the result it yields after being activated. We will stack the parameters in matrix for each layer: $W^{[1]}$ being the matrix with all the $w$ coefficients for the first layer. $W^{[1]}_{1}$ being the coefficients of the first node (neuron) in the layer.
# 
# ## 1 Hidden Layer NNet
# 
# ![1 Hidden Layer NNet](Images/2layers.png)
# 
# ![First computation](Images/first_layer.png)
# 
# ![Matrix Stacking](Images/matrix_stacking.png)
# 
# Thus, for this neural network, for a given example, the only thing to implement are these four equations:
# 
# ![Vectorised_implementation](Images/vectorised.png)
# 
# Replace the $x$ with $X$ and voila!, you have done it for all the observations. 
# 
# ## Activation Functions
# 
# The activation functions are the functions used at each node after the linear combination. In the logistic regression, we used the sigmoid function. However, we sometimes may use: 
# 
# $$tanh = \frac{e^z - e^{-z}}{e^z + e^{-z}}$$
# 
# This has the benefit of being centered around zero, which makes learning easier for the posterior layers. However, the output layer must have a sigmoid when using binary classification. 
# 
# ### Rectified Linear Unit (ReLu)
# 
# One of the downsides of both the sigmoid and the $tanh$ is that for extreme values of x, the derivative will be close to zero. Making the job of gradient descent much slower. ReLu:
# 
# $$ ReLu = max\{0, z \} $$
# 
# Most common at hidden layers.
# 
# ### Linear Activation
# 
# Linear activation functions for a regression problem. 
# 
# ## Bakcward Propagation
# 
# ![Backward](Images/backward_propagation.png)
# 
# ## Random Initialization
# 
# It's important to initialize weights randomly. If you set all to zero, then all the transformation for all the nodes will start and keep being identical thorough the learning process. Thus, you won't leverage the power of the Neural Network. 
# 
# Also, make sure your values are small, such that the values for the activation functiona are not extreme and don't slow down the learning. 
# 
# ## Conclusion
# 
# **Reminder**: The general methodology to build a Neural Network is to:
#     1. Define the neural network structure ( # of input units,  # of hidden units, etc). 
#     2. Initialize the model's parameters
#     3. Loop:
#         - Implement forward propagation
#         - Compute loss
#         - Implement backward propagation to get the gradients
#         - Update parameters (gradient descent)
# 
# 

# In[ ]:



