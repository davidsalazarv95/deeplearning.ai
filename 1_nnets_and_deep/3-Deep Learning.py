
# coding: utf-8

# # Table of Contents
#  <p><div class="lev1 toc-item"><a href="#Why-bother-going-deep?" data-toc-modified-id="Why-bother-going-deep?-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Why bother going deep?</a></div><div class="lev2 toc-item"><a href="#Complex-transformations" data-toc-modified-id="Complex-transformations-1.1"><span class="toc-item-num">1.1&nbsp;&nbsp;</span>Complex transformations</a></div><div class="lev2 toc-item"><a href="#Circuit-Theory" data-toc-modified-id="Circuit-Theory-1.2"><span class="toc-item-num">1.2&nbsp;&nbsp;</span>Circuit Theory</a></div><div class="lev1 toc-item"><a href="#Hyperparameters" data-toc-modified-id="Hyperparameters-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>Hyperparameters</a></div>

# # Why bother going deep?
# 
# ## Complex transformations
# 
# Remember that the gist of a neural network is the possibility of multiple and complex transformations of the inputs, all in a data-driven manner. With one layer, the outputs are not transformed at all, and thus the ability of the algorithm to recognize useful information and use it to improve prediction is low. However, as the number of layers increases, so it does the ability to do complex transformations and finding valuable information to improve the accuracy of our predictions. 
# 
# ## Circuit Theory
# 
# Deep neural networks require a much lower number of nodes per layer than shallow neural networks to approximate some functions. 
# 
# # Hyperparameters
# 
# Parameters that determine the learning but are not learned in the algorithm. Instead, they are set by you. For example:
# 
#     - # of hidden layers.
#     - # of units in hidden layers.
#     - Choice of activation function by layer.
#     - Learning rate in gradient descent.
#     - # of iterations in gradient descent.
#     
# Try out different values, evaluate in validation set, and then evaluate generalizable error. 

# In[ ]:



