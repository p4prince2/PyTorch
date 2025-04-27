#!/usr/bin/env python
# coding: utf-8

# ***PyTorch, the term autograde typically refers to automatic differentiation (or autograd), which is a key feature that enables PyTorch to automatically compute gradients for backpropagation during training of machine learning models. The Autograd module in PyTorch is designed to automatically compute the gradients of tensors during the forward pass, and this is crucial for training neural networks using optimization techniques like gradient descent.***
# 
# **Key Concepts of Autograd in PyTorch:**
# 
# **Tensors and Gradients:**
# 
# - PyTorch provides a special class of tensors called torch.Tensor which can track computation history.
# 
# - Tensors with requires_grad=True will track all operations applied to them and store the gradient for those operations.
# 
# **Backward Pass:**
# 
# - The .backward() method computes all gradients for the tensors involved in the computation graph.
# 
# - After calling .backward(), gradients are stored in the .grad attribute of each tensor that requires gradients.
# 
# **Computation Graph:**
# 
# - PyTorch builds a dynamic computation graph as you define operations. Each operation adds a node to the graph.
# 
# - When you call .backward(), the system traverses the graph and computes gradients for each node.
# 
# **Gradient Descent:**
# 
# - Once gradients are computed, they can be used to update model parameters during training, usually through optimizers like torch.optim.

# ## Example Code for Autograd in PyTorch:

# In[4]:


# if you want to differente in normal using python
def dy_dx(x):
    return 2*x


# In[6]:


dy_dx(3)


# **By using tensor-autograde**

# In[10]:


import torch


# In[12]:


x=torch.tensor(3.0,requires_grad=True)


# In[14]:


y=x**2


# In[16]:


x


# In[18]:


y


# In[20]:


y.backward()


# In[24]:


x.grad  ## dy_dx(3)


# In[28]:


## In one line
X=torch.tensor(3.0,requires_grad=True)
y=X**2
y.backward()
x.grad


# In[ ]:





# **Another without using Autograde**

# In[37]:


# y=x^2
# z=sin(y)


import math
def dz_dx(x):
    return 2* x* math.cos(x**2)


# In[34]:


dz_dx(4)


# **with Autograde**

# In[40]:


x=torch.tensor(4.0, requires_grad=True)


# In[44]:


y=x**2


# In[46]:


z=torch.sin(y)


# In[48]:


x


# In[52]:


y


# In[54]:


z


# In[56]:


z.backward()


# In[58]:


x.grad


# In[60]:


## in one line 
x=torch.tensor(4.0, requires_grad=True)
y=x**2
z=torch.sin(y)
z.backward()
x.grad  ## samee with dz_dx  but simple to use


# In[64]:


y.grad


# **Single perceptron without using Autograde**

# In[72]:


x=torch.tensor(6.7)  # input feature 
y=torch.tensor(0.0)  ## True label (binary)

w=torch.tensor(1.0) # weight
b=torch.tensor(0.0)  # bias



# In[74]:


# binary cross-Entropy loss for scaler

def binary_cross_entropy_loss(prediction,target):
    epsilon=1e-8   # to prevent log(0)
    prediction=torch.clamp(prediction,epsilon,1-epsilon)#torch.clamp(input, min=0, max=10)
    return -(target*torch.log(prediction)+ (1 -target)*torch.log(1-prediction))


# In[76]:


# Forwar pass

z=w*x+b  # Weighted sum (liner part)
y_pred=torch.sigmoid(z)  # predtion probability

# compute binary cross-entropy_loss
loss=binary_cross_entropy_loss(y_pred,y)


# In[78]:


loss


# In[82]:


## Derivaties:

#1. dl/d(y_pred): Loss with respect to prediction(y_pred)
dloss_dy_pred=(y_pred  -y )/(y_pred*(1-y_pred))

#2. dy_pred/dz: Prediction(y_pred) with repect toz (sigmoid  derivatie)
dy_pred_dz=y_pred*(1-y_pred)

#3. dz/dw and dz/db : z with respect to w and b 
dz_dw = x #dz/dw=x
dz_db=1 # dz/db=1(bias contribution directly to z)


dL_dw = dloss_dy_pred * dy_pred_dz * dz_dw
dL_db = dloss_dy_pred * dy_pred_dz * dz_db


# In[84]:


print(f"Manual Gradient of loss w.r.t weight (dw): {dL_dw}")
print(f"Manual Gradient of loss w.r.t bias (db): {dL_db}")


# **Single perceptron with using Autograde**

# In[88]:


x= torch.tensor(6.7)
y=torch.tensor(0.0)



# In[92]:


w=torch.tensor(1.0, requires_grad=True)
b=torch.tensor(0.0,requires_grad=True)


# In[94]:


w


# In[96]:


b


# In[98]:


z=w*x+b
z


# In[102]:


y_pred=torch.sigmoid(z)
y_pred


# In[108]:


loss=binary_cross_entropy_loss(y_pred,y)
loss


# In[110]:


loss.backward()


# In[114]:


print(w.grad)
print(b.grad)   ## same as above   here we dont have to write the code for Derivaties:


# **for vector value** 

# In[121]:


x=torch.tensor([1.0,2.0,3.0],requires_grad=True)
y=(x**2).mean()
y


# In[123]:


y.backward()


# In[125]:


x.grad


# # clearing grad

# **Why clearing grad  important**
# 
# - Clearing gradients is important to avoid the accumulation of gradients across different iterations, ensure fresh gradient computation, and prevent unnecessary memory usage.
# 
# - It guarantees that each iteration correctly updates the model parameters based on the current batch's data.

# In[190]:


x = torch.tensor(2.0, requires_grad=True)
x


# In[192]:


y=x**2
y


# In[194]:


y.backward()


# In[196]:


x.grad


# In[198]:


x.grad.zero_()


# # Disabling gradient tracking
# Disabling gradient tracking is useful when you don’t need to compute gradients for certain parts of the model, or during inference (prediction) where you don't want the computational graph to be built, which can save memory and computational resources. This is done using the torch.no_grad() context manager or by setting the requires_grad attribute of tensors to False.
# 
# **option 1 - requires_grad_(False)**
# 
# **option 2 - detach()**
# 
# **option 3 - torch.no_grad()**

# In[202]:


x = torch.tensor(2.0, requires_grad=True)
x


# In[204]:


y = x ** 2
y


# In[206]:


y.backward()


# In[208]:


x.grad


# In[210]:


# option 1 - requires_grad_(False)
# option 2 - detach()
# option 3 - torch.no_grad()


# In[216]:


# option 1 - requires_grad_(False)
x.requires_grad_(False)


# In[214]:


x


# In[218]:


# option 2 - detach()
x = torch.tensor(2.0, requires_grad=True)
x


# In[220]:


z = x.detach()
z


# In[222]:


# option 3 - torch.no_grad()

# Create a tensor with requires_grad=True
x = torch.randn(3, 3, requires_grad=True)

# Disable gradient tracking
with torch.no_grad():
    y = x * 2  # Operations here won't track gradients

print(y.requires_grad)  # Output will be False


# In[ ]:




