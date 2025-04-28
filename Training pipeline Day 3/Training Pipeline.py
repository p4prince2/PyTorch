#!/usr/bin/env python
# coding: utf-8

# In[22]:


import pandas as  pd
import numpy as np
from sklearn.model_selection  import train_test_split
from sklearn.preprocessing import StandardScaler,LabelEncoder
import torch


# In[24]:


df=pd.read_csv('https://raw.githubusercontent.com/gscdit/Breast-Cancer-Detection/refs/heads/master/data.csv')
df.head()


# In[25]:


df.shape


# In[26]:


df.drop(columns=['id', 'Unnamed: 32'],inplace=True)


# In[27]:


df.head()


# ### Train test split

# In[29]:


x_train,x_test,y_train,y_test=train_test_split(df.iloc[:,1:],df.iloc[:,0],test_size=0.2,random_state=42)


# ### scaling

# In[36]:


scaler=StandardScaler()
X_train=scaler.fit_transform(x_train)
X_test=scaler.transform(x_test)


# In[38]:


X_train


# In[42]:


y_train


# ### Label Encoding
# 

# In[48]:


encoder = LabelEncoder()
y_train = encoder.fit_transform(y_train)
y_test = encoder.transform(y_test)


# In[50]:


y_train


# ## PyTorch Started 

# ## Numpy arrays to PyTorch tensors

# In[54]:


X_train_tensor=torch.from_numpy(X_train)
X_test_tensor=torch.from_numpy(X_test)
y_train_tensor=torch.from_numpy(y_train)
y_test_tensor=torch.from_numpy(y_test)


# In[56]:


X_train_tensor.shape


# In[58]:


y_train_tensor.shape


# ### Defining the model

# In[69]:


class MysampleNN():

    def __init__(self,X):
        self.weights = torch.rand(X.shape[1], 1, dtype=torch.float64, requires_grad=True)
        self.bias = torch.zeros(1, dtype=torch.float64, requires_grad=True)


    def forward(self,X):
        z=torch.matmul(X,self.weights)+self.bias
        y_pred=torch.sigmoid(z)
        return y_pred
    def loss_function(self, y_pred,y):
        ## Clamp predictions to avoid log(0)
        epsilon=1e-7
        y_pred = torch.clamp(y_pred, epsilon, 1 - epsilon)

        # Calculate loss
        loss = -(y_train_tensor * torch.log(y_pred) + (1 - y_train_tensor) * torch.log(1 - y_pred)).mean()
        return loss


# ### Important Parameters

# In[72]:


learning_rate = 0.1
epochs = 25


# ### Training Pipeline

# In[75]:


## Creating model

model=MysampleNN(X_train_tensor)


#define loop

for epoch in range(epochs):

    # forward pass
    y_pred=model.forward(X_train_tensor)

    # loss Calculate
    loss=model.loss_function(y_pred,y_train_tensor)

    # backward pass
    loss.backward()

    # parameters update
    with torch.no_grad():
        model.weights -= learning_rate * model.weights.grad
        model.bias -= learning_rate * model.bias.grad

    # Zero gradients
    model.weights.grad.zero_()
    model.bias.grad.zero_()
    
    # print loss in each epoch
    print(f'Epoch: {epoch + 1}, Loss: {loss.item()}')



# In[ ]:





# ### Evaluation

# In[87]:


# model evaluation
with torch.no_grad():
  y_pred = model.forward(X_test_tensor)
  y_pred = (y_pred > 0.90).float()
  accuracy = (y_pred == y_test_tensor).float().mean()
  print(f'Accuracy: {accuracy.item()}')


# ## ***Its not about  the accuracy , its about the learning***

# In[ ]:




