import torch
from torch import nn

def kali():
  print ('kali')
  
# Define a neural network YOUR ROLL NUMBER (all small letters) should prefix the classname
class YourRollNumberNN(nn.Module):
  pass
  # ... your code ...
  # ... write init and forward functions appropriately ...
    
# sample invocation torch.hub.load(myrepo,'get_model',train_data_loader=train_data_loader,n_epochs=5, force_reload=True)
def get_model(train_data_loader=None, n_epochs=10):
  model = None
  
  
  print ('Returning model... (rollnumber: xx)')
  
  return model

# sample invocation torch.hub.load(myrepo,'get_model_advanced',train_data_loader=train_data_loader,n_epochs=5, force_reload=True)
def get_model_advanced(train_data_loader=None, n_epochs=10,lr=1e-4,config=None):
  model = None
  return model
  
  
  print ('Returning model... (rollnumber: xx)')
  
  return model

# sample invocation torch.hub.load(myrepo,'test_model',model1=model,test_data_loader=test_data_loader,force_reload=True)
def test_model(model1=None, test_data_loader=None):

  accuracy_val, precision_val, recall_val, f1score_val = 0, 0, 0, 0

  
  print ('Returning metrics... (rollnumber: xx)')
  
  return accuracy_val, precision_val, recall_val, f1score_val
