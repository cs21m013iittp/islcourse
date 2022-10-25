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

  # write your code here as per instructions
  # ... your code ...
  # ... your code ...
  # ... and so on ...
  # Use softmax and cross entropy loss functions
  # set model variable to proper object, make use of train_dat
  device = 'cuda' if torch.cuda.is_available() else 'cpu'
  for X_b, y_b in train_data_loader:
    X = X_b[0]
    (ch, H, W) = X.shape
    break
  
  labels = set()
  for X_b, y_b in train_data_loader:
    labels = labels.union(set(y_b.cpu().detach().numpy()))
    
  num_classes = len(labels)
  
  
  class cs21m013(nn.Module):
    def __init__(self, ch, H, W, num_classes):
      super(cs21m013, self).__init__()
      self.conv_layers = nn.Sequential(
          nn.Conv2d(ch, 8, 3, padding='same'),
          nn.Conv2d(8, 16, 3, padding='same'),
          )
      in_features = 16 * H * W
      self.fc = nn.Sequential(
          nn.Linear(in_features, 2048),
          nn.Linear(2048, 1024),
          nn.Linear(1024, 512),
          nn.Linear(512, 256),
          nn.Linear(256, num_classes)
          )
      
      
      def forward(self, x):
        feat_maps = self.conv_layers(x)
        feats = nn.Flatten(start_dim=1)(feat_maps)
        logits = self.fc(feats)
        return logits
        
        
        
 model = cs21m013(ch, H, W, num_classes)
 model.to(device)
 optimizer = optim.Adam(model.parameters()
 total_loss = 0.0
 correct = 0
 for X_b, y_b in tqdm(train_data_loader):
   X_b, y_b = X_b.to(device), y_b.to(device)
   logits = model(X_b)
   loss = nn.CrossEntropyLoss()(logits, y_b)
   optimizer.zero_grad()
   loss.backward()
   optimizer.step()
   total_loss += loss.item()
   probs: torch.Tensor = nn.Softmax(dim=1)(logits)
   preds = probs.argmax(dim=1)
   correct += (preds == y_b).sum().item()
    

 print(f"Accuracy of the model: {correct/len(train_data_loader.dataset)}")
 return model

  
  print ('Returning model... (rollnumber: xx)')
  
  return model

# sample invocation torch.hub.load(myrepo,'get_model_advanced',train_data_loader=train_data_loader,n_epochs=5, force_reload=True)
def get_model_advanced(train_data_loader=None, n_epochs=10,lr=1e-4,config=None):
  model = None

  # write your code here as per instructions
  # ... your code ...
  # ... your code ...
  # ... and so on ...
  # Use softmax and cross entropy loss functions
  # set model variable to proper object, make use of train_data
  
  # In addition,
  # Refer to config dict, where learning rate is given, 
  # List of (in_channels, out_channels, kernel_size, stride=1, padding='same')  are specified
  # Example, config = [(1,10,(3,3),1,'same'), (10,3,(5,5),1,'same'), (3,1,(7,7),1,'same')], it can have any number of elements
  # You need to create 2d convoution layers as per specification above in each element
  # You need to add a proper fully connected layer as the last layer
  
  # HINT: You can print sizes of tensors to get an idea of the size of the fc layer required
  # HINT: Flatten function can also be used if required
  return model
  
  
  print ('Returning model... (rollnumber: xx)')
  
  return model

# sample invocation torch.hub.load(myrepo,'test_model',model1=model,test_data_loader=test_data_loader,force_reload=True)
def test_model(model1=None, test_data_loader=None):

  accuracy_val, precision_val, recall_val, f1score_val = 0, 0, 0, 0
  # write your code here as per instructions
  # ... your code ...
  # ... your code ...
  # ... and so on ...
  # calculate accuracy, precision, recall and f1score
  
  def test_model(model1=None, test_data_loader=None):
  accuracy_val, precision_val, recall_val, f1score_val = 0, 0, 0, 0
  device = 'cuda' if torch.cuda.is_available() else 'cpu'
  
  total_loss = 0.0
  correct = 0
  for X_b, y_b in test_data_loader:
    X_b, y_b = X_b.to(device), y_b.to(device)
    logits = model1(X_b)
    loss = nn.CrossEntropyLoss()(logits, y_b)
    total_loss += loss.item()

    probs = nn.Softmax(dim=1)(logits)
    preds = probs.argmax(dim=1)

    correct = (preds == y_b).sum().item()

    preds = preds.cpu().detach().numpy()
    y_b = y_b.cpu().detach().numpy()


    accuracy_val = correct / len(test_data_loader.dataset)
    precision_val = precision_score(y_b, preds, average='macro')
    recall_val = recall_score(y_b, preds, average='macro')
    f1score_val = f1_score(y_b, preds, average='macro')

    print ('Returning metrics... (cs21m013: xx)')
    return accuracy_val, precision_val, recall_val, f1score_val
