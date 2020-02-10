import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        
        return output   
    
    def prune(self, method, percentage):
    
        if method == "layerwise":
            mask = torch.Tensor([])

            for name, layer in self.named_parameters():
                if "weight" in name:
                    
                    weights = layer.view(1, -1)
                    normed_weights = torch.abs(weights)
                    
                    k = int(normed_weights.size(-1) * percentage)
                    _, indices = torch.topk(normed_weights, k, -1)

                    layer_mask = torch.zeros(normed_weights.shape)
                    layer_mask = torch.scatter(layer_mask, -1, indices.long(), 1)
                    
                    mask = torch.cat((mask, layer_mask), -1)
        
        elif method == "global":
            weights = torch.Tensor([])

            for name, layer in self.named_parameters():
                if "weight" in name:
                    layer_weights = layer.view(1, -1)    
                    weights = torch.cat((weights, layer_weights), -1)
            
            normed_weights = torch.abs(weights)
            k = int(normed_weights.size(-1) * percentage)
            _, indices = torch.topk(normed_weights, k, -1)

            mask = torch.zeros(normed_weights.shape)
            mask.scatter_(layer_mask, indices.long(), 1)
        
        return mask
    
    def mask(self, mask):
        start_idx = 0
        for name, layer in self.named_parameters():
            if "weight" in name:
                shape = layer.shape
                end_idx = start_idx + layer.view(1, -1).size(-1)

                layer_weights = layer.view(1, -1)
                layer_mask = mask[:, start_idx:end_idx]
                masked_layer = layer_weights * layer_mask
                masked_layer = masked_layer.view(shape)

                self._modules[name[:-7]].weight = torch.nn.Parameter(masked_layer)  # index to remove ".weight" from key

    def save(self, save_dir):
        torch.save(self.state_dict(), save_dir)