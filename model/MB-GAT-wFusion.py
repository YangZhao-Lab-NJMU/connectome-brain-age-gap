"""
MB-GAT-wFusion: Multi-modal Brain Graph Attention Network with Fusion

A deep learning model for brain age prediction using multi-modal neuroimaging data.

Author: Bowen Xie
License: MIT
"""

import torch
from torch import nn
import torch.nn.functional as F
import math

class ETE(nn.Module): # Convolutional layer with residual and SE mechanism

    def __init__(self, in_dim, out_dim, node, lamda, SE):
        super(ETE, self).__init__()
        self.out_dim = out_dim
        self.node = node
        self.lamda = lamda
        self.SE = SE

        # Convolutional kernel of size (1, node) for convolution along rows
        self.conv_row = nn.Conv2d(in_dim, out_dim, (1, node))
        nn.init.normal_(self.conv_row.weight, std=math.sqrt(2*(1-lamda)/(node*in_dim+node*out_dim)))

        # Convolutional kernel of size (node, 1) for convolution along columns
        self.conv_col = nn.Conv2d(in_dim, out_dim, (node, 1))
        nn.init.normal_(self.conv_col.weight, std=math.sqrt(2*(1-lamda)/(node*in_dim+node*out_dim)))

        self.convres = nn.Conv2d(in_dim, out_dim, 1)
        nn.init.normal_(self.convres.weight, std=math.sqrt(4*lamda/(in_dim+out_dim)))

        self.sed = nn.Linear(out_dim, int(out_dim/SE), False)
        self.seu = nn.Linear(int(out_dim/SE), out_dim, False)

    def forward(self, x):

        x_res = self.convres(x)

        # Convolve along the rows of the original correlation matrix
        x_row_conv = self.conv_row(x)
        
        # Convolve along the columns of the original correlation matrix
        x_col_conv = self.conv_col(x)

        # Element-wise multiplication of the two convolution results
        x_combined = x_row_conv * x_col_conv

        # SE mechanism
        # 1. Squeeze: Global average pooling on the channel dimension to get global channel information
        se = torch.mean(x_combined,(2,3))
        # 2. Excitation: Adjust channel weights through fully connected layers
        se = self.sed(se) # Dimensionality reduction through a fully connected layer
        se = F.relu(se)
        se = self.seu(se)
        se = torch.sigmoid(se)
        # 3. Scaling: Apply the weights obtained from the SE mechanism to the feature map
        se = se.unsqueeze(2).unsqueeze(3)
        x_se = x_combined * se  # Element-wise multiplication
        x = x_se+x_res

        return x

class ETN(nn.Module):

    def __init__(self, in_dim, out_dim, node):
        super(ETN, self).__init__()
        self.conv = nn.Conv2d(in_dim, out_dim, (1, node))
        nn.init.normal_(self.conv.weight, std=math.sqrt(4/(node*in_dim+out_dim)))

    def forward(self, x):
        x = self.conv(x)     
        return x

class NTG(nn.Module):

    def __init__(self, in_dim, out_dim, node):
        super(NTG, self).__init__()
        self.conv = nn.Conv2d(in_dim, out_dim, (node, 1))
        nn.init.normal_(self.conv.weight, std=math.sqrt(4/(node*in_dim+out_dim)))

    def forward(self, x):
        x = self.conv(x)      
        return x

class MB_GAT_wFusion(nn.Module):
    def __init__(self, ETE_dim, ETN_dim, NTG_dim, node, lamda, SE, dropout):
        super(MB_GAT_wFusion, self).__init__()
        
        self.encoder = nn.GRU(input_size=node, hidden_size=8, num_layers=1, batch_first=True, bidirectional=True)
        self.encoder_bn1 = nn.BatchNorm2d(16)
        self.encoder_bn2 = nn.BatchNorm2d(16)
        self.encoder_bn3 = nn.BatchNorm2d(16)
        self.encoder_bn4 = nn.BatchNorm2d(16)
        self.encoder_bn5 = nn.BatchNorm2d(16)

        self.ETE = ETE(5, ETE_dim, node, lamda, SE)
        
        self.bn1 = nn.BatchNorm2d(ETE_dim)
        self.bn2 = nn.BatchNorm2d(ETN_dim)
        self.bn3 = nn.BatchNorm2d(NTG_dim)

        self.dropout1 = nn.Dropout(p=dropout)
        self.dropout2 = nn.Dropout(p=dropout)
        self.dropout3 = nn.Dropout(p=dropout)

        self.ETN = ETN(ETE_dim, ETN_dim, node)
        self.NTG = NTG(ETN_dim, NTG_dim, node)

        self.fc = nn.Linear(NTG_dim, 1)
        nn.init.constant_(self.fc.bias, 0)

    def forward(self, x):
        ## Cross-Attention Mechanism Part
        h_list = []
        for i in range(5):
            # Use different bn layers
            bn_layer = getattr(self, f'encoder_bn{i+1}')

            h = self.encoder(x[:, i])[0]
            h = bn_layer(h.permute(0, 2, 1)[..., None])[..., 0].permute(0, 2, 1)
            h_list.append(h)
        # Initialize similarity matrix
        S_sum = 0

        # Calculate the similarity matrix between each pair of channels and sum them up
        for i in range(5):
            for j in range(i + 1, 5):
                S1 = h_list[i] @ h_list[j].transpose(-1, -2)
                S2 = h_list[j] @ h_list[i].transpose(-1, -2)
                S_sum += (S1 + S2) / 2

        # Calculate the final similarity matrix
        S = S_sum / 10
        S_w = torch.softmax(S,dim=1)  # Combined similarity matrix of 5 channels        

        # Add a dimension in the second axis
        S_w = S_w.unsqueeze(1)
        x1 = x * S_w
        
        ## Convolution Part (ETE, ETN, NTG)
        x1 = self.ETE(x1)
        x1 = self.bn1(x1)
        x1 = F.relu(x1)
        x1 = self.dropout1(x1)

        x1 = self.ETN(x1)
        x1 = self.bn2(x1)
        x1 = F.relu(x1)
        x1 = self.dropout2(x1)
        
        x1 = self.NTG(x1)
        x1 = self.bn3(x1)
        x1 = F.relu(x1)
        x1 = self.dropout3(x1)

        x1 = x1.view(x1.size(0),-1)
        x1 = self.fc(x1)

        return x1


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)

if __name__ == '__main__':
    # Example of how to initialize and use the model
    
    # Model parameters (these would typically be arguments or read from a config)
    ETE_dim = 8
    ETN_dim = 32
    NTG_dim = 64
    node = 164
    lamda = 0.5
    SE = 4
    dropout = 0.5
    
    # Create a model instance
    model = MB_GAT_wFusion(
        ETE_dim=ETE_dim,
        ETN_dim=ETN_dim,
        NTG_dim=NTG_dim,
        node=node,
        lamda=lamda,
        SE=SE,
        dropout=dropout
    )
    
    # Apply weight initialization
    model.apply(weights_init)
    
    # Print model summary
    print(model)
    
    # Create a dummy input tensor
    # Batch size = 4, Channels = 5, Height = 164, Width = 164
    dummy_input = torch.randn(4, 5, 164, 164)
    
    # Perform a forward pass
    print("\nTesting forward pass...")
    output = model(dummy_input)
    
    # Print output shape
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    assert output.shape == (4, 1), "Output shape is incorrect!"
    print("Forward pass successful!")