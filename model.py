import torch 
import torch.nn as nn
import torch.optim as optim

"""Utility functions"""
def vec2block_diag(x):
    block_diag_list = torch.stack([torch.block_diag(*batch.T).T for batch in x], dim=0)
    return block_diag_list

def TwoWay(H, N, MLP_tx, MLP_rx, y_real_tx, y_real_rx):
    'Calculate final optimal outputs'
    # Transmitter side 
    output_A_t = MLP_tx(y_real_tx)
    angle_A_t = torch.sigmoid(output_A_t[:,:N]) * torch.pi
    angle_A_t = angle_A_t.unsqueeze(1)
    Pol_A_t = torch.hstack((torch.cos(angle_A_t), torch.sin(angle_A_t)))
    Pol_A_t_blk = vec2block_diag(Pol_A_t).to(torch.complex64)

    W_A_t = output_A_t[:,N:]
    W_A_t_norm = torch.norm(W_A_t, dim=1,keepdim=True)
    W_A_t = W_A_t/W_A_t_norm
    W_A_t = torch.complex(W_A_t[:,:N], W_A_t[:,N:])
    W_A_t = W_A_t.unsqueeze(-1)
    
    # Receiver side
    output_B_r = MLP_rx(y_real_rx)
    angle_B_r = torch.sigmoid(output_B_r) * torch.pi
    Pol_B_r = torch.hstack((torch.cos(angle_B_r), torch.sin(angle_B_r)))
    Pol_B_r = Pol_B_r.unsqueeze(-1)
    Pol_B_r_blk = vec2block_diag(Pol_B_r)
    Pol_B_r_blk_T = torch.transpose(Pol_B_r_blk, 1, 2).to(torch.complex64)

    Heff_final = Pol_B_r_blk_T @ H @ Pol_A_t_blk
    y_final = Heff_final @ W_A_t
    
    return y_final

def beamforming_loss(bf):
    bf_gain = torch.mean(torch.abs(bf)**2)
    return -bf_gain


"""MLP Model"""
class MLPBlock(nn.Module):
    def __init__(self, num_layers, dims):
        super().__init__()
        layers = []
        for i in range(num_layers - 2):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(dims[i+1]))

        layers.append(nn.Linear(dims[-2], dims[-1]))
        self.mlp = nn.Sequential(*layers)
    
    def forward(self, inputs):
        return self.mlp(inputs)
    
"""Beamforming Model Class"""
class BeamformingModel(nn.Module):

    def __init__(self, MLP_tx_dim, MLP_rx_dim):
        super().__init__()
        self.MLP_tx = MLPBlock(len(MLP_tx_dim), MLP_tx_dim)
        self.MLP_rx = MLPBlock(len(MLP_rx_dim), MLP_rx_dim)

    def forward(self, H, N, y_tx, y_rx):
        bf_loss = TwoWay(H, N, self.MLP_tx, self.MLP_rx, y_tx, y_rx)
        return bf_loss
