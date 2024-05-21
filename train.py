import torch
import torch.nn as nn 
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from time import gmtime, strftime
import matplotlib.pyplot as plt
import os
from utility_func import *
from model import *

# Device will determine whether to run the training on GPU or CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

"""Simulation and Hyper Parameters"""
# Simulation parameters
N = 64
M = 1
L = 20          # total number of pilots
SNR_dB = torch.tensor(0)
P_dBm = 10**(SNR_dB/10)
SNR = 10**(SNR_dB/10)
P = 10**(P_dBm/10)
N0 = P/SNR

# Hyperparameters
batch_size = 1024
n_epochs = 20 
max_epochs = 50 

# Define parameters for NN 
input_sizeTx, input_sizeRx = 2*N*L, 2*L
output_sizeTx, output_sizeRx = 3*N, M  
sim_parameters = (N, M, L, N0)

"""Training Dataset"""
# Polarization vectors used in pilot stage
Pol_BS_t, Pol_BS_r, Pol_UE_t, Pol_UE_r, Pol_BS_t_blk, Pol_BS_r_blk_T, Pol_UE_t_blk, Pol_UE_r_blk_T \
    = MISO_polarization_pilot_tensor_fixed(N, M, L)

# Transmit beamformer used in the pilot stage
W_A_t_real = torch.randn((L, 1, N))
W_A_t_imag = torch.randn((L, 1, N))
W_A_t = torch.complex(W_A_t_real, W_A_t_imag)
W_A_t = W_A_t / torch.norm(W_A_t, dim=2, keepdim=True)
W_A_t = torch.transpose(W_A_t, 1, 2) * torch.sqrt(P)

"""batch data"""
# total number of generated samples
num_generated_sample = batch_size * 100 
# H_p_train generation
H_p_training_shaped = channel_generation_batch_tensor(num_generated_sample, N, M)
y_real_tx_train, y_real_rx_train = generate_twoway_pilots(H_p_training_shaped, 
                                                          Pol_BS_t_blk, 
                                                          Pol_BS_r_blk_T, 
                                                          Pol_UE_t_blk, 
                                                          Pol_UE_r_blk_T, 
                                                          W_A_t, 
                                                          sim_parameters)
"""Dev Dataset"""
# total number of generated samples
num_generated_dev_sample = 1000
# H_p_dev generation
H_p_dev_batch = channel_generation_batch_tensor(num_generated_dev_sample, N, M)
y_real_tx_test, y_real_rx_test = generate_twoway_pilots(H_p_dev_batch, 
                                                        Pol_BS_t_blk,
                                                        Pol_BS_r_blk_T, 
                                                        Pol_UE_t_blk, 
                                                        Pol_UE_r_blk_T,
                                                        W_A_t, 
                                                        sim_parameters)

"""Training"""
# Get parameters for the model 
MLP_tx_dim = [input_sizeTx, 256, 256, output_sizeTx]
MLP_rx_dim = [input_sizeRx, 128, 128, output_sizeRx]

# Calling the model 
model = BeamformingModel(MLP_tx_dim, MLP_rx_dim)
exp_id = strftime("%m-%d_%H_%M_%S", gmtime())
output_dir = f'/Users/seungcheoloh/Desktop/Primary/Research/DNN Applied P_MIMO/Two_Way_MISO_Method/trained_TwoWay_model/{exp_id}'
os.makedirs(output_dir, exist_ok=True)
print(output_dir)

# Setting learning rate and optimizers
learning_rate = 0.001
optimizer_bf = optim.Adam(model.parameters(), lr=learning_rate)
lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer_bf, gamma=0.9992)

# Initializer
no_increase = 0 
best_loss = float('inf')

# Initizalize lists for data visualization
training_loss = []
test_loss = []
epochs = []

for epoch in tqdm(range(max_epochs)):
    batch_iter = 0
    
    for epoch_per_batch in range(n_epochs):
        model.train()
        H_p_batch = channel_generation_batch_tensor(batch_size, N, M)
        y_tx_train, y_rx_train = generate_twoway_pilots(H_p_batch, 
                                                        Pol_BS_t_blk, 
                                                        Pol_BS_r_blk_T, 
                                                        Pol_UE_t_blk, 
                                                        Pol_UE_r_blk_T, 
                                                        W_A_t, 
                                                        sim_parameters)
        # Zeros the gradients
        optimizer_bf.zero_grad()

        # Forward pass
        bf = model(H_p_batch, y_tx_train, y_rx_train)
        loss = beamforming_loss(bf)
        train_loss_dB = 10*torch.log(-loss)

        # Backward propagation
        loss.backward()
        
        # Optimizer
        optimizer_bf.step()
        lr_scheduler.step()
        batch_iter += 1

    """Evaluating Model"""
    torch.no_grad()
    model.eval()
    dev_bf = model(H_p_dev_batch, y_real_tx_test, y_real_rx_test)
    dev_loss = beamforming_loss(dev_bf)
    dev_loss_dB = 10*torch.log(-dev_loss)

    epochs.append(epoch)
    dev_loss.append(dev_loss_dB.detach().numpy())
    training_loss.append(train_loss_dB.detach().numpy())
    print('epoch', epoch, '  loss_train:%2.5f' % train_loss_dB, '  loss_dev:%2.5f' % dev_loss_dB, '  best_dev:%2.5f  ' % dev_loss_dB, 'no_increase:', no_increase, f"lr: {lr_scheduler.get_lr()}")
    
    if dev_loss_dB > best_loss:
        torch.save(model.state_dict(), os.path.join(output_dir, f"{epoch}.pth"))
        best_loss = dev_loss_dB 
        no_increase = 0
    else: 
        no_increase = no_increase + 1

    if no_increase > 20:
        break

plt.plot(epochs, training_loss, label='Training Loss')
plt.plot(epochs, dev_loss, label='Test Loss')
plt.savefig(os.path.join(output_dir, f"training_loss.png"))
plt.savefig(os.path.join(output_dir, f"test_loss.png"))
plt.show()