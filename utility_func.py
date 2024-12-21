import torch

"""Utility functions"""
def vec2block_diag(x):
    batch_size, _, N = x.size()
    block_diag_list = torch.zeros((batch_size, 2*N, N), dtype=x.dtype, device=x.device)
    y = torch.ones(batch_size,1)
    for i in range(N):
        block_diag_list[:, 2*i:2*i+2, i] = x[:,:,i]

    return block_diag_list

def TwoWay(H, N, MLP_tx, MLP_rx, y_real_tx, y_real_rx):
    'Calculate final optimal outputs'
    # Transmitter side
    output_A_t = MLP_tx(y_real_tx)
    # output_A_t_norm = torch.norm(output_A_t, dim=1, keepdim=True)
    # output_A_t = output_A_t/output_A_t_norm # adjust the range of output so it works well with Sigmoid
    angle_A_t = torch.sigmoid(output_A_t[:,:N]) * (torch.pi / 2)
    angle_A_t = angle_A_t.unsqueeze(1)
    Pol_A_t = torch.hstack((torch.cos(angle_A_t), torch.sin(angle_A_t)))
    Pol_A_t_blk = vec2block_diag(Pol_A_t).to(torch.complex64)

    W_A_t = output_A_t[:,N:]
    W_A_t_norm = torch.norm(W_A_t, dim=1, keepdim=True)
    W_A_t = W_A_t/W_A_t_norm
    W_A_t = torch.complex(W_A_t[:,:N], W_A_t[:,N:])
    W_A_t = W_A_t.unsqueeze(-1)

    # Receiver side
    output_B_r = MLP_rx(y_real_rx)
    # output_B_r_norm = torch.norm(output_B_r, dim=1, keepdim=True)
    # output_B_r = output_B_r / output_B_r_norm
    angle_B_r = torch.sigmoid(output_B_r) * (torch.pi / 2)
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


def channel_generation_batch_tensor(batch_size, N_Tx, N_Rx):
    """
    Function generates channel with polarization using PyTorch tensors
    """
    # Generate real and imaginary parts separately using PyTorch's random module
    H_re_tensor = torch.randn((batch_size, 2*N_Rx, 2*N_Tx))
    H_im_tensor = torch.randn((batch_size, 2*N_Rx, 2*N_Tx))

    # Combine real and imaginary parts to create complex-valued tensor
    H_p_tensor = torch.complex(H_re_tensor, H_im_tensor)

    # Normalization
    sqrt_2_tensor = torch.sqrt(torch.tensor(2.0))
    H_p_tensor = H_p_tensor / sqrt_2_tensor.unsqueeze(-1).unsqueeze(-1)

    return H_p_tensor

def MISO_polarization_pilot_tensor_fixed(N_Tx, N_Rx, tau):
    """
    This function generates polarization vectors
    in the case where receiver polarization is fixed.
    """
    # initialization
    Pol_BS = torch.zeros((tau, 2, N_Tx))
    Pol_UE = torch.zeros((tau, 2, N_Rx))

    # generate random angles
    theta_BS_t = torch.randn(1, N_Tx) * (torch.pi / 2); theta_BS_t = theta_BS_t.expand(tau, 1, N_Tx)
    repeat_index = tau//4; truncate_index = 4-tau%4
    theta_BS_r = torch.randn(4, 1, N_Tx); theta_BS_r = theta_BS_r.repeat(repeat_index+1,1,1)[:-truncate_index]
    theta_UE_t = torch.randn(1, N_Rx) * torch.pi; theta_UE_t = theta_UE_t.expand(tau,1,N_Rx)
    theta_UE_r = torch.randn(1, N_Rx) * torch.pi; theta_UE_r = theta_UE_r.expand(tau,1,N_Rx)

    # vstack stacks ups cos(theta_tx) and sin(theta_tx) as column vectors
    Pol_BS_t = torch.hstack((torch.cos(theta_BS_t), torch.sin(theta_BS_t))).to(device)
    Pol_BS_r = torch.hstack((torch.cos(theta_BS_r), torch.sin(theta_BS_r))).to(device)
    Pol_UE_t = torch.hstack((torch.cos(theta_UE_t), torch.sin(theta_UE_t))).to(device)
    Pol_UE_r = torch.hstack((torch.cos(theta_UE_r), torch.sin(theta_UE_r))).to(device)

    Pol_BS_t_blk = vec2block_diag(Pol_BS_t).to(torch.complex64).to(device)
    Pol_BS_r_blk = vec2block_diag(Pol_BS_r).to(torch.complex64).to(device)
    Pol_UE_t_blk = vec2block_diag(Pol_UE_t).to(torch.complex64).to(device)
    Pol_UE_r_blk = vec2block_diag(Pol_UE_r).to(torch.complex64).to(device)

    Pol_BS_r_blk_T = torch.transpose(Pol_BS_r_blk, 1, 2).to(device)
    Pol_UE_r_blk_T = torch.transpose(Pol_UE_r_blk, 1, 2).to(device)

    return Pol_BS_t, Pol_BS_r, Pol_UE_t, Pol_UE_r, Pol_BS_t_blk, Pol_BS_r_blk_T, Pol_UE_t_blk, Pol_UE_r_blk_T


def generate_twoway_pilots(H, Pol_Tx_t_blk, Pol_Tx_r_blk_T, Pol_Rx_t_blk, Pol_Rx_r_blk_T, W_A_t, sim_parameters):
    # Get parameter values
    N, M, L, N0 = sim_parameters
    # Get batch_size
    batch_size = H.shape[0]
    # Initialize storage for pilots
    y_real_B = torch.zeros((batch_size,2,L))
    y_real_A = torch.zeros((batch_size,N,2,L))
    # Transpose channel matrix
    H_T = torch.transpose(H, 1, 2)

    """Start Pilot Generation"""
    for t in range(L):
        'Receiver observes L pilots'
        Heff = Pol_Rx_r_blk_T[t] @ H @ Pol_Tx_t_blk[t]
        y_noiseless_B = Heff @ W_A_t[t]
        noise_sqrt = torch.sqrt(N0)
        noise_real  = torch.randn((batch_size, 1, 1))
        noise_imag = torch.randn((batch_size, 1, 1))
        noise = torch.complex(noise_real, noise_imag).to(device)
        sqrt_2_tensor = torch.sqrt(torch.tensor(2.0)).to(device)
        noise = noise / sqrt_2_tensor.unsqueeze(-1).unsqueeze(-1)
        noise = noise * noise_sqrt
        y_B = y_noiseless_B + noise
        y_real_B[:,:,t] = torch.squeeze(torch.concatenate([y_B.real, y_B.imag], axis=2))

        'Transmitter observes L pilots'
        H_eff_T = Pol_Tx_r_blk_T[t] @ H_T @ Pol_Rx_t_blk[t]
        y_noiseless_A = H_eff_T
        noise_real  = torch.randn((batch_size, N, 1))
        noise_imag = torch.randn((batch_size, N, 1))
        noise = torch.complex(noise_real, noise_imag).to(device)
        noise = noise / sqrt_2_tensor.unsqueeze(-1).unsqueeze(-1)
        noise = noise * noise_sqrt
        y_A = y_noiseless_A + noise
        y_real_A[:,:,:,t] = torch.squeeze(torch.concatenate([y_A.real, y_A.imag], axis=2))

    'Calculate final optimal outputs'
    y_A_real = torch.transpose(y_real_A[:,:,0,:],1,2).reshape(batch_size,-1)
    y_A_imag = torch.transpose(y_real_A[:,:,1,:],1,2).reshape(batch_size,-1)
    y_real_A_flat = torch.concatenate([y_A_real, y_A_imag], axis=1).to(device)
    y_real_B_flat = torch.concatenate([y_real_B[:,0,:], y_real_B[:,1,:]], axis=1).to(device)

    return y_real_A_flat, y_real_B_flat

def MIMO_polarization_init(M, batch_size):
    """
    This function generates polarization vectors
    in the case where receiver polarization is fixed.
    """
    initB = torch.rand(1) * torch.pi
    # generate random angles
    theta_Br = torch.full((1, M), initB.item()); theta_Br = theta_Br.expand(batch_size, 1, M)
    # vstack stacks ups cos(theta_tx) and sin(theta_tx) as column vectors
    Pol_Br = torch.hstack((torch.cos(theta_Br), torch.sin(theta_Br)))
    # create polarization block diagonal matrix
    Pol_Br_blk = vec2block_diag(Pol_Br).to(torch.complex64).to(device)

    return Pol_Br_blk

def opt_pol_alg(H, joint_iteration=5):
    batch_size, M, N = H.shape
    M = M//2; N = N//2
    Pol_B_blk = MIMO_polarization_init(M, batch_size)
    for k in range(joint_iteration):
        angle_A, angle_B = [], []
        # Polarization determinant matrix for A
        H_PD_A = H.mH @ Pol_B_blk @ Pol_B_blk.mH @ H
        for A_iter in range(0, 2*N, 2):
            C = H_PD_A[:,A_iter:A_iter+2,A_iter:A_iter+2]
            eig_val, eig_vec = torch.linalg.eig(C.real)
            sorted_indices = torch.argsort(eig_val.real, descending=True)
            sorted_vec0 = eig_vec[:,0,:].gather(dim=1, index=sorted_indices).reshape(batch_size,1,2)
            sorted_vec1 = eig_vec[:,1,:].gather(dim=1, index=sorted_indices).reshape(batch_size,1,2)
            sorted_vec = torch.concatenate([sorted_vec0, sorted_vec1], dim=1)
            sorted_eigVec = sorted_vec[:,:,0].reshape(batch_size,2,1)
            angle = torch.angle(sorted_eigVec[:,0] + 1j*sorted_eigVec[:,1])
            angle_A.append(angle)
        angle_A_all = torch.hstack(angle_A).unsqueeze(1)
        Pol_A = torch.hstack((torch.cos(angle_A_all), torch.sin(angle_A_all)))
        Pol_A_blk = vec2block_diag(Pol_A).to(torch.complex64).to(device)

        # Polarization determinant matrix for B
        H_PD_B = H @ Pol_A_blk @ Pol_A_blk.mH @ H.mH
        for B_iter in range(0, 2*M, 2):
            C = H_PD_B[:,B_iter:B_iter+2,B_iter:B_iter+2]
            eig_val, eig_vec = torch.linalg.eig(C.real)
            sorted_indices = torch.argsort(eig_val.real, descending=True)
            sorted_vec0 = eig_vec[:,0,:].gather(dim=1, index=sorted_indices).reshape(batch_size,1,2)
            sorted_vec1 = eig_vec[:,1,:].gather(dim=1, index=sorted_indices).reshape(batch_size,1,2)
            sorted_vec = torch.concatenate([sorted_vec0, sorted_vec1], dim=1)
            sorted_eigVec = sorted_vec[:,:,0].reshape(batch_size,2,1)
            angle = torch.angle(sorted_eigVec[:,0] + 1j*sorted_eigVec[:,1])
            angle_B.append(angle)
        angle_B_all = torch.hstack(angle_B).unsqueeze(1)
        Pol_B = torch.hstack((torch.cos(angle_B_all), torch.sin(angle_B_all)))
        Pol_B_blk = vec2block_diag(Pol_B).to(torch.complex64).to(device)
    return Pol_A_blk, Pol_B_blk, angle_A_all, angle_B_all