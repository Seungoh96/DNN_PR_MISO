import numpy as np
import torch
import math
# torch.manual_seed(42)


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


def flatten_input(H_p, batch_size):
    """
    Function flattens the input batch of H_p by separating the real and imaginary part first.
    """
    H_p_flatten_real = H_p.real.reshape(batch_size, -1)
    H_p_flatten_imag = H_p.imag.reshape(batch_size, -1)
    H_p_flatten = torch.concatenate([H_p_flatten_real, H_p_flatten_imag], axis=1)

    return H_p_flatten


def complex2real(H):
    """
    This function effectively converts the complex matrix into into real and imaginary for
    PyTorch to work with. 
    """
    H_real1 = torch.concatenate([H.real, H.imag])
    H_real2 = torch.concatenate([-H.imag, H.real])
    H_real = torch.concatenate([H_real1, H_real2], axis=1)

    return H_real


def ls_estimator(y, x):
    """
    y = h * x + n
    y: batch_size*m*l
    h: batch_size*m*n
    x: batch_size*n*l

    Output: h = y*x^H*(x*x^H)^-1
    """
    n, ell = x.shape[0], x.shape[1]
    x_H = np.transpose(x.conjugate())
    if ell < n:
        x_Hx = np.matmul(x_H, x)
        # print('Cond number:',np.linalg.cond(x_Hx))
        x_Hx_inv = np.linalg.inv(x_Hx)
        h = np.matmul(y, x_Hx_inv)
        h = np.matmul(h, x_H)
    elif ell == n:
        # print('Cond number:',np.linalg.cond(x))
        h = np.linalg.inv(x)
        h = np.matmul(y, h)
    else:
        xx_H = np.matmul(x, x_H)
        # print('Cond number:',np.linalg.cond(xx_H))
        xx_H_inv = np.linalg.inv(xx_H)
        h = np.matmul(y, x_H)
        h = np.matmul(h, xx_H_inv)
    return h


def make_blk_diag(x):
    """
    Function takes in stacked column vectors in to block_diagonal matrix.
    """
    X = torch.block_diag(*[vector for vector in x.T])
    return X.T


def dft_matrix_2d(N):
    # Create a 1D array of indices
    indices = torch.arange(N).float()

    # Create a matrix of shape (N, N) with elements e^(i * 2 * pi * (j + k) / N)
    omega_real = torch.cos((-2 * torch.pi / N) * indices.unsqueeze(1) @ indices.unsqueeze(0))
    omega_imag = torch.sin((-2 * torch.pi / N) * indices.unsqueeze(1) @ indices.unsqueeze(0))
    omega = torch.complex(omega_real, omega_imag)
    return omega


def make_pilot_symbols(N, tau):
    if tau < N:
        return dft_matrix_2d(N)[:,:tau]
    else:
        return dft_matrix_2d(tau)[:N,:]


def MISO_polarization_pilot_tensor(N_Tx, N_Rx, tau):
    """
    This function generates polarization vectors
    in the case where receiver polarization is fixed.
    """
    Pol_Tx = torch.zeros((tau, 2, N_Tx))
    Pol_Rx = torch.zeros((tau, 2, N_Rx))
    theta_tx = torch.randn(tau, 1, N_Tx) * torch.pi
    theta_rx = torch.randn(tau, 1, N_Rx) * torch.pi
    # vstack stacks ups cos(theta_tx) and sin(theta_tx) as column vectors
    Pol_Tx = torch.hstack((torch.cos(theta_tx), torch.sin(theta_tx)))
    Pol_Rx = torch.hstack((torch.cos(theta_rx), torch.sin(theta_rx)))
    Pol_Tx_blk = vec2block_diag(Pol_Tx).to(torch.complex64)
    Pol_Rx_blk = vec2block_diag(Pol_Rx).to(torch.complex64)

    return Pol_Tx, Pol_Rx, Pol_Tx_blk, Pol_Rx_blk


def MISO_polarization_pilot_tensor_fixed(N_Tx, N_Rx, tau):
    """
    This function generates polarization vectors
    in the case where receiver polarization is fixed.
    """
    # initialization 
    Pol_BS = torch.zeros((tau, 2, N_Tx))
    Pol_UE = torch.zeros((tau, 2, N_Rx))

    # generate random angles
    theta_BS_t = torch.randn(1, N_Tx) * torch.pi; theta_BS_t = theta_BS_t.expand(tau, 1, N_Tx)
    repeat_index = tau//4; truncate_index = 4-tau%4
    theta_BS_r = torch.randn(4, 1, N_Tx); theta_BS_r = theta_BS_r.repeat(repeat_index+1,1,1)[:-truncate_index]
    theta_UE_t = torch.randn(1, N_Rx) * torch.pi; theta_UE_t = theta_UE_t.expand(tau,1,N_Rx)
    theta_UE_r = torch.randn(1, N_Rx) * torch.pi; theta_UE_r = theta_UE_r.expand(tau,1,N_Rx)

    # vstack stacks ups cos(theta_tx) and sin(theta_tx) as column vectors
    Pol_BS_t = torch.hstack((torch.cos(theta_BS_t), torch.sin(theta_BS_t)))
    Pol_BS_r = torch.hstack((torch.cos(theta_BS_r), torch.sin(theta_BS_r)))
    Pol_UE_t = torch.hstack((torch.cos(theta_UE_t), torch.sin(theta_UE_t)))
    Pol_UE_r = torch.hstack((torch.cos(theta_UE_r), torch.sin(theta_UE_r)))

    Pol_BS_t_blk = vec2block_diag(Pol_BS_t).to(torch.complex64)
    Pol_BS_r_blk = vec2block_diag(Pol_BS_r).to(torch.complex64)
    Pol_UE_t_blk = vec2block_diag(Pol_UE_t).to(torch.complex64)
    Pol_UE_r_blk = vec2block_diag(Pol_UE_r).to(torch.complex64)

    Pol_BS_r_blk_T = torch.transpose(Pol_BS_r_blk, 1, 2)
    Pol_UE_r_blk_T = torch.transpose(Pol_UE_r_blk, 1, 2)

    return Pol_BS_t, Pol_BS_r, Pol_UE_t, Pol_UE_r, Pol_BS_t_blk, Pol_BS_r_blk_T, Pol_UE_t_blk, Pol_UE_r_blk_T


def PingPong_MISO_polarization_init(N_Tx, N_Rx, batch_number):
    """
    This function generates pilot in the case where receiver polarization is fixed.
    """
    Pol_Tx = torch.zeros((batch_number, 2, N_Tx))
    Pol_Rx = torch.zeros((batch_number, 2, N_Rx))
    theta_tx = torch.randn(batch_number, 1, N_Tx) * np.pi
    theta_rx = torch.randn(batch_number, 1, N_Rx) * np.pi
    # vstack stacks ups cos(theta_tx) and sin(theta_tx) as column vectors
    Pol_Tx = torch.hstack((np.cos(theta_tx), np.sin(theta_tx)))
    Pol_Rx = torch.hstack((np.cos(theta_rx), np.sin(theta_rx)))
    Pol_Tx_blk = vec2block_diag(Pol_Tx)
    Pol_Rx_blk = vec2block_diag(Pol_Rx)

    return Pol_Tx, Pol_Rx, Pol_Tx_blk, Pol_Rx_blk


def polarization_pilot_tensor(N_Tx, tau):
    Pol_Tx = torch.zeros((tau, 2, N_Tx))
    theta_tx = torch.randn(tau, 1, N_Tx) * 2 * np.pi
    # vstack stacks ups cos(theta_tx) and sin(theta_tx) as column vectors
    Pol_Tx = torch.hstack((np.cos(theta_tx), np.sin(theta_tx)))

    return Pol_Tx


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
        noise = torch.complex(noise_real, noise_imag)
        sqrt_2_tensor = torch.sqrt(torch.tensor(2.0))
        noise = noise / sqrt_2_tensor.unsqueeze(-1).unsqueeze(-1)
        noise = noise * noise_sqrt
        y_B = y_noiseless_B
        y_B = y_noiseless_B + noise
        y_real_B[:,:,t] = torch.squeeze(torch.concatenate([y_B.real, y_B.imag], axis=2))

        'Transmitter observes L pilots'
        H_eff_T = Pol_Tx_r_blk_T[t] @ H_T @ Pol_Rx_t_blk[t]
        y_noiseless_A = H_eff_T
        noise_sqrt = torch.sqrt(N0)
        noise_real  = torch.randn((batch_size, N, 1))
        noise_imag = torch.randn((batch_size, N, 1))
        noise = torch.complex(noise_real, noise_imag)
        sqrt_2_tensor = torch.sqrt(torch.tensor(2.0))
        noise = noise / sqrt_2_tensor.unsqueeze(-1).unsqueeze(-1)
        noise = noise * noise_sqrt
        y_A = y_noiseless_A + noise
        y_A = y_noiseless_A
        y_real_A[:,:,:,t] = torch.squeeze(torch.concatenate([y_A.real, y_A.imag], axis=2))

    'Calculate final optimal outputs'
    y_A_real = torch.transpose(y_real_A[:,:,0,:],1,2).reshape(batch_size,-1)
    y_A_imag = torch.transpose(y_real_A[:,:,1,:],1,2).reshape(batch_size,-1)
    y_real_A_flat = torch.concatenate([y_A_real, y_A_imag], axis=1)
    y_real_B_flat = torch.concatenate([y_real_B[:,0,:], y_real_B[:,1,:]], axis=1)

    return y_real_A_flat, y_real_B_flat

    

def vec2block_diag(x):
    block_diag_list = torch.stack([torch.block_diag(*batch.T).T for batch in x], dim=0)
    return block_diag_list


def sum_rate_loss(DNN_output_batch, H_p_shaped, Pt, N0, N_Tx, N_Rx):
    """
    Arguments:
    DNN_output_batch: tensor of size (batch_size, output_size)
    H_p_batch: tensor of size (batch_size, 2*N_Rx, 2*N_Tx)
    Pt: power constraint for BS
    N0: noise power for the specific SNR
    N_Tx: number of transmit antenna element
    N_Rx: number of receive antenna element

    Output: sum_rate_loss mean
    """

    batch_num = DNN_output_batch.shape[0]               # number of batch
    N = N_Tx; M = N_Rx                                  # number of transmitter and receivers
    rank = torch.min(torch.tensor(N),torch.tensor(M))   # rank of the system
    V_len = 2*N**2                                      # index for precoder


    """Output processing for Polarization Vectors"""
    # turning the range of first N+M outputs to 0 to pi
    Tx_angles = torch.sigmoid(DNN_output_batch[:,:N].unsqueeze(1))*torch.tensor(math.pi)
    Rx_angles = torch.sigmoid(DNN_output_batch[:,N:N+M].unsqueeze(1))*torch.tensor(math.pi)
    # use angles to stack polarization column vectors  
    Pol_Tx = torch.hstack((torch.cos(Tx_angles), torch.sin(Tx_angles)))
    Pol_Rx = torch.hstack((torch.cos(Rx_angles), torch.sin(Rx_angles)))
    # turn the polarization vectors to block diagonal matrix 
    Pol_Tx_blk = vec2block_diag(Pol_Tx).to(torch.complex64)
    Pol_Rx_blk = vec2block_diag(Pol_Rx)
    # find the transpose of Pol_Rx_blk 
    Pol_Rx_blk_T = torch.transpose(Pol_Rx_blk,1,2).to(torch.complex64)


    """Output processing for precoders and postcoders"""
    # allocate the data from the output 
    V = DNN_output_batch[:,N+M:N+M+V_len].unsqueeze(1).to(torch.float64)
    W = DNN_output_batch[:,N+M+V_len:].unsqueeze(1).to(torch.float64)
    # normalize the data to apply constraints
    V_norm = torch.norm(V, dim=2, keepdim=True) * torch.sqrt(Pt)
    W_norm = torch.norm(W, dim=2, keepdim=True) 
    V = V/V_norm; W = W/W_norm
    # make complex precoder, postcoder, consists of stacked column vectors for each antennas
    V_real = V[:,:,:N**2]; V_imag = V[:,:,N**2:]
    V_complex = torch.complex(V_real, V_imag)
    V_precoder = torch.transpose(V_complex.reshape(batch_num,N,N),1,2).to(torch.complex64)
    W_real = W[:,:,:M**2]; W_imag = W[:,:,M**2:]
    W_complex = torch.complex(W_real, W_imag)
    W_postcoder = W_complex.reshape(batch_num,M,M).to(torch.complex64)
    # transpose the postcoder to multiply 
    W_postcoder_T = torch.transpose(W_postcoder.conj(),1,2)


    """Compute Sum Rate"""
    # process the received signal through matrix multiplication
    processed_y = W_postcoder_T @ Pol_Rx_blk_T @ H_p_shaped @ Pol_Tx_blk @ V_precoder
    # calculate channel gain of each data stream
    channel_gains = torch.abs(torch.diagonal(processed_y, dim1=1, dim2=2))**2
    summed_channel_gains = torch.sum(channel_gains, dim=1, keepdim=True)
    capacity = torch.log2(1+channel_gains/N0)
    sum_rate = torch.sum(capacity, dim=1)
    sum_rate_loss = -(torch.mean(sum_rate))

    return sum_rate_loss

def beamforming_loss(H, LSTMA, LSTMB, MLP_A_t, MLP_A_r, MLP_B_t, MLP_B_r, MLP_A, MLP_B, NN_parameters, train_parameters):
    'Get parameter values'
    (_, _, hidden_sizeA, hidden_sizeB) = NN_parameters
    (batch_size, N, M, L, N0) = train_parameters
    H_her = torch.transpose(H.conj(), 1, 2)
    # hidden_sizeA = parameters['A_hidden']
    # hidden_sizeB = parameters['B_hidden']
    # batch_size = parameters['batch_size']
    # N = parameters['Tx_size']
    # M = parameters['Rx_size']
    # L = parameters['numofRounds']
    # N0 = parameters['noisePower']

    """Initialize the hidden unit and the cell state"""
    h_A = torch.zeros(([batch_size, hidden_sizeA]))
    c_A = torch.zeros(([batch_size, hidden_sizeA]))
    h_B = torch.zeros(([batch_size, hidden_sizeB]))
    c_B = torch.zeros(([batch_size, hidden_sizeB]))
    

    """Start the ping pong pilot rounds"""
    for t in range(L):
        if t == 0:
            """Initialize parameters for the first pilot stage"""
            # Initialize polarization vectors
            _, _, Pol_A_t_blk, Pol_B_r_blk = PingPong_MISO_polarization_init(N,M,batch_size)
            Pol_A_t_blk = Pol_A_t_blk.to(torch.complex64)
            Pol_A_r_blk = Pol_A_t_blk
            Pol_A_r_blk_T = torch.transpose(Pol_A_r_blk, 1, 2)
            Pol_B_r_blk_T = torch.transpose(Pol_B_r_blk, 1, 2).to(torch.complex64)
            # Initialize transmit vector from Agent A
            W_A_t_real = torch.randn((batch_size, 1, N))
            W_A_t_imag = torch.randn((batch_size, 1, N))
            W_A_t = torch.complex(W_A_t_real, W_A_t_imag)
            W_A_t = W_A_t / torch.norm(W_A_t, dim=2, keepdim=True)
            W_A_t = torch.transpose(W_A_t, 1, 2)
            # Initialize receive vector from Agent A 
            W_A_r_real = torch.randn((batch_size, 1, N))
            W_A_r_imag = torch.randn((batch_size, 1, N))
            W_A_r = torch.complex(W_A_r_real, W_A_r_imag)
            W_A_r = W_A_r / torch.norm(W_A_r, dim=2, keepdim=True)
            W_A_r_Her = W_A_r.conj()

        """Agent B observes the measurement"""
        y_noiseless_B = Pol_B_r_blk_T @ H @ Pol_A_t_blk @ W_A_t
        noise_sqrt = torch.sqrt(N0)
        noise_real  = torch.randn((batch_size, 1, 1))
        noise_imag = torch.randn((batch_size, 1, 1))
        noise = torch.complex(noise_real, noise_imag)
        sqrt_2_tensor = torch.sqrt(torch.tensor(2.0))
        noise = noise / sqrt_2_tensor.unsqueeze(-1).unsqueeze(-1)
        noise = noise * noise_sqrt
        y_B = y_noiseless_B + noise
        y_real_B = torch.squeeze(torch.concatenate([y_B.real, y_B.imag], axis=2))

        """Agent B design the next receive polarization"""
        h_B, c_B = LSTMB((y_real_B, h_B, c_B))
        output_B_r = MLP_B_r(h_B)
        angle_B_r = torch.sigmoid(output_B_r) * torch.pi
        Pol_B_r = torch.hstack((torch.cos(angle_B_r), torch.sin(angle_B_r)))
        Pol_B_r_blk = vec2block_diag(Pol_B_r)
        Pol_B_r_blk_T = torch.transpose(Pol_B_r_blk, 1, 2)

        """Agent B design the next transmit polarization"""
        output_B_t = MLP_B_t(h_B)
        angle_B_t = torch.sigmoid(output_B_t) * torch.pi
        Pol_B_t = torch.hstack((torch.cos(angle_B_t), torch.sin(angle_B_t)))
        Pol_B_t_blk = vec2block_diag(Pol_B_t).to(torch.complex64)

        """Agent A observes the measurement"""
        y_noiseless_A = W_A_r_Her @ Pol_A_r_blk_T @ H_her @ Pol_B_t_blk
        noise_sqrt = torch.sqrt(N0)
        noise_real  = torch.randn((batch_size, 1, 1))
        noise_imag = torch.randn((batch_size, 1, 1))
        noise = torch.complex(noise_real, noise_imag)
        sqrt_2_tensor = torch.sqrt(torch.tensor(2.0))
        noise = noise / sqrt_2_tensor.unsqueeze(-1).unsqueeze(-1)
        noise = noise * noise_sqrt
        y_A = y_noiseless_A + noise
        y_real_A = torch.squeeze(torch.concatenate([y_A.real, y_A.imag], axis=2))

        """Agent A design the next receive polarization and beamformer"""
        print(y_real_A.shape)
        print(h_A.shape)
        print(c_A.shape)

        h_A, c_A = LSTMA((y_real_A, h_A, c_A))
        output_A_r = MLP_A_r(h_A)
        angle_A_r = torch.sigmoid(output_A_r[:,:,:N]) * torch.pi
        Pol_A_r = torch.hstack((torch.cos(angle_A_r), torch.sin(angle_A_r)))
        Pol_A_r_blk = vec2block_diag(Pol_A_r)
        Pol_A_r_blk_T = torch.transpose(Pol_A_r_blk,1,2)

        W_A_r = output_A_r[:,:,N:]
        W_A_r_norm = torch.norm(W_A_r, dim=2, keepdim=True)
        W_A_r = W_A_r/W_A_r_norm
        W_A_r = torch.complex(W_A_r[:,:,:N],W_A_r[:,:,N:])
        W_A_r_Her = W_A_r.conj()

        """Agent A design the next transmit polarization and beamformer"""
        output_A_t = MLP_A_t(h_A)
        angle_A_t = torch.sigmoid(output_A_t[:,:,:N]) * torch.pi
        Pol_A_t = torch.hstack((torch.cos(angle_A_t), torch.sin(angle_A_t)))
        Pol_A_t_blk = vec2block_diag(Pol_A_t)

        W_A_t = output_A_t[:,:,N:]
        W_A_t_norm = torch.norm(W_A_t, dim=2, keepdim=True)
        W_A_t = W_A_t/W_A_t_norm
        W_A_t = torch.complex(W_A_t[:,:,:N],W_A_t[:,:,N:])
        W_A_t = torch.transpose(W_A_r, 1, 2)

    'Calculate final optimal outputs'
    # Agent A side 
    output_A_t = MLP_A(c_A)
    angle_A_t = torch.sigmoid(output_A_t[:,:,:N]) * torch.pi
    Pol_A_t = torch.hstack((torch.cos(angle_A_t), torch.sin(angle_A_t)))
    Pol_A_t_blk = vec2block_diag(Pol_A_t)
    W_A_t = output_A_t[:,:,N:]
    W_A_t_norm = torch.norm(W_A_t, dim=2, keepdim=True)
    W_A_t = W_A_t/W_A_t_norm
    W_A_t = torch.complex(W_A_t[:,:,:N],W_A_t[:,:,N:])

    # Agent B side
    output_B_r = MLP_B(c_B)
    angle_B_r = torch.sigmoid(output_B_r) * torch.pi
    Pol_B_r = torch.hstack((torch.cos(angle_B_r), torch.sin(angle_B_r)))
    Pol_B_r_blk = vec2block_diag(Pol_B_r)
    Pol_B_r_blk_T = torch.transpose(Pol_B_r_blk,1,2)

    bf_gain = torch.sum(torch.abs((Pol_B_r_blk_T @ H @ Pol_A_t_blk @ W_A_t)**2))
    bf_gain_loss = -bf_gain
    
    return bf_gain_loss
    

def main():

    if __name__ == "__main__":
        main()
