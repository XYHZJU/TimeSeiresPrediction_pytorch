import numpy as np
from PyEMD import EMD

def concatenate(matrix_x: np.ndarray, num_interval: int):
    num_length = matrix_x.shape[0]
    num_signal = matrix_x.shape[1]

    matrix_a = matrix_x[:num_interval, 1:]
    matrix_b = matrix_x[-num_interval:, :-1]

    vector_a = np.linspace(0, 1, num_interval+2)[1:-1].reshape(-1, 1)
    vector_u = np.ones(num_signal-1).reshape(-1, 1)

    # transition
    matrix_t_a = np.flipud(matrix_a) * np.dot(vector_a, vector_u.T)
    matrix_t_b = np.flipud(matrix_b) * np.dot(np.flipud(vector_a), vector_u.T)

    matrix_t = matrix_t_a + matrix_t_b

    matrix_z = np.zeros(num_interval).reshape(-1, 1)

    # concatenate transition with zeros
    matrix_t = np.concatenate([matrix_t, matrix_z], axis=1)

    # result
    matrix_r = np.concatenate([matrix_x, matrix_t], axis=0)
    matrix_r = matrix_r.flatten(order='F')
    matrix_r = matrix_r[:-num_interval].reshape(-1, 1)

    return matrix_r

def deconcatenate(matrix_r: np.ndarray, num_interval: int, num_signal: int):
    num_mode = matrix_r.shape[1]

    # fill zeros
    matrix_z = np.zeros([num_interval, num_mode])
    matrix_r = np.concatenate([matrix_r, matrix_z], axis=0)

    matrix_imf = matrix_r.reshape([-1, num_signal, num_mode], order='F')
    matrix_imf = matrix_imf[:-num_interval, :, :]
    matrix_imf = matrix_imf.transpose((0, 2, 1))

    return matrix_imf


if __name__ == '__main__':
    # toy examples

    # num_signal: the number of signals
    # num_length: the length of each signal
    num_signal = 15
    num_length = 500  

    # x: toy signals, to verify the function. each column is a signal, time-points are in the rows.
    x = np.kron(np.random.rand(num_signal, 1), np.ones((1, num_length))).T
    print(f'multi-dimensional signal shape:\t{x.shape}')

    # num_interval: the only parameter needed. Amount of samples that we will use between each column of our signals (columns of X) as the transition between signals. 
    num_interval = 50

    serilized_x = concatenate(x, num_interval).reshape(-1)
    print(f'concatenated signal shape:\t{serilized_x.shape}')

    serilized_imfs = EMD()(serilized_x).T
    print(f'concatenated imfs shape:\t{serilized_imfs.shape}')

    # imfs for each signal, [num_length, num_imf_modes, num_signal]
    imfs = deconcatenate(serilized_imfs, num_interval, num_signal)
    print(f'multi-dimensional imfs shape:\t{imfs.shape}')