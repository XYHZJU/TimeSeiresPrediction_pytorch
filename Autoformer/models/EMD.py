import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from PyEMD import EMD
from scipy.fftpack import fft
from scipy.signal import find_peaks
from scipy.interpolate import interp1d

df = pd.read_csv('./dataset/datagen/BTP.csv',delimiter=',')

BTP = df['OT'].to_numpy()
print(BTP[:30])

emd = EMD()
IMFs = emd(BTP)

print(IMFs.shape)
np.savetxt('./dataset/datagen/EMD.csv',IMFs.T,delimiter=',')

t = np.arange(0,BTP.shape[0])

N = IMFs.shape[0]+1

# Plot results
plt.subplot(N,1,1)
plt.plot(t, BTP, 'r')
plt.title("Input signal")
plt.xlabel("Time [s]")



for n, imf in enumerate(IMFs):
    plt.subplot(N,1,n+2)
    plt.plot(t, imf, 'g')
    plt.title("IMF "+str(n+1))
    plt.xlabel("Time [s]")

plt.tight_layout()
plt.savefig('simple_example')
plt.show()