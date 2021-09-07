import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import heartpy as hp
import scipy.signal as sig
from scipy import signal
from sklearn.decomposition import FastICA, PCA

# Butterworth forward-backward band-pass filter
def bandpass(signal, fs, order, fc_low, fc_hig, debug=False):
    """Butterworth forward-backward band-pass filter.

    :param signal: list of ints or floats; The vector containing the signal samples.
    :param fs: float; The sampling frequency in Hz.
    :param order: int; The order of the filter.
    :param fc_low: int or float; The lower cutoff frequency of the filter.
    :param fc_hig: int or float; The upper cutoff frequency of the filter.
    :param debug: bool, default=False; Flag to enable the debug mode that prints additional information.

    :return: list of floats; The filtered signal.
    """
    nyq = 0.5 * fs  # Calculate the Nyquist frequency.
    cut_low = fc_low / nyq  # Calculate the lower cutoff frequency (-3 dB).
    cut_hig = fc_hig / nyq  # Calculate the upper cutoff frequency (-3 dB).
    bp_b, bp_a = sig.butter(order, (cut_low, cut_hig), btype="bandpass")  # Design and apply the band-pass filter.
    bp_data = list(sig.filtfilt(bp_b, bp_a, signal))  # Apply forward-backward filter with linear phase.
    return bp_data

# Fast Fourier Transform
def fft(data, fs, scale="mag"):
   # Apply Hanning window function to the data.
   data_win = data * np.hanning(len(data))
   if scale == "mag":  # Select magnitude scale.
     mag = 2.0 * np.abs(np.fft.rfft(tuple(data_win)) / len(data_win))  # Single-sided DFT -> FFT
   elif scale == "pwr":  # Select power scale.
     mag = np.abs(np.fft.rfft(tuple(data_win)))**2  # Spectral power
   bin = np.fft.rfftfreq(len(data_win), d=1.0/fs)  # Calculate bins, single-sided
   return bin, mag


fps = 60
source = 'rPPGVideo.xlsx'

df = pd.read_excel(source, index_col=0)
df_list = df.values.tolist()

S = np.c_[df_list[0], df_list[1], df_list[2]]
P = S
S += 0.2 * np.random.normal(size=S.shape)  # Add noise

S /= S.std(axis=0)  # Standardize data
# Mix data
A = np.array([[1, 1, 1], [0.5, 2, 1.0], [1.5, 1.0, 2.0]])  # Mixing matrix
X = np.dot(S, A.T)  # Generate observations

# Compute ICA
ica = FastICA(n_components=3)
S_ = ica.fit_transform(P)  # Reconstruct signals
A_ = ica.mixing_  # Get estimated mixing matrix

bp_r_plot0 = bandpass(S_[:, 0], fps, 2, 0.5, 2.5)  # Heart Rate : 60-100 bpm (1-1.7 Hz), taking 30-150 (0.5 - 2.5)
bp_r_plot1 = bandpass(S_[:, 1], fps, 2, 0.5, 2.5)  # Heart Rate : 60-100 bpm (1-1.7 Hz), taking 30-150 (0.5 - 2.5)
bp_r_plot2 = bandpass(S_[:, 2], fps, 2, 0.5, 2.5)  # Heart Rate : 60-100 bpm (1-1.7 Hz), taking 30-150 (0.5 - 2.5)

# Calculate and display FFT
X_fft0, Y_fft0 = fft(bp_r_plot0, fps, scale="mag")
X_fft1, Y_fft1 = fft(bp_r_plot1, fps, scale="mag")
X_fft2, Y_fft2 = fft(bp_r_plot2, fps, scale="mag")
fig = plt.figure()
plt.plot(X_fft0, Y_fft0, 'b')
plt.plot(X_fft1, Y_fft1, 'g')
plt.plot(X_fft2, Y_fft2, 'r')
# fig.savefig('FFTplotExcel.png', dpi=100)
# plt.show()

working_data0, measures0 = hp.process(bp_r_plot0, fps)
working_data1, measures1 = hp.process(bp_r_plot1, fps)
working_data2, measures2 = hp.process(bp_r_plot2, fps)
plot_object0 = hp.plotter(working_data0, measures0, show=False)
plot_object1 = hp.plotter(working_data1, measures1, show=False)
plot_object2 = hp.plotter(working_data2, measures2, show=False)
plot_object0.savefig('bpmPlotOriginal0.png', dpi=100)
plot_object1.savefig('bpmPlotOriginal1.png', dpi=100)
plot_object2.savefig('bpmPlotOriginal2.png', dpi=100)
plt.show()

'''fig = plt.figure()
models = [X, S, S_]
names = ['mixtures', 'real sources', 'predicted sources']
colors = ['red', 'blue', 'orange']
for i, (name, model) in enumerate(zip(names, models)):
    plt.subplot(4, 1, i + 1)
    plt.title(name)
    for sig, color in zip(model.T, colors):
        plt.plot(sig, color=color)

fig.tight_layout()
plt.show()'''