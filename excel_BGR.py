import matplotlib.pyplot as plt
import numpy as np
import heartpy as hp
import pandas as pd
from scipy import signal
import scipy.signal as sig
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support


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
        mag = np.abs(np.fft.rfft(tuple(data_win))) ** 2  # Spectral power
    bin = np.fft.rfftfreq(len(data_win), d=1.0 / fs)  # Calculate bins, single-sided
    return bin, mag


# Set FPS, Get source_mp4 Video, Ground truth values
fps = 60
original_data = '01-base PPG.csv'
df_original_HR = pd.read_csv(original_data, index_col=None)

# Get Detected raw BGR values and Heart rate values
detected_BGR = 'rPPGVideo.xlsx'
detected_data = 'Heartrate_video.xlsx'
df_BGR = pd.read_excel(detected_BGR, index_col=0)
Blue, Green, Red = df_BGR['Blue mean'], df_BGR['Green mean'], df_BGR['Red mean']
df_detected_HR = pd.read_excel(detected_data, index_col=None)

# 2nd order butterworth bandpass filtering
bp_r_plot = bandpass(Red, fps, 2, 0.5, 2.5)  # Heart Rate : 60-100 bpm (1-1.7 Hz), taking 30-150 (0.5 - 2.5)
bp_g_plot = bandpass(Green, fps, 2, 0.5, 2.5)  # Heart Rate : 60-100 bpm (1-1.7 Hz)
bp_b_plot = bandpass(Blue, fps, 2, 0.5, 2.5)  # Heart Rate : 60-100 bpm (1-1.7 Hz)
plt.plot(Blue, 'b', label='Blue')
plt.plot(Green, 'g', label='Green')
plt.plot(Red, 'r', label='Red')
# plt.plot(bp_r_plot, 'r', label='BPFiltered_red')
plt.plot(bp_g_plot, 'g', label='BPFiltered_green')
# plt.plot(bp_b_plot, 'b', label='BPFiltered_Blue')
plt.title("Raw and Filtered Signals")
# plt.legend()
# plt.show()

# Calculate and display FFT
X_fft, Y_fft = fft(bp_g_plot, fps, scale="mag")
fig2 = plt.figure(2)
plt.plot(X_fft, Y_fft)
plt.title("FFT of filtered Signal")
fig2.savefig('FFTplotVideo.png', dpi=100)
# plt.show()

# Welch's Periodogram
f_set, Pxx_den = signal.welch(bp_g_plot, fps)
fig3 = plt.figure(3)
plt.semilogy(f_set, Pxx_den)
plt.ylim([0.5e-3, 1])
plt.xlabel('frequency [Hz]')
plt.ylabel('PSD [V**2/Hz]')
plt.title("Welchplot")
fig3.savefig('WelchplotVideo.png', dpi=100)
# plt.show()

# Calculate Heart Rate and Plot
working_data, measures = hp.process(bp_g_plot, fps)
plot_object = hp.plotter(working_data, measures, show=False, title='Final_Heart Rate Signal Peak Detection')
plot_object.savefig('bpmPlotVideo.png', dpi=100)

# Calculate and display original FFT
X_fft, Y_fft = fft(df_original_HR['Signal'], fps, scale="mag")
fig4 = plt.figure(4)
plt.plot(X_fft, Y_fft)
plt.title("FFT of Original Signal")
fig4.savefig('Original_FFTplotVideo.png', dpi=100)
# plt.show()

# Original Welch's Periodogram
f_set, Pxx_den = signal.welch(df_original_HR['Signal'], fps)
fig5 = plt.figure(5)
plt.semilogy(f_set, Pxx_den)
plt.ylim([0.5e-3, 1])
plt.xlabel('frequency [Hz]')
plt.ylabel('PSD [V**2/Hz]')
plt.title("Original_Welchplot")
fig5.savefig('Original_WelchplotVideo.png', dpi=100)
# plt.show()

# Calculate Original Heart Rate and Plot
hrdata = hp.get_data(original_data, column_name='Signal')
timerdata = hp.get_data(original_data, column_name='Time')
working_data1, measures1 = hp.process(hrdata, hp.get_samplerate_mstimer(timerdata))
plot_object1 = hp.plotter(working_data1, measures1, show=False, title='Original_Heart Rate Signal Peak Detection')
plot_object1.savefig('bpmPlotOriginal.png', dpi=100)
plt.show()

# calculate Evaluation metrics
Accuracy = accuracy_score(df_original_HR['Peaks'][0:df_detected_HR['Peaks'].size], df_detected_HR['Peaks'])
Metrics = precision_recall_fscore_support(df_original_HR['Peaks'][0:df_detected_HR['Peaks'].size],
                                          df_detected_HR['Peaks'])
Precision, Recall, f1_score = Metrics[0][0], Metrics[1][0], Metrics[2][0]