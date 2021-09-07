import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import heartpy as hp

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


source_csv = '01-base PPG.csv'
fps = 60

# data = pd.read_csv('01-base PPG.csv', usecols=["Time", "Peaks"], engine="python", dtype=float)
# data2 = data.loc[data['Peaks'] == 1]
# time_stamps = data2['Time'].tolist()

hrdata = hp.get_data(source_csv, column_name='Signal')
timerdata = hp.get_data(source_csv, column_name='Time')

# Calculate and display FFT
X_fft, Y_fft = fft(hrdata, fps, scale="mag")
fig = plt.figure()
plt.plot(X_fft, Y_fft)
fig.savefig('FFTplotOriginal.png', dpi=100)
# plt.show()

working_data, measures = hp.process(hrdata, hp.get_samplerate_mstimer(timerdata))
plot_object = hp.plotter(working_data, measures, show=False)
plot_object.savefig('bpmPlotOriginal.png', dpi=100)
plt.show()
