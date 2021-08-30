import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

data = pd.read_csv('01-base PPG.csv', usecols=["Time", "Peaks"], engine="python", dtype=float)
data2 = data.loc[data['Peaks'] == 1]
time_stamps = data2['Time'].tolist()

import heartpy as hp
hrdata = hp.get_data('01-base PPG.csv', column_name='Signal')
timerdata = hp.get_data('01-base PPG.csv', column_name='Time')

working_data, measures = hp.process(hrdata, hp.get_samplerate_mstimer(timerdata))
plot_object = hp.plotter(working_data, measures, show=False)
plot_object.savefig('bpmPlotOriginal.png', dpi=100)
plt.show()
