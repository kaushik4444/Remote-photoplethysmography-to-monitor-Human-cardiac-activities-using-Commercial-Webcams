import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

data = pd.read_csv('01-base PPG.csv', usecols=["Time", "Peaks"], engine="python", dtype=float)
exp_data = pd.read_excel('rPPGLive.xlsx')
a = exp_data.to_numpy()
print(a)
plt.plot(np.array(data['Time']), np.array(data['Peaks']))
plt.show()


