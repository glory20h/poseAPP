import tkinter as tk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

import test4 as t4

elbow_1 = np.load('elbow_1.npy')
elbow_2 = np.load('elbow_2.npy')
X, Y = t4.DTW(elbow_1, elbow_2)

root = tk.Tk()

fig = plt.Figure()
ax = fig.add_subplot(111)
bar1 = FigureCanvasTkAgg(fig, root)
bar1.get_tk_widget().pack()
ax.plot(X)
ax.plot(Y)
ax.set_title('Comparison')

# How to update the contents of a FigureCanvasTkAgg
# ax.clear()
# ax.plot()

root.mainloop()