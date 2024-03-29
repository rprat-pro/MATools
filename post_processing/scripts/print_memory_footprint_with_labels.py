#!/usr/bin/python3
import pandas as pd
import numpy as np
import matplotlib as mlp
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# setup plot
fig, ax = plt.subplots()

fig.set_figheight(10)
fig.set_figwidth(10)
ax.set_ylabel('Memory(KB)')
ax.set_xlabel('Labels')

data = pd.read_csv('MAMemoryFootprint.mem', sep='\s+',header=None)
data = pd.DataFrame(data)

xplot = data[0]
yplot = data[1]

ax.plot(xplot,yplot)
fig.autofmt_xdate(rotation=75)
plt.tight_layout()
namepng='memory.png'
plt.savefig(namepng)
