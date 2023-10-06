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
minplot = data[1]
maxplot = data[2]
meanplot = data[3]

ax.plot(xplot, minplot, label="Min")
ax.plot(xplot, maxplot, label="Max")
ax.plot(xplot, meanplot, label="Average")
ax.legend()
fig.autofmt_xdate(rotation=75)
plt.tight_layout()
namepng='min_max_mean.png'
plt.savefig(namepng)
