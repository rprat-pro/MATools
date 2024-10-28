#!/usr/bin/python3
import pandas as pd
import numpy as np
import matplotlib as mlp
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import glob

# get files

filelist = glob.glob('MAMemoryFootprint.*.mem')
filelist.sort()
print(filelist)

# setup plot
fig, ax = plt.subplots()

fig.set_figheight(10)
fig.set_figwidth(10)
ax.set_ylabel('Memory Per MPI Process(KB)')
ax.set_xlabel('Labels')


for file in filelist:
	plotname = file
	plotname = plotname.replace('MAMemoryFootprint.','')
	plotname = plotname.replace('.mem','')
	data = pd.read_csv(file, sep='\s+',header=None)
	data = pd.DataFrame(data)
	xplot = data[0]
	yplot = data[1]/ float(plotname)
	plabel = "#mpi: " + plotname
	ax.plot(xplot,yplot, label=plabel)
	fig.autofmt_xdate(rotation=75)
	plt.tight_layout()

plt.legend()
namepng='memory.png'
plt.savefig(namepng)
