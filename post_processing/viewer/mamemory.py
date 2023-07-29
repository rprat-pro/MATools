import tkinter as tk
from tkinter import * #Pour python3.x Tkinter devient tkinter
from tkinter import ttk
from tkinter import filedialog as fd
from tkinter.messagebox import showinfo
import pandas as pd
import numpy as np
import matplotlib as mlp
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as mamemory_plt

class mamemory:
	def __init__(self):
		'''rien'''

	def __init__(self,filename):
		self.filename = filename

	def set_bous(self):
		self.toolbar=tk.Frame(self.window, background="#d5e8d4", height=60)
		self.toolbar.grid(row=0, column=0, sticky="ew")
		self.toolbar.bou_save_png_mamemory = Button(self.toolbar)
		self.toolbar.bou_save_png_mamemory.config(text='Export to png', command=self.save_png_mamemory)
		self.toolbar.bou_save_png_mamemory.grid(row=0,column=0)
		self.toolbar.bou_save_pdf_mamemory = Button(self.toolbar)
		self.toolbar.bou_save_pdf_mamemory.config(text='Export to pdf', command=self.save_pdf_mamemory)
		self.toolbar.bou_save_pdf_mamemory.grid(row=0,column=1)

	def plot_mamemory(self):
		self.window = tk.Tk()
		self.window.title("self.window plot")
		self.window.resizable(True, True)
		self.window.geometry('800x800')
		self.set_bous()
		data = pd.read_csv(self.filename, sep='\s+',header=None)
		data = pd.DataFrame(data)
		x=data[0].astype(float)
		y=data[1].astype(float)/1E6
		fig, ax = mamemory_plt.subplots(figsize=(7,7))
		ax.grid(color='b', linestyle='-', linewidth=0.2)
		ax.plot(x,y,label='your data')
		tmp = FigureCanvasTkAgg(fig, self.window)
#		tmp.get_tk_widget().pack()
		tmp.get_tk_widget().grid(row=1,column=0)
		ax.set_ylabel('data cosumption (mb)')
		ax.set_xlabel('memory point')
		ax.legend()

	def save_png_mamemory(self):
		plt.savefig('MAMemory.png')

	def save_pdf_mamemory(self):
		plt.savefig('MAMemory.pdf')

	def switch_data_name(self):
		tkinter.messagebox.showinfo(title="Work in progess",message='Not implemented')
