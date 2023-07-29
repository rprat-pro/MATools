import os
import tkinter
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
import mamemory


class ApplicationBasic():
	'''Application principale'''
	def __init__(self):
		'''constructeur'''
		self.fen = Tk()
		self.fen.title('MAVisualizator')
		self.fen.resizable(True, True)
		self.fen.geometry('600x600')
		self.toolbar=Frame(self.fen, background="#d5e8d4", height=60)
		self.toolbar.grid(row=0, column=0, sticky="ew")

		self.bou_MATimer = Button(self.toolbar)
		self.bou_MATimer.config(text='MATimer', command=self.matimer)
		self.bou_MATimer.grid(row=0,column=0)

		self.bou_MATrace = Button(self.toolbar)
		self.bou_MATrace.config(text='MATrace', command=self.matrace)
		self.bou_MATrace.grid(row=0,column=1)

		self.bou_MAMemory = Button(self.toolbar)
		self.bou_MAMemory.config(text='MAMemory', command=self.mamemory)
		self.bou_MAMemory.grid(row=0,column=2)

		self.bou_kat = Button(self.toolbar)
		self.bou_kat.config(text='Katherine', command=self.kat)
		self.bou_kat.grid(row=0,column=3)

		self.bou_quitter = Button(self.toolbar)
		self.bou_quitter.config(text='Exit', command=self.fen.destroy)
		self.bou_quitter.grid(row=0,column=4)

		self.doc=Label(self.fen)
#		content = 
#		self.doc.config(text=open('../README.md','r').read(), justify="left")
#		self.doc.grid(row=1,column=0,sticky='ns')
#		vsb = Scrollbar(self.doc, orient="vertical")
#		vsb.grid(row=2,column=0,sticky='ns')	
		#Create a canvas object
		doc=Canvas(self.fen, width= 800, height=800)
		content = open('../README.md','r').read()
		doc.create_text(0, 60, text=content, fill="black", font=('Helvetica 15 bold'))
		doc.grid(row=1,column=0,sticky='ns')

#		scrollbar = Scrollbar(self.doc)
#		scrollbar.pack(side=RIGHT, fill=Y)
#		scrollbar.config(command=self.doc.yview)


	def run(self):
		self.fen.mainloop()


	def print_timetable(self):
		f = open(self.filename, 'r')
		MATimerPrintTT = Tk()
		MATimerPrintTT.title(self.filename)
		MATimerPrintTT.resizable(True, True)
		MATimerPrintTT.geometry('600x600')
		
		content = f.read()
		print_tt = Label(MATimerPrintTT)
		print_tt.config(text=content, justify="left")
		print_tt.pack()

	def plot_mamemory(self):
		MAMemoryWin = mamemory.mamemory(self.filename)
		MAMemoryWin.plot_mamemory()
	
	def select_matimer_file(self):
		filetypes = (
					('text files', '*.perf'),
					('All files', '*.*')
					)
		self.filename = fd.askopenfilename(
					title='Open a file',
					initialdir=os.getcwd(),
					filetypes=filetypes)
	
	def select_mamemory_file(self):
		filetypes = (
					('text files', '*.mem'),
					('All files', '*.*')
					)
		self.filename = fd.askopenfilename(
					title='Open a file',
					initialdir=os.getcwd(),
					filetypes=filetypes)

	def matimer(self):
		self.select_matimer_file()
		self.print_timetable()

	def kat(self):
		'''Action sur un bouton'''
		tkinter.messagebox.showinfo(title="La vie est belle",message='Casse toi!!!')


	def matrace(self):
		'''Action sur un bouton'''
		tkinter.messagebox.showinfo(title="MATrace",message='Not Implemented!!!')

	def mamemory(self):
		'''Action sur un bouton'''
		self.select_mamemory_file()
		self.plot_mamemory()


if __name__ == '__main__':
	app = ApplicationBasic()
	app.run()

