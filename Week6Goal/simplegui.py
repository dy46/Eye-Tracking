# -*- coding: utf-8 -*-
# import time
# import threading
# from Tkinter import *
# import sys
# import os

# class App(Frame):
# 	def __init__(self, master):
# 		#initialize the frame
# 		Frame.__init__(self, master)
# 		self.grid()
# 	def resetbutton(self):
# 		button1 = Button(frame, text = "This is a button", command=self.startthread)
# 		button1.grid()
# 	def startthread(self):
# 		newthread = threading.Thread(target=self.printints)
# 		newthread.start()
# 	def printints(self):
# 		os.system('python HomographyObjectFind.py')

# #create window
# root = Tk()

# #modify root window
# root.title("Simple GUI")
# root.geometry("200x200")
# app = App(root)

# root.mainloop()



# import time
# import threading
# import os
# import sys
# try: import tkinter
# except ImportError:
#     import Tkinter as tkinter
#     import ttk
#     import Queue as queue
# else:
#     from tkinter import ttk
#     import queue

# class GUI_Core(object):

#     def __init__(self):
#         self.root = tkinter.Tk()

#         self.int_var = tkinter.IntVar()
#         progbar = ttk.Progressbar(self.root, maximum=4)
#         # associate self.int_var with the progress value
#         progbar['variable'] = self.int_var
#         progbar.pack()

#         self.label = ttk.Label(self.root, text='0/4')
#         self.label.pack()

#         self.b_start = ttk.Button(self.root, text='Start')
#         self.b_start['command'] = self.start_thread
#         self.b_start.pack()

#     def start_thread(self):
#         self.b_start['state'] = 'disable'
#         self.int_var.set(0) # empty the Progressbar
#         self.label['text'] = '0/4'
#         # create then start a secondary thread to run arbitrary()
#         self.secondary_thread = threading.Thread(target=arbitrary)
#         self.secondary_thread.start()
#         # check the Queue in 50ms
#         self.root.after(50, self.check_que)
#     def arbitrary(self):
# 		execfile('python HomographyObjectFind.py')
#     def check_que(self):
#         while True:
#             try: x = que.get_nowait()
#             except queue.Empty:
#                 self.root.after(25, self.check_que)
#                 break
#             else: # continue from the try suite
#                 self.label['text'] = '{}/4'.format(x)
#                 self.int_var.set(x)
#                 if x == 4:
#                     self.b_start['state'] = 'normal'
#                     break


# def func_a():
#     time.sleep(1) # simulate some work

# def func_b():
#     time.sleep(0.3)

# def func_c():
#     time.sleep(0.9)

# def func_d():
#     time.sleep(0.6)

# def arbitrary():
#     func_a()
#     que.put(1)
#     func_b()
#     que.put(2)
#     func_c()
#     que.put(3)
#     func_d()
#     que.put(4)

# que = queue.Queue()
# gui = GUI_Core() # see GUI_Core's __init__ method
# gui.root.mainloop()

# # import time
# # import threading

# # try: import tkinter
# # except ImportError:
# #     import Tkinter as tkinter
# #     import ttk
# # else: from tkinter import ttk

# # class GUI_Core(object):

# #     def __init__(self):
# #         self.root = tkinter.Tk()

# #         self.progbar = ttk.Progressbar(self.root)
# #         self.progbar.config(maximum=4, mode='indeterminate')
# #         self.progbar.pack()

# #         self.b_start = ttk.Button(self.root, text='Start')
# #         self.b_start['command'] = self.start_thread
# #         self.b_start.pack()

# #     def start_thread(self):
# #         self.b_start['state'] = 'disable'
# #         self.progbar.start()
# #         self.secondary_thread = threading.Thread(target=arbitrary)
# #         self.secondary_thread.start()
# #         self.root.after(50, self.check_thread)

# #     def check_thread(self):
# #         if self.secondary_thread.is_alive():
# #             self.root.after(50, self.check_thread)
# #         else:
# #             self.progbar.stop()
# #             self.b_start['state'] = 'normal'
# #     def arbitrary(self):
# # 		os.system('python HomographyObjectFind.py')

# # def func_a():
# #     time.sleep(1) # simulate some work

# # def func_b():
# #     time.sleep(0.3)

# # def func_c():
# #     time.sleep(0.9)

# # def func_d():
# #     time.sleep(0.6)

# # def arbitrary():
# #     func_a()
# #     func_b()
# #     func_c()
# #     func_d()

# # gui = GUI_Core()
# # gui.root.mainloop()

# simple GUI

from Tkinter import *
import tkFileDialog as filedialog
import sys
import os

import subprocess


class Application(Frame):
	""" A GUI application with three buttons."""
	def __init__(self, master):
		#initialize the frame
		Frame.__init__(self, master)
		self.grid()
		self.create_widgets()
	def create_widgets(self):
		#Create button which displays number of clicks
		# self.button1 = Button(self, text = "Total Clicks: 0")
		# self.button1["command"] = self.update_count
		# self.button1.grid()
		self.run_button = Button(self, text = "OPEN Program", command = self.callback)
		self.run_button.grid(row = 2, column = 0, sticky = W)
		self.run_button = Button(self, text = "OPEN Video", command = self.openvid)
		self.run_button.grid(row = 4, column = 0, sticky = W)

	def launch(self):
		os.system('python HomographyObjectFind.py')
	def callback(self):
		fileName = filedialog.askopenfilename(filetypes = (("Python files", "*.py"),("All files", "*.*"))) 
		os.system('time python '+str(fileName))
	def openvid(self):
		videoName = filedialog.askopenfilename(filetypes = (("Video files", "*.mp4"),("All files", "*.*"))) 
		os.system('vlc '+videoName)
		#os.system(fileName)
	# def update_count(self):
	# 	self.button_clicks += 1
	# 	self.button1["text"] = "Total Clicks" + str(self.button_clicks)
#create window
root = Tk()

#modify root window
root.title("Simple GUI")
root.geometry("200x200")
app = Application(root)

root.mainloop()
# app = Frame(root)
# app.grid()
# label = Label(app, text = "this is cool")
# label.grid()
# button1 = Button(app, text = "This is a button")
# button1.grid()

# button2 = Button(app)
# button2.grid()
# button2.configure(text="This is second button")

# button3 = Button(app)
# button3.grid()

# button3["text"] = "this is another button"
# #kick off the event loop
# root.mainloop()
