#%% Import Packages

import tkinter as tk
from tkinter import ttk
# %% Class for Main Frame 

class MainFrame(ttk.Frame):
    def __init__(self, container):
        super().__init__(container)
        #field options 
        options = {'padx':5, 'pady':5}

        ## Set the grid configuration for flexibility for the app 
        self.columnconfigure(0, weight = 5)
        self.columnconfigure(1, weight = 5)
        self.rowconfigure(0, weight = 2)
        self.rowconfigure(1, weight = 6)
        self.rowconfigure(2, weight = 4)

    def input(self)
        
        ## Set the style for the app 

#%% Class App 

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        ## Set the title for the window 
        self.title('Business News')
        ## Set the size of the window
        self.geometry('1000x500')
        ## Control if the window will resize
        self.resizable(True,True)
        # Set the background colour for the app
        self.configure(bg = '#e6f2ff')

#%% Call the App 

if __name__ == '__main__':
    app = App()
    MainFrame(app)
    app.mainloop()
# %%
