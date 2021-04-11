from tkinter.scrolledtext import ScrolledText
from tkinter import *
from tkinter import filedialog
from functools import reduce
import numpy as np
import pandas as pd
from PIL import ImageTk, Image
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from math import sqrt
from tkinter import ttk
import pandas as pd
from PIL import ImageTk, Image
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from math import sqrt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt4agg import NavigationToolbar2QT
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)
from matplotlib.backend_bases import key_press_handler
from matplotlib.figure import Figure
import threading
from tkinter.ttk import Separator,Scrollbar,Treeview,Progressbar,Combobox
import pydot
import graphviz
import SVR_backend



window = Tk()
window.title("Support Vector Regression (SVR) model")



t2 = Text(window, height=1, width=35) ## Import MY  Data
t2.grid(row=0, column=2,sticky='ew')


def import_data():
   window.fileName = filedialog.askopenfilename \
                (filetypes=(("Python Stuff", ".csv"), \
                            ("All files","*.*")))

                
   
   t2.insert(END, window.fileName)
   
op_imgt=Image.open("import_data.png")
IMAGEt = op_imgt.resize((130,53), Image.ANTIALIAS)
imgt = ImageTk.PhotoImage(IMAGEt) 
###### ***** Create BUTTON to Import Training Data  *** ########
trainB=Button(window,border="0" ,image=imgt,width="130",height="52",command=import_data)
trainB.grid(row=0, column=3,padx=5)
   
def num_rows():
    
    df = pd.read_csv(t2.get("1.0",'end-1c'))
    count_row = df.shape[0]

    dt1.insert(END, count_row+1) 
 

def run_command():
    
    global Y_pred
    global X_test
    global Y_test
    
    if Kernel_function.get()=="rbf"  or Kernel_function.get()=="sigmoid":
        
        C=c_par.get()
        Gamma=gamma.get()
        Degree=1
        epsilon=eps.get()
        
    elif Kernel_function.get()=="poly":
        
        C=c_par.get()
        Gamma=gamma.get()
        Degree=degree.get()
        epsilon=eps.get()
        
    elif Kernel_function.get()=="linear":
        
        C=c_par.get()
        Gamma=1
        Degree=1
        epsilon=eps.get()
        
    Y_pred,Y_test,X_test,mape=SVR_backend.svr(t2.get("1.0",'end-1c'),   
                                        Kernel_function.get(),
                                        C,
                                        Gamma,
                                        Degree,
                                        epsilon
                                         )
    tmape_inversed.delete('1.0', END) 
    tmape_inversed.insert(END,mape)
        
def config():

    if Kernel_function.get()=="rbf"  or Kernel_function.get()=="sigmoid":
        
        exhc.config(state=NORMAL)
        exggg.config(state=NORMAL)
        
    elif Kernel_function.get()=="poly":
        
        exhc.config(state=NORMAL)
        exggg.config(state=NORMAL)
        exhdd.config(state=NORMAL)
        
    elif Kernel_function.get()=="linear":
        
        exhc.config(state=NORMAL)
        exggg.config(state=DISABLED)  
        exhdd.config(state=DISABLED)

def new_window():

    # Visualising the SVR results (for higher resolution and smoother curve)
    plt.scatter(X_test, Y_test, color = 'red')
    plt.plot(X_test, Y_pred, color = 'blue')
    plt.title('Truth or Bluff (SVR)')
    plt.xlabel('Position level')
    plt.ylabel('Salary')
    plt.show()

NEW = Button(window, text="Actual Vs Forecasted values graph",width=32,command=new_window)
NEW.grid(row=160, column=2,columnspan=2,pady=18) 
NEW.config(bg="gold")       
  
#########################################################################################
######### LABEL 
count = Button(window, text="Count_rows", width=20,command=num_rows)
count.grid(row=1, column=3)


dt=Label(window, text="Number of rows")
dt.grid(row=1, column=1)

dt1 = Text(window, height=1, width=10) ## NUMBER OF ROWS IN MY DATA
dt1.grid(row=1, column=2)


l2=Label(window, text="My Data",fg="blue")
l2.grid(row=0, column=1)
 
l3=Label(window, text="")
l3.grid(row=14, column=1)

lz=Label(window, text="")
lz.grid(row=5, column=2)  

####################################### Kernel function  ########################################################
    # Create a container to hold labels
buttons_frame5 = ttk.LabelFrame(window, text=' Kernel function ')
buttons_frame5.grid(row=4,column=2,pady=34,sticky='nwse')        # padx, pady
 
Kernel_function =StringVar()
Radiobutton(buttons_frame5, text = "poly", variable = Kernel_function,command=config, value = "poly").grid(row=1,column=1,padx=15)
Radiobutton(buttons_frame5, text = "rbf", variable = Kernel_function,command=config, value = "rbf").grid(row=1,column=2,padx=15)
Radiobutton(buttons_frame5, text = "linear", variable = Kernel_function,command=config, value = "linear").grid(row=1,column=3,padx=15)
Radiobutton(buttons_frame5, text = "Sigmoid", variable = Kernel_function,command=config, value = "sigmoid").grid(row=1,column=4,padx=15,pady=10)

Kernel_function.set("poly")

####################################### Grid search optimization ########################################################

####################################### Model parameters########################################################
    # Create a container to hold labels
buttons_frame4 = ttk.LabelFrame(window, text=' Model parameters ')
buttons_frame4.grid(row=5,column=2,padx=5,sticky='nwse')        # padx, pady

Label(buttons_frame4, text="").grid(row=1, column=0,padx=40)

Label(buttons_frame4, text="C:").grid(row=1, column=1, sticky='w',pady=6,padx=18)

c_par= DoubleVar()
exhc = Entry(buttons_frame4,width=9, textvariable=c_par) ##
exhc.grid(row=1, column=2, sticky='e')

##### parametre Gamma ########
Label(buttons_frame4, text="Gamma:").grid(row=2, column=1, sticky='w',pady=6,padx=18)


gamma= DoubleVar()
exggg = Entry(buttons_frame4,width=9, textvariable=gamma) ##
exggg.grid(row=2, column=2, sticky='e')
  
   ##### parametre degree ########
Label(buttons_frame4, text="Degree:").grid(row=3, column=1, sticky='w',pady=6,padx=18)

degree= IntVar()
exhdd = Entry(buttons_frame4,width=9, textvariable=degree) ##
exhdd.grid(row=3, column=2, sticky='e')


Label(buttons_frame4, text="Epsilon :").grid(row=4, column=1, sticky='w',pady=6,padx=18)

eps= DoubleVar()
epss = Entry(buttons_frame4,width=9, textvariable=eps) ##
epss.grid(row=4, column=2, sticky='e')

############# Forecasting ###############

###### ***** Create Radiobutton for selecting time lags option  *** ########"
# Create a container to hold labels
   
progress_bar = Progressbar(window, orient='horizontal',  length=500, mode='indeterminate')
progress_bar.grid(row=80,column=2,ipadx=8,columnspan=2,pady=13,sticky='nwse') 

def start_progressbar():
    global tread1
    progress_bar.start()
    tread1 = threading.Thread(target=run_command, args=())
    tread1.start()
    window.after(20, check_foo_thread)
    

def check_foo_thread():
    if tread1.is_alive():
        window.after(20, check_foo_thread)
    else:
        progress_bar.stop()

op_img111=Image.open("button.png")
IMAGE111 = op_img111.resize((150,50), Image.ANTIALIAS) #
img111 = ImageTk.PhotoImage(IMAGE111)      
      
Run_model = Button(window,border="0", text="Run model", width=20,command=start_progressbar,relief=GROOVE)
Run_model.grid(row=80, column=1)
Run_model.config(image=img111,width="170",height="50")  

########################################## 
Label(window, text="MAPE Testing").grid(row=159, column=2, sticky=W)

tmape_inversed = Text(window, height=1, width=20) ## Calculate MAPE
tmape_inversed.grid(row=159, column=3) 

####################################################
#Button(window, text="Back to main window ",bg="yellow",command=back_to_main_wind,relief=GROOVE).grid(row=0, column=4)
window.geometry("980x530")
window.mainloop()




