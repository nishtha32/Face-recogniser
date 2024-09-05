import tkinter as tk
from tkinter import messagebox
import cv2
import os
from PIL import Image,ImageTk
import numpy as np

from facerecogniser import generate_dataset
from trainclassifier import train_classifier
from datectface import detect_face


# Function to handle the training button click
def handle_train():
    try:
        train_classifier("data")
        messagebox.showinfo("Info", "Training completed successfully...")
        t1.delete(0, 'end')
        t2.delete(0, 'end')
        t3.delete(0, 'end')
    except Exception as e:
        messagebox.showerror("Error", f"Training failed: {e}")

# Function to handle the register button click
def handle_register():
    if t1.get() == "" or t2.get() == "" or t3.get() == "":
        messagebox.showerror("Error", "Please enter all the details")
        return
    try:
        user_id = int(t3.get())
        generate_dataset(user_id)
        messagebox.showinfo("Info", "Collecting samples is completed...")
    except ValueError:
        messagebox.showerror("Error", "Please enter a valid User ID")
    except Exception as e:
        messagebox.showerror("Error", f"Error during sample collection: {e}")


# Function to handle the recognize button click
def handle_recognize():
    try:
        detect_face()
    except Exception as e:
        messagebox.showerror("Error", f"Recognition failed: {e}")


window = tk.Tk()
window.title("Face Recognition System")
window.config(background='lavender')

l1= tk.Label(window,text="Name",font=("Times New Roman",30))
l1.grid(column=0,row=0)
t1=tk.Entry(window,width=50,bd=5)
t1.grid(column=1,row=0)

l2= tk.Label(window,text="Age",font=("Times New Roman",30))
l2.grid(column=0,row=1)
t2=tk.Entry(window,width=50,bd=5)
t2.grid(column=1,row=1)

l3= tk.Label(window,text="User Id",font=("Times New Roman",30))
l3.grid(column=0,row=2)
t3=tk.Entry(window,width=50,bd=5)
t3.grid(column=1,row=2)

b1= tk.Button(window,text="Register",font=("Times New Roman",30),bg='green',fg='white',command=handle_register)
b1.grid(column=0,row=4)

b2= tk.Button(window,text="Training",font=("Times New Roman",30),bg='green',fg='white',command=handle_train)
b2.grid(column=1,row=4)

b3= tk.Button(window,text="Detect Face",font=("Times New Roman",30),bg='red',fg='white',command=handle_recognize)
b3.grid(column=2,row=4)


window.geometry("800x300")
window.mainloop()

