import tkinter as tk
from tkinter import ttk
import serial
import threading
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import joblib
from matplotlib.animation import FuncAnimation
from sklearn.preprocessing import StandardScaler

# --- Load model and label encoder ---
model = joblib.load('random_forest_model.pkl')
label_encoder = joblib.load('label_encoder.pkl')
print("Model and label encoder loaded successfully.")

# Serial config
serial_port = 'COM8'  # Update if needed
baud_rate = 9600

# Category mappings
category_colors = {
    "Normal": "green",
    "Mild": "yellow",
    "Moderate": "orange",
    "Severe": "red"
}

# Initialize buffer and timeline
data_buffer = {
    'GSR': [], 'Pulse': [], 'Temp': [],
    'Pressure': [], 'Altitude': [], 'Prediction': []
}
time_steps = []

# --- Serial Reading ---
def get_sensor_data(ser):
    try:
        line = ser.readline().decode('utf-8').strip()
        print("Received:", line)
        parts = line.split('|')
        gsr = float(parts[0].split(':')[1].strip())
        pulse = float(parts[1].split(':')[1].strip())
        temp = float(parts[2].split(':')[1].strip().split()[0])
        pressure = float(parts[3].split(':')[1].strip().split()[0])
        altitude = float(parts[4].split(':')[1].strip().split()[0])
        return {
            'GSR': gsr,
            'Pulse': pulse,
            'Temp': temp,
            'Pressure': pressure,
            'Altitude': altitude
        }
    except Exception as e:
        print("Parsing error:", e)
        return None

def read_data():
    try:
        ser = serial.Serial(serial_port, baud_rate, timeout=1)
        print(f"Connected to {serial_port}")
        while running:
            sensor_data = get_sensor_data(ser)
            if sensor_data:
                time_steps.append(time_steps[-1] + 1 if time_steps else 1)
                for key in sensor_data:
                    data_buffer[key].append(sensor_data[key])

                # Format for model
                df = pd.DataFrame([sensor_data])[['GSR', 'Pulse', 'Temp', 'Pressure', 'Altitude']]
                scaled_df = scaler.fit_transform(df)  # Use saved scaler if available
                prediction = model.predict(scaled_df)[0]
                label = label_encoder.inverse_transform([prediction])[0]
                category = str(label).capitalize()
                data_buffer['Prediction'].append(category)

                # Serial response
                if category.lower() == "normal":
                    ser.write(b'B\n')
                else:
                    ser.write(b'C\n')

                update_light(category)
                update_prediction_label(category)

                # Limit buffer size
                max_len = 50
                for key in data_buffer:
                    data_buffer[key] = data_buffer[key][-max_len:]
                time_steps[:] = time_steps[-max_len:]
    except serial.SerialException as e:
        print(f"Serial error: {e}")

# --- Start/Stop ---
def start_reading():
    global running
    if not running:
        running = True
        threading.Thread(target=read_data, daemon=True).start()

def stop_reading():
    global running
    running = False

# --- GUI Feedback ---
def update_light(category):
    color = category_colors.get(category, "white")
    root.configure(bg=color)

def update_prediction_label(category):
    prediction_label.config(text=f"Prediction: {category}")

# --- Plot Update ---
def update_plot(frame):
    if time_steps:
        ax.clear()
        ax.plot(time_steps, data_buffer['GSR'], label='GSR', color='blue')
        ax.plot(time_steps, data_buffer['Pulse'], label='Pulse', color='red')
        ax.plot(time_steps, data_buffer['Temp'], label='Temp', color='green')
        ax.plot(time_steps, data_buffer['Pressure'], label='Pressure', color='purple')
        ax.plot(time_steps, data_buffer['Altitude'], label='Altitude', color='brown')
        ax.set_title('Sensor Data Over Time')
        ax.legend()
        canvas.draw()

# --- GUI Setup ---
root = tk.Tk()
root.title("Real-Time Sensor Monitoring and Prediction")
root.geometry("850x600")

title_label = tk.Label(root, text="Sensor Data Monitoring", font=("Helvetica", 16))
title_label.pack(pady=10)

button_frame = ttk.Frame(root)
button_frame.pack(pady=10)

start_button = ttk.Button(button_frame, text="Start", command=start_reading)
start_button.grid(row=0, column=0, padx=5)

stop_button = ttk.Button(button_frame, text="Stop", command=stop_reading)
stop_button.grid(row=0, column=1, padx=5)

prediction_label = tk.Label(root, text="Prediction: N/A", font=("Helvetica", 14))
prediction_label.pack(pady=10)

# --- Matplotlib Plot ---
fig, ax = plt.subplots(figsize=(8, 4))
canvas = FigureCanvasTkAgg(fig, master=root)
canvas.get_tk_widget().pack()

# --- Animation ---
ani = FuncAnimation(fig, update_plot, interval=1000)

# --- Globals ---
running = False
scaler = joblib.load('scaler.pkl') if 'scaler.pkl' in joblib.os.listdir() else StandardScaler()

# --- Mainloop ---
root.protocol("WM_DELETE_WINDOW", stop_reading)
root.mainloop()
