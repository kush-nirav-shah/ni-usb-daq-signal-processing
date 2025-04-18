# NI USB DAQ Signal Interface Project - Easy Guide

Welcome to my internship project from the Physical Research Laboratory (PRL), Ahmedabad! This project uses a device called the NI USB-6363 DAQ to collect and control signals in real-time. Think of the DAQ as a smart connector that lets your computer talk to sensors or gadgets to measure or control things.

This guide is written for anyone, even if you’re new to this. I’ll explain what the project does and how you can try it out. The project mainly uses the NI USB-6363 DAQ, but I’ve also included some Arduino code from an earlier phase of my work.

## 🌟 What Does This Project Do?

This project uses the NI USB-6363 DAQ to:

- Read signals: Collect data from multiple sensors or devices at once (like measuring voltages or other signals).
- Send signals: Control devices by sending signals (like setting a voltage).

The Arduino code was used in the early stages to test ideas, but the real project runs entirely on the NI USB-6363 DAQ using Python programs.

## 🛠️ What’s Inside the Project?

The project has two parts:

### 1. Arduino Code (arduino_code/ folder)

These were used in the early phase and are included for reference:

- analog_read.ino: Reads sensor data (like voltages).
- digital_read.ino: Reads digital signals (like high/low states).
- digital_write.ino: Sends digital signals to control devices.

Note: You don’t need these Arduino files to run the main project. They’re here to show the starting point of my work.

### 2. NI USB DAQ Code (ni_usb_python_code/ folder)

These are the main Python programs that run the NI USB-6363 DAQ:

- digital_write.py: Sends digital signals to control connected devices.
- final_digital_read_multichannel.py: Reads multiple digital signals at once.
- final_analog_read_multichannel.py: Reads multiple analog signals (like voltages) at once.

## 🚀 How to Run the Project

This guide focuses on running the NI USB-6363 DAQ project, as that’s the main part of the work. The Arduino code is optional and not needed for the core project.

### What You’ll Need

- Hardware:
  - NI USB-6363 DAQ device (connected to your computer via USB).
  - Sensors or devices to test (e.g., something that outputs a voltage or accepts a control signal).

- Software:
  - A computer with Python installed (version 3.6 or higher).
  - NI-DAQmx driver (lets your computer communicate with the DAQ).

### Step 1: Set Up Your Computer

#### Install Python:

- Download and install Python from [python.org](https://www.python.org/)
- During installation, **check** “Add Python to PATH.”

#### Install NI-DAQmx Driver:

- Download the NI-DAQmx driver from the [National Instruments](https://www.ni.com/en-us/support/downloads/drivers/download.ni-daqmx.html)
- Follow the installation instructions.

#### Install Python Libraries:

- Download the `requirements.txt` file from this repository.

- Open your terminal (Command Prompt on Windows, Terminal on Mac/Linux) and navigate to the folder where requirements.txt is saved.

- Run this command:
  ```bash
pip install -r requirements.txt
  
This installs the Python libraries needed to control the DAQ.

### Step 2: Connect the Hardware

- Plug the NI USB-6363 DAQ into your computer’s USB port.
- Connect your sensors or devices to the DAQ’s input/output ports. For example, you might connect a sensor to an analog input port to measure a signal. Check the NI USB-6363 manual for details on which ports to use.

### Step 3: Run the DAQ Code

- Open a terminal or a Python editor (like VS Code, PyCharm, or IDLE).

- Navigate to the ni_usb_python_code/ folder in the terminal. For example:

- cd path/to/ni_usb_python_code

- Run a Python file by typing:
- python final_analog_read_multichannel.py

- 
(Replace final_analog_read_multichannel.py with the file you want to run, like digital_write.py or final_digital_read_multichannel.py.)

- The program will start reading or sending signals, depending on the code.

### Step 4: Test It!

- If you’re reading signals (e.g., with final_analog_read_multichannel.py), you should see data (like voltage values) printed on your screen.
- If you’re sending signals (e.g., with digital_write.py), check if the connected device responds as expected.
- Ensure your sensors or devices are properly connected to the DAQ’s ports.

## 🧰 Troubleshooting Tips

- DAQ not responding? Ensure the NI-DAQmx driver is installed and the DAQ is plugged into the USB port.
- Python errors? Confirm all libraries are installed (pip install -r requirements.txt) and you’re using Python 3.6 or higher.
- No data? Double-check your sensor/device connections to the DAQ and verify the correct port numbers in the Python code.

## 📚 Want to Learn More?

- Read the NI USB-6363 manual for details on connecting devices.
- Check the comments in the .py files for more info on how the code works.
- If you’re curious about the Arduino part, explore the .ino files, but they’re not needed for the main project.

Have fun experimenting with the NI USB-6363 DAQ! If you get stuck, feel free to reach out.
