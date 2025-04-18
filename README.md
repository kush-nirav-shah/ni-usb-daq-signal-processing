# NI USB DAQ Signal Interface Project

This repository contains my internship project files developed at Physical Research Laboratory (PRL), Ahmedabad. The project focuses on real-time signal acquisition and control using NI USB-6363 DAQ and Arduino Uno.

## 🔌 Project Components

### 1. Arduino Codes (`arduino_code/`
- **analog_read.ino** – Reads analog values from sensors.
- **digital_read.ino** – Reads digital signal input.
- **digital_write.ino** – Sends digital signals as output.

### 2. NI USB Python Codes (`ni_usb_python_code/`)
- **digital_write.py** – Sends digital signal via NI USB-6363.
- **final_digital_read_multichannel.py** – Reads multiple digital inputs.
- **final_analog_read_multichannel.py** – Reads multiple analog inputs.

## 📦 Dependencies

To install the necessary dependencies, use the following `requirements.txt` file:

To install them, run the following command in your terminal:
   pip install -r requirements.txt
