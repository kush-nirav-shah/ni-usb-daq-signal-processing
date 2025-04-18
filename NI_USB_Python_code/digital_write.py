import sys
import numpy as np
import nidaqmx
from nidaqmx.constants import AcquisitionType
from PyQt5 import QtWidgets, QtCore, QtGui
from pyqtgraph import PlotWidget
import pyqtgraph as pg
from collections import deque

class SquareWaveGUI(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Real-Time Square Wave Oscilloscope")
        self.setGeometry(100, 100, 800, 600)

        # Central widget setup
        self.central_widget = QtWidgets.QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QtWidgets.QVBoxLayout(self.central_widget)

        # Input controls
        self.freq_input = QtWidgets.QLineEdit("1000")
        self.freq_input.setPlaceholderText("Enter frequency (Hz)")
        self.cycles_input = QtWidgets.QLineEdit("5")
        self.cycles_input.setPlaceholderText("Cycles to display")

        # Buttons
        self.start_button = QtWidgets.QPushButton("Start")
        self.stop_button = QtWidgets.QPushButton("Stop")
        self.stop_button.setEnabled(False)

        # Plot widget with oscilloscope-like appearance
        self.plot_widget = PlotWidget()
        self.plot_widget.setLabel('left', 'Amplitude')
        self.plot_widget.setLabel('bottom', 'Time (μs)')
        self.plot_widget.setYRange(-0.2, 1.2)
        self.plot_widget.setXRange(0, 1000)  # Initial 1000μs range
        self.plot_widget.setBackground('k')
        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)
        
        # Set axis colors for better visibility
        self.plot_widget.getAxis('left').setPen(pg.mkPen('w'))
        self.plot_widget.getAxis('bottom').setPen(pg.mkPen('w'))

        # Add widgets to layout
        self.layout.addWidget(self.freq_input)
        self.layout.addWidget(self.cycles_input)
        self.layout.addWidget(self.start_button)
        self.layout.addWidget(self.stop_button)
        self.layout.addWidget(self.plot_widget)

        # Connect signals
        self.start_button.clicked.connect(self.start_wave)
        self.stop_button.clicked.connect(self.stop_wave)

        # DAQ tasks
        self.output_task = None
        self.input_task = None
        
        # Timer for animation
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_display)
        
        # Data handling
        self.data_buffer = deque(maxlen=1000000)  # Circular buffer
        self.time_buffer = deque(maxlen=1000000)
        self.display_data = deque(maxlen=5000)    # Display buffer
        self.display_times = deque(maxlen=5000)
        
        # Plot curve
        self.curve = self.plot_widget.plot(pen=pg.mkPen('g', width=2))
        
        # Configuration
        self.max_sample_rate = 2000000  # 2 MS/s
        self.min_samples_per_cycle = 4  # Minimum for decent waveform
        self.max_samples_per_cycle = 10000  # Cap for low frequencies
        self.sample_rate = 1000000
        self.time_per_sample = 1.0  # μs at 1MS/s
        self.current_time = 0
        self.last_display_time = 0
        self.display_refresh_rate = 30  # Hz

    def start_wave(self):
        try:
            freq = float(self.freq_input.text())
            cycles_to_display = int(self.cycles_input.text())
            
            if not (1 <= freq <= 500000):
                raise ValueError("Frequency must be between 1 Hz and 500 kHz")
            if cycles_to_display <= 0:
                raise ValueError("Number of cycles must be positive")

            # Calculate parameters
            samples_per_cycle = min(self.max_samples_per_cycle, max(self.min_samples_per_cycle, int(self.max_sample_rate / freq)))
            self.sample_rate = min(freq * samples_per_cycle, self.max_sample_rate)
            self.time_per_sample = (1.0 / self.sample_rate) * 1000000  # μs
            self.period_us = (1.0 / freq) * 1000000
            self.display_window = cycles_to_display * self.period_us
            
            # Debug print to verify parameters
            print(f"Frequency: {freq} Hz, Sample Rate: {self.sample_rate} S/s, Samples per Cycle: {samples_per_cycle}")

            # Clear buffers
            self.data_buffer.clear()
            self.time_buffer.clear()
            self.display_data.clear()
            self.display_times.clear()
            self.current_time = 0
            self.last_display_time = 0
            
            # Stop any existing tasks
            self.stop_wave()
                
            # Configure output task
            self.output_task = nidaqmx.Task()
            self.output_task.do_channels.add_do_chan("Dev1/port0/line17")
            
            # Generate square wave
            half_cycle_samples = max(2, samples_per_cycle // 2)
            single_cycle = np.concatenate([
                np.ones(half_cycle_samples), 
                np.zeros(samples_per_cycle - half_cycle_samples)
            ])
            wave_data = np.tile(single_cycle, 100).astype(bool)  # Buffer for 100 cycles
            
            # Configure output timing
            self.output_task.timing.cfg_samp_clk_timing(
                rate=self.sample_rate,
                sample_mode=AcquisitionType.CONTINUOUS,
                samps_per_chan=len(wave_data)
            )
            
            # Configure input task
            self.input_task = nidaqmx.Task()
            self.input_task.di_channels.add_di_chan("Dev1/port0/line0")
            self.input_task.timing.cfg_samp_clk_timing(
                rate=self.sample_rate,
                sample_mode=AcquisitionType.CONTINUOUS,
                samps_per_chan=100000  # Large buffer
            )
            self.input_task.in_stream.input_buf_size = 1000000  # 1M samples
            
            # Start tasks
            self.output_task.write(wave_data, auto_start=True)
            self.input_task.start()
            
            # Configure plot
            self.plot_widget.setXRange(0, self.display_window)
            
            # Start animation timer
            self.start_button.setEnabled(False)
            self.stop_button.setEnabled(True)
            self.timer.start(int(1000 / self.display_refresh_rate))  # 30Hz refresh

        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", str(e))
            self.stop_wave()

    def update_display(self):
        try:
            # Read all available samples
            available_samples = self.input_task.in_stream.avail_samp_per_chan
            if available_samples == 0:
                return
                
            # Read data (limit to prevent UI freeze)
            samples_to_read = min(available_samples, 10000)
            data = self.input_task.read(number_of_samples_per_channel=samples_to_read)
            
            if len(data) > 0:
                # Generate time points and add to buffer
                new_times = np.arange(
                    self.current_time,
                    self.current_time + len(data) * self.time_per_sample,
                    self.time_per_sample
                )
                self.current_time += len(data) * self.time_per_sample
                
                self.data_buffer.extend(data)
                self.time_buffer.extend(new_times)
                
                # Update display with the most recent data within the time window
                display_end = self.current_time
                display_start = max(0, display_end - self.display_window)
                
                # Convert to numpy arrays for efficient slicing
                time_array = np.array(self.time_buffer)
                data_array = np.array(self.data_buffer)
                
                # Ensure arrays are the same length
                min_len = min(len(time_array), len(data_array))
                time_array = time_array[:min_len]
                data_array = data_array[:min_len]
                
                # Get indices of data within display window
                mask = (time_array >= display_start) & (time_array <= display_end)
                display_times = time_array[mask] - display_start  # Normalize to 0
                display_data = data_array[mask]
                
                # Downsample if needed for smooth display
                if len(display_times) > 2000:
                    step = len(display_times) // 2000
                    display_times = display_times[::step]
                    display_data = display_data[::step]
                
                # Update plot with step mode for sharp transitions
                if len(display_times) > 0:
                    self.curve.setData(display_times, display_data, stepMode="left")  # Use stepMode for square wave
                    
                    # Auto-scroll effect
                    self.plot_widget.setXRange(0, self.display_window, padding=0)

        except nidaqmx.errors.DaqReadError:
            pass  # Skip if no data available
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", f"Display error: {str(e)}")
            self.stop_wave()

    def stop_wave(self):
        self.timer.stop()
        if self.output_task:
            try:
                self.output_task.stop()
                self.output_task.close()
            except:
                pass
            self.output_task = None
        if self.input_task:
            try:
                self.input_task.stop()
                self.input_task.close()
            except:
                pass
            self.input_task = None
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)

    def closeEvent(self, event):
        self.stop_wave()
        event.accept()

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    
    # Set dark theme for better oscilloscope look
    app.setStyle('Fusion')
    dark_palette = QtGui.QPalette()
    dark_palette.setColor(QtGui.QPalette.Window, QtGui.QColor(30, 30, 30))
    dark_palette.setColor(QtGui.QPalette.WindowText, QtCore.Qt.white)
    app.setPalette(dark_palette)
    
    window = SquareWaveGUI()
    window.show()
    sys.exit(app.exec_())


