import sys
import nidaqmx
from nidaqmx.constants import AcquisitionType, TerminalConfiguration, Edge
from nidaqmx.stream_readers import AnalogMultiChannelReader
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QLabel, QPushButton, QCheckBox, QDoubleSpinBox, QComboBox,
                             QMessageBox, QGroupBox, QGridLayout)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QMutex
from PyQt5.QtGui import QIcon
import pyqtgraph as pg
import scipy.signal as signal
from collections import deque
import time

class AcquisitionThread(QThread):
    data_ready = pyqtSignal(np.ndarray, np.ndarray, list, list)
    error_occurred = pyqtSignal(str)
    sampling_rate_updated = pyqtSignal(float)

    def __init__(self, num_channels, parent=None):
        super().__init__(parent)
        self.num_channels = num_channels
        self.sampling_rate = 100000
        self.buffer_size = 100000
        self.running = False
        self.task = None
        self.reader = None
        self.trigger_level = 0.0
        self.trigger_channel = 0
        self.trigger_enabled = True
        self.trigger_edge = Edge.RISING
        self.selected_channels = set(range(num_channels))
        self.v_per_div = [1.0] * num_channels
        self.min_sampling_rate = 1000
        self.max_aggregate_rate = 2000000
        self.max_per_channel_rate = self.max_aggregate_rate // max(1, len(self.selected_channels))
        self.mutex = QMutex()
        self.data_buffer = deque(maxlen=5)
        self.trigger_timeout = 1.0
        self.trigger_position = 0.5
        self.restart_needed = False
        self.new_sampling_rate = None

    def update_trigger(self, level, channel, enabled=True, edge=Edge.RISING):
        self.mutex.lock()
        try:
            self.trigger_level = level
            self.trigger_channel = channel
            self.trigger_enabled = enabled
            self.trigger_edge = edge
        finally:
            self.mutex.unlock()

    def update_trigger_position(self, position):
        self.mutex.lock()
        try:
            self.trigger_position = max(0.1, min(0.9, position))
        finally:
            self.mutex.unlock()

    def update_v_per_div(self, channel, value):
        self.mutex.lock()
        try:
            if 0 <= channel < self.num_channels:
                self.v_per_div[channel] = max(0.01, value)
        finally:
            self.mutex.unlock()

    def update_selected_channels(self, channels):
        self.mutex.lock()
        try:
            self.selected_channels = {ch for ch in channels if 0 <= ch < self.num_channels}
            self.max_per_channel_rate = self.max_aggregate_rate // max(1, len(self.selected_channels))
            if self.sampling_rate > self.max_per_channel_rate:
                self.update_sampling_rate(self.max_per_channel_rate)
        finally:
            self.mutex.unlock()

    def update_sampling_rate(self, new_rate):
        self.mutex.lock()
        try:
            active_channels = len(self.selected_channels)
            max_safe_rate = min(self.max_aggregate_rate // max(1, active_channels), 500000)
            new_rate = max(self.min_sampling_rate, min(int(new_rate), max_safe_rate))
            if new_rate != self.sampling_rate:
                self.new_sampling_rate = new_rate
                self.restart_needed = True
                self.sampling_rate_updated.emit(new_rate / 1000)
        finally:
            self.mutex.unlock()

    def restart_task(self):
        self.mutex.lock()
        try:
            if self.task is not None:
                try:
                    self.task.stop()
                    self.task.close()
                except Exception as e:
                    self.error_occurred.emit(f"Error stopping task: {str(e)}")
                finally:
                    self.task = None
                    self.reader = None
            self.sampling_rate = self.new_sampling_rate
            self.buffer_size = max(10000, int(self.sampling_rate * 0.2))
            self.create_task()
        finally:
            self.mutex.unlock()

    def create_task(self):
        try:
            self.task = nidaqmx.Task()
            for i in range(self.num_channels):
                self.task.ai_channels.add_ai_voltage_chan(
                    f"Dev1/ai{i}",
                    terminal_config=TerminalConfiguration.RSE,
                    min_val=-10.0,
                    max_val=10.0
                )
            self.task.timing.cfg_samp_clk_timing(
                self.sampling_rate,
                sample_mode=AcquisitionType.CONTINUOUS,
                samps_per_chan=self.buffer_size
            )
            self.task.in_stream.input_buf_size = self.buffer_size * 2
            self.reader = AnalogMultiChannelReader(self.task.in_stream)
            self.task.start()
        except Exception as e:
            self.error_occurred.emit(f"Task creation error: {str(e)}")
            if self.task is not None:
                try:
                    self.task.close()
                except:
                    pass
                self.task = None
            self.reader = None

    def calculate_frequency(self, data):
        if len(data) < 10:
            return 0.0
        try:
            data = data - np.mean(data)
            window = signal.windows.hann(len(data))
            windowed_data = data * window
            n_fft = max(16384, len(data) * 2)
            fft_data = np.fft.rfft(windowed_data, n=n_fft)
            fft_freq = np.fft.rfftfreq(n_fft, 1/self.sampling_rate)
            spectrum = np.abs(fft_data)
            peak_idx = np.argmax(spectrum[1:]) + 1
            freq = fft_freq[peak_idx]
            return max(freq, 0.01) if freq < self.sampling_rate/2 else 0.0
        except:
            return 0.0

    def find_trigger_point(self, data):
        if self.trigger_edge == Edge.RISING:
            crossings = np.where(np.diff(np.sign(data - self.trigger_level)) > 0)[0]
        else:
            crossings = np.where(np.diff(np.sign(data - self.trigger_level)) < 0)[0]
        return crossings[0] if crossings.size > 0 else None

    def run(self):
        try:
            self.create_task()
            while self.running:
                if self.restart_needed:
                    self.restart_task()
                    self.restart_needed = False
                    self.new_sampling_rate = None
                    continue

                if self.task is None:
                    self.create_task()
                    if self.task is None:
                        time.sleep(0.1)
                        continue

                self.mutex.lock()
                try:
                    data = np.zeros((self.num_channels, self.buffer_size), dtype=np.float64)
                    samples_read = self.reader.read_many_sample(
                        data,
                        number_of_samples_per_channel=self.buffer_size,
                        timeout=2.0
                    ) if self.reader else 0
                except Exception as e:
                    self.error_occurred.emit(f"Read error: {str(e)}")
                    time.sleep(0.1)
                    continue
                finally:
                    self.mutex.unlock()

                if samples_read <= 0:
                    continue

                self.data_buffer.append((time.time(), data[:, :samples_read]))

                if self.trigger_enabled:
                    trigger_data = data[self.trigger_channel, :samples_read]
                    trigger_idx = self.find_trigger_point(trigger_data)
                    
                    if trigger_idx is not None:
                        pre_trigger = int(self.buffer_size * self.trigger_position)
                        post_trigger = self.buffer_size - pre_trigger
                        start_idx = max(0, trigger_idx - pre_trigger)
                        end_idx = min(samples_read, trigger_idx + post_trigger)
                        if start_idx < end_idx:
                            data = data[:, start_idx:end_idx]
                            samples_read = end_idx - start_idx
                    else:
                        continue

                time_axis = np.arange(samples_read) / self.sampling_rate
                frequencies = [0.0] * self.num_channels
                periods = [0.0] * self.num_channels

                for ch in self.selected_channels:
                    freq = self.calculate_frequency(data[ch, :samples_read])
                    if 0 < freq < self.sampling_rate / 2:
                        frequencies[ch] = freq
                        periods[ch] = 1000 / freq if freq > 0 else 0
                    else:
                        frequencies[ch] = 0.0
                        periods[ch] = 0.0

                if data.size > 0 and time_axis.size == data.shape[1]:
                    self.data_ready.emit(time_axis, data, frequencies, periods)

        except Exception as e:
            self.error_occurred.emit(f"Thread error: {str(e)}")
        finally:
            self.mutex.lock()
            try:
                if self.task is not None:
                    self.task.stop()
                    self.task.close()
            except:
                pass
            self.task = None
            self.reader = None
            self.mutex.unlock()

    def stop(self):
        self.running = False
        self.wait()

class OscilloscopeGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Analog Read Oscilloscope")
        self.setGeometry(100, 100, 1600, 900)
        self.num_channels = 4
        self.channel_colors = ['#FFFF00', '#00FFFF', '#FF0000', '#00FF00']  # Yellow, Cyan, Red, Green
        self.divisions = 10
        self.timebase = 0.001
        self.trigger_position = 0.5

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QHBoxLayout(self.central_widget)

        # Create plot widget
        self.plot_widget = pg.PlotWidget(background='k')
        self.plot_widget.showGrid(x=True, y=True, alpha=1)  # Grey grid lines
        self.plot_widget.setMinimumSize(1200, 700)
        self.plot_widget.getViewBox().setMouseEnabled(x=True, y=True)
        self.plot_widget.setMenuEnabled(False)
        # Remove axis labels and values, keep only lines
        self.plot_widget.getAxis('left').setStyle(showValues=False, tickLength=0)
        self.plot_widget.getAxis('bottom').setStyle(showValues=False, tickLength=0)
        self.plot_widget.setLabel('left', '')
        self.plot_widget.setLabel('bottom', '')
        self.main_layout.addWidget(self.plot_widget, 7)

        # Add reference axes at the middle to create 4 quadrants
        self.x_axis = pg.InfiniteLine(pos=0, angle=0, movable=False, pen=pg.mkPen('w', width=2))
        self.y_axis = pg.InfiniteLine(pos=0, angle=90, movable=False, pen=pg.mkPen('w', width=2))
        self.plot_widget.addItem(self.x_axis)
        self.plot_widget.addItem(self.y_axis)

        # Right panel with controls
        self.right_panel = QWidget()
        self.right_layout = QVBoxLayout(self.right_panel)
        self.right_layout.setAlignment(Qt.AlignTop)
        self.right_layout.setSpacing(10)
        self.main_layout.addWidget(self.right_panel, 3)

        # Acquisition controls at the top
        self.create_acquisition_controls()

        # Channel controls
        self.create_channel_controls()
        
        # Measurement display section
        self.create_measurement_display()
        
        # Trigger controls
        self.create_trigger_controls()
        
        # Timebase controls
        self.create_timebase_controls()

        # Initialize acquisition thread
        self.acquisition_thread = AcquisitionThread(self.num_channels, self)
        self.acquisition_thread.data_ready.connect(self.update_gui)
        self.acquisition_thread.error_occurred.connect(self.handle_error)
        self.acquisition_thread.sampling_rate_updated.connect(self.update_sampling_rate_display)

        # Initialize plot curves
        self.plot_curves = []
        for i in range(self.num_channels):
            pen = pg.mkPen(color=self.channel_colors[i], width=4)
            curve = self.plot_widget.plot(pen=pen)
            self.plot_curves.append(curve)
        
        # Initialize trigger line (pink with arrow at right end)
        self.trigger_line = pg.InfiniteLine(angle=0, movable=True, pen=pg.mkPen('#C71585', width=2.75, style=Qt.SolidLine))
        self.plot_widget.addItem(self.trigger_line)
        # Add arrow at the right end of the trigger line
        self.trigger_arrow = pg.ArrowItem(
            pos=(0, 0),  # Will be updated in update_trigger
            angle=180,   # Pointing left (toward positive x)
            pen=pg.mkPen('#C71585', width=2),
            brush=pg.mkBrush('#C71585')
        )
        self.plot_widget.addItem(self.trigger_arrow)
        
        # Set initial ranges
        self.update_plot_ranges()
        
        # Apply styles (light mode with larger fonts)
        self.setStyleSheet("""
            QMainWindow { background-color: #f0f0f0; }
            QGroupBox { 
                border: 1px solid #ccc; 
                border-radius: 5px; 
                margin-top: 10px; 
                padding-top: 15px;
                color: black;
                font-size: 18px;
            }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; }
            QPushButton { 
                background-color: #FF0000; 
                color: white; 
                padding: 8px; 
                border-radius: 4px; 
                font-size: 18px; 
                min-width: 100px;
            }
            QPushButton:disabled { background-color: #ccc; color: #666; }
            QCheckBox, QLabel { 
                font-size: 18px; 
                color: black;
            }
            QDoubleSpinBox, QComboBox { 
                background-color: white; 
                color: black; 
                border: 1px solid #ccc; 
                padding: 6px; 
                font-size: 18px; 
                min-width: 100px;
            }
            QComboBox QAbstractItemView { 
                background-color: white; 
                color: black;
                font-size: 18px;
            }
        """)

    def create_acquisition_controls(self):
        acq_group = QGroupBox("Acquisition")
        acq_layout = QHBoxLayout()
        
        self.start_btn = QPushButton("Run")
        self.start_btn.clicked.connect(self.start_acquisition)
        
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.clicked.connect(self.stop_acquisition)
        self.stop_btn.setEnabled(False)
        
        acq_layout.addWidget(self.start_btn)
        acq_layout.addWidget(self.stop_btn)
        acq_group.setLayout(acq_layout)
        self.right_layout.addWidget(acq_group)

    def create_channel_controls(self):
        channel_group = QGroupBox("Channels")
        channel_layout = QGridLayout()
        
        self.channel_checkboxes = []
        self.vdiv_spins = []
        self.channel_visibilities = [True] * self.num_channels
        channel_colors = ["Yellow", "Blue", "Red", "Green"]  # Colors for each channel
        for i in range(self.num_channels):
            checkbox = QCheckBox(f"Channel {i+1} ({channel_colors[i]})")
            checkbox.setChecked(True)
            checkbox.stateChanged.connect(lambda state, ch=i: self.toggle_channel(ch, state))
            self.channel_checkboxes.append(checkbox)
            channel_layout.addWidget(checkbox, i, 0)
            
            vdiv_label = QLabel("V/div:")
            vdiv_spin = QDoubleSpinBox()
            vdiv_spin.setRange(0.01, 10.0)
            vdiv_spin.setValue(1.0)
            vdiv_spin.setSingleStep(0.1)
            vdiv_spin.valueChanged.connect(lambda value, ch=i: self.update_vdiv(ch, value))
            channel_layout.addWidget(vdiv_label, i, 1)
            channel_layout.addWidget(vdiv_spin, i, 2)
            self.vdiv_spins.append(vdiv_spin)
        
        # Add Select All and Deselect All buttons
        select_all_btn = QPushButton("Select All")
        select_all_btn.clicked.connect(self.select_all_channels)
        channel_layout.addWidget(select_all_btn, self.num_channels, 0)

        deselect_all_btn = QPushButton("Deselect All")
        deselect_all_btn.clicked.connect(self.deselect_all_channels)
        channel_layout.addWidget(deselect_all_btn, self.num_channels, 1)

        channel_group.setLayout(channel_layout)
        self.right_layout.addWidget(channel_group)

    def create_measurement_display(self):
        measurement_group = QGroupBox("Measurements")
        measurement_layout = QGridLayout()
        
        self.freq_labels = []
        self.period_labels = []
        for i in range(self.num_channels):
            freq_label = QLabel(f"Ch {i+1} Freq: 0.00 Hz")
            period_label = QLabel(f"Ch {i+1} Period: 0.00 ms")
            measurement_layout.addWidget(freq_label, i, 0)
            measurement_layout.addWidget(period_label, i, 1)
            self.freq_labels.append(freq_label)
            self.period_labels.append(period_label)
        
        measurement_group.setLayout(measurement_layout)
        self.right_layout.addWidget(measurement_group)

    def create_trigger_controls(self):
        trigger_group = QGroupBox("Trigger")
        trigger_layout = QGridLayout()
        
        self.trigger_enable = QCheckBox("Enable")
        self.trigger_enable.setChecked(True)
        self.trigger_enable.stateChanged.connect(self.update_trigger)
        trigger_layout.addWidget(self.trigger_enable, 0, 0, 1, 2)
        
        trigger_source_label = QLabel("Source:")
        self.trigger_source = QComboBox()
        self.trigger_source.addItems([f"Channel {i+1}" for i in range(self.num_channels)])
        self.trigger_source.currentIndexChanged.connect(self.update_trigger)
        trigger_layout.addWidget(trigger_source_label, 1, 0)
        trigger_layout.addWidget(self.trigger_source, 1, 1)
        
        trigger_edge_label = QLabel("Edge:")
        self.trigger_edge = QComboBox()
        self.trigger_edge.addItems(["Rising", "Falling"])
        self.trigger_edge.currentIndexChanged.connect(self.update_trigger)
        trigger_layout.addWidget(trigger_edge_label, 2, 0)
        trigger_layout.addWidget(self.trigger_edge, 2, 1)
        
        trigger_level_label = QLabel("Level (V):")
        self.trigger_level = QDoubleSpinBox()
        self.trigger_level.setRange(-10.0, 10.0)
        self.trigger_level.setValue(0.0)
        self.trigger_level.setSingleStep(0.1)
        self.trigger_level.valueChanged.connect(self.update_trigger)
        trigger_layout.addWidget(trigger_level_label, 3, 0)
        trigger_layout.addWidget(self.trigger_level, 3, 1)
        
        trigger_pos_label = QLabel("Position:")
        self.trigger_pos = QDoubleSpinBox()
        self.trigger_pos.setRange(0.1, 0.9)
        self.trigger_pos.setValue(0.5)
        self.trigger_pos.setSingleStep(0.1)
        self.trigger_pos.valueChanged.connect(self.update_trigger_position)
        trigger_layout.addWidget(trigger_pos_label, 4, 0)
        trigger_layout.addWidget(self.trigger_pos, 4, 1)
        
        trigger_group.setLayout(trigger_layout)
        self.right_layout.addWidget(trigger_group)

    def create_timebase_controls(self):
        timebase_group = QGroupBox("Timebase")
        timebase_layout = QVBoxLayout()
        
        time_div_label = QLabel("Time/div:")
        self.time_div = QComboBox()
        self.time_div.addItems(["1μs", "2μs", "5μs", "10μs", "20μs", "50μs", 
                              "100μs", "200μs", "500μs", "1ms", "2ms", "5ms", 
                              "10ms", "20ms", "50ms", "100ms", "200ms", "500ms"])
        self.time_div.setCurrentIndex(9)
        self.time_div.currentIndexChanged.connect(self.update_timebase)
        timebase_layout.addWidget(time_div_label)
        timebase_layout.addWidget(self.time_div)
        
        sample_rate_label = QLabel("Sampling Rate (kS/s):")
        self.sample_rate = QComboBox()
        self.sample_rate.addItems(["50", "100", "200", "250", "300", "400", "500"])
        self.sample_rate.setCurrentIndex(1)
        self.sample_rate.currentIndexChanged.connect(self.update_sampling_rate)
        timebase_layout.addWidget(sample_rate_label)
        timebase_layout.addWidget(self.sample_rate)
        
        timebase_group.setLayout(timebase_layout)
        self.right_layout.addWidget(timebase_group)

    def select_all_channels(self):
        for checkbox in self.channel_checkboxes:
            checkbox.setChecked(True)
        self.toggle_channel(0, Qt.Checked)  # Trigger update

    def deselect_all_channels(self):
        for checkbox in self.channel_checkboxes:
            checkbox.setChecked(False)
        self.toggle_channel(0, Qt.Unchecked)  # Trigger update

    def toggle_channel(self, channel, state):
        self.channel_visibilities[channel] = state == Qt.Checked
        self.acquisition_thread.update_selected_channels([i for i in range(self.num_channels) if self.channel_visibilities[i]])
        self.update_plot()

    def update_vdiv(self, channel, value):
        self.acquisition_thread.update_v_per_div(channel, value)
        self.update_plot_ranges()

    def update_trigger(self):
        level = self.trigger_level.value()
        channel = self.trigger_source.currentIndex()
        enabled = self.trigger_enable.isChecked()
        edge = Edge.RISING if self.trigger_edge.currentIndex() == 0 else Edge.FALLING
        self.acquisition_thread.update_trigger(level, channel, enabled, edge)
        self.trigger_line.setValue(level)
        self.trigger_line.setVisible(enabled)
        # Update arrow position to the right end of the trigger line
        x_range = self.timebase * self.divisions
        self.trigger_arrow.setPos(x_range, level)
        self.trigger_arrow.setVisible(enabled)

    def update_trigger_position(self):
        position = self.trigger_pos.value()
        self.acquisition_thread.update_trigger_position(position)
        self.update_plot_ranges()

    def update_timebase(self):
        time_div_text = self.time_div.currentText()
        if time_div_text.endswith("μs"):
            self.timebase = float(time_div_text[:-2]) * 1e-6
        elif time_div_text.endswith("ms"):
            self.timebase = float(time_div_text[:-2]) * 1e-3
        self.update_plot_ranges()

    def update_sampling_rate(self):
        rate = int(self.sample_rate.currentText()) * 1000
        self.acquisition_thread.update_sampling_rate(rate)

    def update_sampling_rate_display(self, rate_kss):
        rates = [50, 100, 200, 250, 300, 400, 500]
        closest_idx = min(range(len(rates)), key=lambda i: abs(rates[i] - rate_kss))
        self.sample_rate.blockSignals(True)
        self.sample_rate.setCurrentIndex(closest_idx)
        self.sample_rate.blockSignals(False)

    def update_plot_ranges(self):
        max_v_per_div = max(self.acquisition_thread.v_per_div[i] for i in range(self.num_channels) if self.channel_visibilities[i])
        y_range = max_v_per_div * self.divisions
        self.plot_widget.setYRange(-y_range, y_range)
        
        x_range = self.timebase * self.divisions
        self.plot_widget.setXRange(-x_range/2, x_range/2)  # Center x-axis
        
        # Center axes at (0,0)
        self.x_axis.setValue(0)
        self.y_axis.setValue(0)
        
        # Update trigger line and arrow position
        trigger_x = self.trigger_position * x_range - x_range/2  # Adjust for centered x-axis
        self.trigger_line.setValue(trigger_x)
        self.trigger_arrow.setPos(trigger_x + x_range/2, self.trigger_level.value())  # Arrow at right end

    def start_acquisition(self):
        if not self.acquisition_thread.isRunning():
            self.acquisition_thread.running = True
            self.start_btn.setEnabled(False)
            self.stop_btn.setEnabled(True)
            self.acquisition_thread.start()

    def stop_acquisition(self):
        self.acquisition_thread.running = False
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.acquisition_thread.wait()

    def update_gui(self, time_axis, data, frequencies, periods):
        if time_axis is None or data is None or len(time_axis) != data.shape[1]:
            return
            
        # Adjust time_axis to be centered around zero
        time_axis = time_axis - (self.timebase * self.divisions) / 2
        
        for ch in range(self.num_channels):
            if self.channel_visibilities[ch] and len(time_axis) == len(data[ch]):
                scaled_data = data[ch] / self.acquisition_thread.v_per_div[ch]
                self.plot_curves[ch].setData(time_axis, scaled_data)
                self.freq_labels[ch].setText(f"Ch {ch+1} Freq: {frequencies[ch]:.2f} Hz")
                self.period_labels[ch].setText(f"Ch {ch+1} Period: {periods[ch]:.2f} ms")
            else:
                self.plot_curves[ch].setData([], [])
                self.freq_labels[ch].setText(f"Ch {ch+1} Freq: 0.00 Hz")
                self.period_labels[ch].setText(f"Ch {ch+1} Period: 0.00 ms")

    def handle_error(self, error_msg):
        self.stop_acquisition()
        QMessageBox.critical(self, "Error", error_msg)

    def closeEvent(self, event):
        self.stop_acquisition()
        event.accept()

    def update_plot(self):
        for ch in range(self.num_channels):
            self.plot_curves[ch].setVisible(self.channel_visibilities[ch])
        self.update_plot_ranges()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = OscilloscopeGUI()
    window.setWindowIcon(QIcon('logo.png'))
    window.show()
    sys.exit(app.exec_())