import sys
import nidaqmx
from nidaqmx.constants import AcquisitionType, LineGrouping
from nidaqmx.stream_readers import DigitalMultiChannelReader
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QLabel, QPushButton, QGroupBox, QCheckBox, QGridLayout, QDoubleSpinBox)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont
import pyqtgraph as pg
import time

class CustomCheckBox(QCheckBox):
    doubleClicked = pyqtSignal()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def mouseDoubleClickEvent(self, event):
        self.doubleClicked.emit()
        event.accept()

class AcquisitionThread(QThread):
    data_ready = pyqtSignal(list, list, np.ndarray, np.ndarray)

    def __init__(self, sampling_rate=4000000, initial_buffer_size=8000000):
        super().__init__()
        self.sampling_rate = sampling_rate
        self.buffer_size = initial_buffer_size
        self.running = False
        self.task = None
        self.reader = None
        self.num_channels = 16
        self.num_cycles = 2 * (initial_buffer_size / sampling_rate)
        self.last_update_time = 0

    def update_num_cycles(self, num_cycles):
        self.num_cycles = num_cycles

    def create_task(self):
        self.task = nidaqmx.Task()
        self.task.di_channels.add_di_chan(
            "Dev1/port0/line0:15",
            line_grouping=LineGrouping.CHAN_PER_LINE
        )
        self.task.timing.cfg_samp_clk_timing(
            self.sampling_rate,
            sample_mode=AcquisitionType.CONTINUOUS,
            samps_per_chan=self.buffer_size
        )
        self.task.in_stream.input_buf_size = 8000000
        self.reader = DigitalMultiChannelReader(self.task.in_stream)
        self.task.start()
        print("Task created for all 16 channels")

    def run(self):
        try:
            self.create_task()
            last_frequencies = [0] * self.num_channels
            while self.running:
                max_period = 0
                for freq in last_frequencies:
                    if freq > 0:
                        period = 1.0 / freq
                        max_period = max(max_period, period)
                
                if max_period == 0:
                    self.buffer_size = int(self.sampling_rate * 0.5)
                else:
                    self.buffer_size = int(self.sampling_rate * max_period * max(self.num_cycles, 2))
                    if any(freq > 0 and freq % 5 == 0 for freq in last_frequencies):
                        self.buffer_size = int(self.buffer_size * 1.5)
                
                self.buffer_size = max(self.buffer_size, int(self.sampling_rate * 0.05))
                self.buffer_size = min(self.buffer_size, int(self.sampling_rate * 1.0))

                data = np.zeros((self.num_channels, self.buffer_size), dtype=np.uint32)
                try:
                    samples_read = self.reader.read_many_sample_port_uint32(
                        data,
                        number_of_samples_per_channel=self.buffer_size,
                        timeout=2.0
                    )
                except Exception as e:
                    print(f"Error reading data: {e}")
                    continue
                    
                if samples_read == 0:
                    continue

                time_axis = np.arange(samples_read) / self.sampling_rate
                digital_data = np.zeros((self.num_channels, samples_read), dtype=np.uint8)
                for ch in range(self.num_channels):
                    digital_data[ch] = (data[ch] >> ch) & 0x1
                
                frequencies = []
                time_periods_ms = []
                for ch in range(self.num_channels):
                    channel_data = digital_data[ch]
                    rising_edges = np.where((channel_data[:-1] == 0) & (channel_data[1:] == 1))[0]
                    if len(rising_edges) >= 2:
                        periods = np.diff(rising_edges) / self.sampling_rate
                        periods.sort()
                        median_period = np.median(periods)
                        valid_periods = periods[(periods < 2 * median_period) & (periods > 0.5 * median_period)]
                        if len(valid_periods) > 0:
                            avg_period = np.mean(valid_periods)
                            frequency = 1.0 / avg_period
                            time_period_ms = avg_period * 1000
                            last_frequencies[ch] = frequency
                        else:
                            frequency = 0
                            time_period_ms = 0
                            last_frequencies[ch] = 0
                    else:
                        frequency = 0
                        time_period_ms = 0
                        last_frequencies[ch] = 0
                    frequencies.append(frequency)
                    time_periods_ms.append(time_period_ms)

                current_time = time.time()
                if current_time - self.last_update_time >= 0.05:
                    self.data_ready.emit(frequencies, time_periods_ms, time_axis, digital_data)
                    self.last_update_time = current_time

        except Exception as e:
            print(f"Error in run: {e}")
        finally:
            if self.task is not None:
                try:
                    self.task.stop()
                    self.task.close()
                    print("Task closed successfully")
                except Exception as e:
                    print(f"Error closing task: {e}")
                self.task = None
                self.reader = None

    def stop(self):
        self.running = False
        self.wait()

class FrequencyGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Digital Oscilloscope - NI USB-6363 (16 Channels)")
        
        screen = QApplication.primaryScreen().availableGeometry()
        width = min(2560, screen.width())
        height = min(1440, screen.height())
        self.setGeometry(0, 0, width, height)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QHBoxLayout(self.central_widget)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        
        # Left side: Plots
        self.left_widget = QWidget()
        self.left_widget.setStyleSheet("background-color: #000000;")
        self.left_layout = QVBoxLayout(self.left_widget)
        self.left_layout.setSpacing(0)
        self.left_layout.setContentsMargins(0, 0, 0, 0)
        
        self.plot_widgets = [None] * 16
        self.plot_curves = [None] * 16
        self.plot_labels = [None] * 16
        self.visible_plots = []
        
        for i in range(16):
            self._create_plot_widget(i)
        
        self.time_axis_widget = pg.PlotWidget()
        self.time_axis_widget.setMaximumHeight(80)
        self.time_axis_widget.getAxis('left').setTicks([[(0, '0'), (1, '1')]])
        self.time_axis_widget.getAxis('bottom').setLabel('Time (s)')
        self.time_axis_curves = []
        self.left_layout.addWidget(self.time_axis_widget)
        
        # Right side: Controls
        self.right_widget = QWidget()
        self.right_layout = QVBoxLayout(self.right_widget)
        self.right_layout.setContentsMargins(10, 10, 10, 10)
        
        self.buttons_layout = QHBoxLayout()
        self.start_button = QPushButton("START")
        self.start_button.clicked.connect(self.start_acquisition)
        self.stop_button = QPushButton("STOP")
        self.stop_button.clicked.connect(self.stop_acquisition)
        self.stop_button.setEnabled(False)
        self.start_button.setFixedWidth(150)
        self.stop_button.setFixedWidth(150)
        self.buttons_layout.addWidget(self.start_button)
        self.buttons_layout.addWidget(self.stop_button)
        self.right_layout.addLayout(self.buttons_layout)
        self.right_layout.addSpacing(20)
        
        self.channels_group = QGroupBox("Line Selection")
        self.channels_layout = QVBoxLayout()
        self.checkbox_grid = QGridLayout()
        self.channel_checkboxes = []
        self.selected_channels = set()
        
        for i in range(16):
            checkbox = CustomCheckBox(f"Line{i}")
            checkbox.setStyleSheet("font-size: 24px;")
            checkbox.clicked.connect(lambda checked, ch=i: self.toggle_channel(ch, checked))
            checkbox.doubleClicked.connect(lambda ch=i: self.double_click_channel(ch))
            self.channel_checkboxes.append(checkbox)
            row = i % 8
            col = 0 if i < 8 else 1
            self.checkbox_grid.addWidget(checkbox, row, col)
        
        self.channels_layout.addLayout(self.checkbox_grid)
        self.channels_layout.addSpacing(10)
        
        self.select_all_button = QPushButton("Select All")
        self.select_all_button.clicked.connect(self.select_all_channels)
        self.select_all_button.setStyleSheet("font-size: 24px; padding: 5px;")
        self.deselect_all_button = QPushButton("Deselect All")
        self.deselect_all_button.clicked.connect(self.deselect_all_channels)
        self.deselect_all_button.setStyleSheet("font-size: 24px; padding: 5px;")
        
        self.select_buttons_layout = QHBoxLayout()
        self.select_buttons_layout.addWidget(self.select_all_button)
        self.select_buttons_layout.addWidget(self.deselect_all_button)
        self.channels_layout.addLayout(self.select_buttons_layout)
        
        self.channels_group.setLayout(self.channels_layout)
        self.right_layout.addWidget(self.channels_group)
        self.right_layout.addSpacing(20)
        
        self.rate_cycles_layout = QHBoxLayout()
        self.label_daq_rate = QLabel("Sampling Rate: -- MS/s")
        self.label_daq_rate.setStyleSheet("font-size: 28px; padding: 10px;")
        self.rate_cycles_layout.addWidget(self.label_daq_rate)
        
        self.cycles_layout = QHBoxLayout()
        self.label_cycles = QLabel("Number of Cycles:")
        self.label_cycles.setStyleSheet("font-size: 28px; padding: 10px;")
        self.cycles_spinbox = QDoubleSpinBox()
        self.cycles_spinbox.setRange(1, 100)
        self.cycles_spinbox.setValue(4)
        self.cycles_spinbox.setSingleStep(1)
        self.cycles_spinbox.setFixedWidth(100)
        self.cycles_spinbox.setStyleSheet("font-size: 24px; padding: 10px;")
        self.cycles_spinbox.valueChanged.connect(self.update_num_cycles)
        self.cycles_layout.addWidget(self.label_cycles)
        self.cycles_layout.addWidget(self.cycles_spinbox)
        self.rate_cycles_layout.addLayout(self.cycles_layout)
        
        self.right_layout.addLayout(self.rate_cycles_layout)
        self.right_layout.addSpacing(20)
        
        self.measure_group = QGroupBox("Measurements")
        self.measure_layout = QVBoxLayout()
        
        self.toggle_button = QPushButton("Switch to Time Period")
        self.toggle_button.clicked.connect(self.toggle_measure_display)
        self.toggle_button.setStyleSheet("font-size: 24px;")
        self.measure_layout.addWidget(self.toggle_button)
        
        self.measure_table = QGridLayout()
        self.measure_labels = {}
        self.show_frequency = True
        self.measure_layout.addLayout(self.measure_table)
        
        self.measure_group.setLayout(self.measure_layout)
        self.right_layout.addWidget(self.measure_group)
        
        self.right_layout.addStretch()
        
        self.main_layout.addWidget(self.left_widget, 7)
        self.main_layout.addWidget(self.right_widget, 3)
        
        font = QFont()
        font.setPointSize(16)
        self.label_cycles.setFont(font)
        self.setStyleSheet("""
            QMainWindow { background-color: #f0f0f0; }
            QGroupBox { 
                font-size: 28px; 
                font-weight: bold; 
                border: 1px solid #ccc; 
                border-radius: 5px; 
                margin-top: 20px; 
                padding-top: 20px; 
                background-color: #f0f0f0; 
            }
            QGroupBox::title { 
                subcontrol-origin: margin; 
                subcontrol-position: top center; 
                padding: 0 10px 20px 10px; 
            }
            QPushButton { 
                background-color: #4CAF50; 
                color: white; 
                padding: 14px; 
                border-radius: 10px; 
                font-size: 32px; 
            }
            QPushButton:disabled { background-color: #cccccc; }
            QPushButton:hover { background-color: #45a049; }
            QSpinBox, QDoubleSpinBox { 
                border: 1px solid #000000; 
                border-radius: 3px; 
                padding: 5px; 
                font-size: 24px; 
            }
            QCheckBox { font-size: 24px; }
        """)
        
        self.acquisition_thread = AcquisitionThread()
        self.acquisition_thread.data_ready.connect(self.update_gui)
        self.clear_all_plots()
        self.update_plot_layout()

    def get_channel_color(self, channel):
        colors = ['r', 'g', 'b', 'y', 'c', 'm', 'w', 'orange', 'purple', 'pink', 
                  'lime', 'teal', 'violet', 'brown', 'gray', 'indigo']
        return colors[channel % len(colors)]

    def _create_plot_widget(self, channel_num):
        plot_widget = pg.PlotWidget(background='k')
        plot_widget.setYRange(0, 1)
        plot_widget.getAxis('left').setTicks([[(0, '0'), (1, '1')]])
        plot_widget.getAxis('bottom').hide()
        
        label = pg.TextItem(f"Line{channel_num}", color='w', anchor=(0, 0))
        label.setPos(0, 0.8)
        plot_widget.addItem(label)
        
        self.plot_widgets[channel_num] = plot_widget
        self.plot_curves[channel_num] = plot_widget.plot(pen=pg.mkPen(color=self.get_channel_color(channel_num), width=2), stepMode="right")
        self.plot_labels[channel_num] = label

    def clear_all_plots(self):
        for curve in self.plot_curves:
            if curve is not None:
                curve.setData([], [])
        for curve in self.time_axis_curves:
            if curve is not None:
                curve.setData([], [])

    def update_plot_layout(self):
        for plot in self.visible_plots:
            self.left_layout.removeWidget(plot)
            plot.hide()
        self.visible_plots.clear()
        
        for i in range(self.left_layout.count() - 1, -1, -1):
            item = self.left_layout.itemAt(i)
            if item.widget() != self.time_axis_widget and item.spacerItem():
                self.left_layout.removeItem(item)
        
        if not self.selected_channels:
            self.left_layout.insertStretch(0, 1)
            return
        
        total_height = self.left_widget.height() - 80
        num_selected = len(self.selected_channels)
        plot_height = min(200, total_height // max(1, num_selected))
        total_plots_height = plot_height * num_selected
        remaining_space = total_height - total_plots_height
        
        top_spacing = remaining_space // 2
        self.left_layout.insertStretch(0, top_spacing)
        
        for i, channel in enumerate(sorted(self.selected_channels)):
            plot_widget = self.plot_widgets[channel]
            plot_widget.setMaximumHeight(plot_height)
            plot_widget.show()
            self.left_layout.insertWidget(i + 1, plot_widget)
            self.visible_plots.append(plot_widget)
        
        bottom_spacing = remaining_space - top_spacing
        self.left_layout.insertStretch(len(self.visible_plots) + 1, bottom_spacing)

    def toggle_channel(self, channel_index, checked):
        if checked:
            self.selected_channels.add(channel_index)
        else:
            self.selected_channels.discard(channel_index)
        self.update_plot_layout()
        self.update_measure_table()
        if hasattr(self, 'last_digital_data') and hasattr(self, 'last_time_axis'):
            self.update_time_axis_plot()

    def double_click_channel(self, channel_index):
        self.channel_checkboxes[channel_index].setChecked(False)
        self.selected_channels.discard(channel_index)
        self.update_plot_layout()
        self.update_measure_table()
        if hasattr(self, 'last_digital_data') and hasattr(self, 'last_time_axis'):
            self.update_time_axis_plot()

    def select_all_channels(self):
        for i in range(16):
            self.channel_checkboxes[i].setChecked(True)
            self.selected_channels.add(i)
        self.update_plot_layout()
        self.update_measure_table()
        if hasattr(self, 'last_digital_data') and hasattr(self, 'last_time_axis'):
            self.update_time_axis_plot()

    def deselect_all_channels(self):
        for i in range(16):
            self.channel_checkboxes[i].setChecked(False)
        self.selected_channels.clear()
        self.update_plot_layout()
        self.update_measure_table()
        if hasattr(self, 'last_digital_data') and hasattr(self, 'last_time_axis'):
            self.update_time_axis_plot()

    def start_acquisition(self):
        self.acquisition_thread.running = True
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.acquisition_thread.start()

    def stop_acquisition(self):
        self.acquisition_thread.stop()
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)

    def update_num_cycles(self):
        num_cycles = self.cycles_spinbox.value()
        self.acquisition_thread.update_num_cycles(num_cycles)

    def update_time_axis_plot(self):
        num_cycles = self.cycles_spinbox.value()

        if self.selected_channels and hasattr(self, 'last_time_periods_ms'):
            periods = [self.last_time_periods_ms[ch] for ch in self.selected_channels if self.last_time_periods_ms[ch] > 0]
            period_ms = min(periods) if periods else 0
        else:
            period_ms = 0

        if period_ms > 0:
            time_limit = (period_ms / 1000) * num_cycles
        else:
            time_limit = 0.1 * num_cycles

        idx_limit = min(int(time_limit * self.acquisition_thread.sampling_rate), self.last_digital_data.shape[1])
        max_points = 10000
        if idx_limit > max_points:
            step = idx_limit // max_points
            plot_time_axis = self.last_time_axis[::step]
            plot_digital_data = self.last_digital_data[:, ::step]
        else:
            plot_time_axis = self.last_time_axis[:idx_limit]
            plot_digital_data = self.last_digital_data[:, :idx_limit]

        for curve in self.time_axis_curves:
            self.time_axis_widget.removeItem(curve)
        self.time_axis_curves.clear()

        for i in self.selected_channels:
            curve = self.time_axis_widget.plot(pen=pg.mkPen(color=self.get_channel_color(i), width=2), stepMode="right")
            curve.setData(plot_time_axis, plot_digital_data[i])
            self.time_axis_curves.append(curve)

        self.time_axis_widget.setXRange(0, time_limit)

    def toggle_measure_display(self):
        self.show_frequency = not self.show_frequency
        self.toggle_button.setText("Switch to " + ("Time Period" if self.show_frequency else "Frequency"))
        self.update_measure_table()

    def update_measure_table(self):
        for i in reversed(range(self.measure_table.count())):
            self.measure_table.itemAt(i).widget().setParent(None)
        
        if not self.selected_channels:
            return
        
        for i, ch in enumerate(sorted(self.selected_channels)):
            line_label = QLabel(f"Line{ch}")
            line_label.setStyleSheet("font-size: 24px;")
            if ch not in self.measure_labels:
                freq_label = QLabel("-- Hz")
                period_label = QLabel("-- ms")
                freq_label.setStyleSheet("font-size: 24px;")
                period_label.setStyleSheet("font-size: 24px;")
                self.measure_labels[ch] = (freq_label, period_label)
            
            freq_label, period_label = self.measure_labels[ch]
            row = i % 8
            col = 0 if ch < 8 else 1
            self.measure_table.addWidget(line_label, row, col * 2)
            self.measure_table.addWidget(freq_label if self.show_frequency else period_label, row, col * 2 + 1)
        
        if hasattr(self, 'last_frequencies'):
            self.update_gui_measurements()

    def update_gui_measurements(self):
        for ch in self.selected_channels:
            freq = self.last_frequencies[ch]
            period = self.last_time_periods_ms[ch]
            freq_label, period_label = self.measure_labels[ch]
            if freq == 0:
                freq_label.setText("-- Hz")
                period_label.setText("-- ms")
            elif freq < 1000:
                freq_label.setText(f"{freq:.3f} Hz")
                period_label.setText(f"{period:.6f} ms")
            else:
                freq_label.setText(f"{freq/1000:.3f} kHz")
                period_label.setText(f"{period:.6f} ms")

    def update_gui(self, frequencies, time_periods_ms, time_axis, digital_data):
        self.last_digital_data = digital_data
        self.last_time_axis = time_axis
        self.last_time_periods_ms = time_periods_ms
        self.last_frequencies = frequencies
        
        daq_rate = self.acquisition_thread.sampling_rate / 1e6
        self.label_daq_rate.setText(f"Sampling Rate: {daq_rate:.3f} MS/s")
        
        self.update_gui_measurements()
        
        num_cycles = self.cycles_spinbox.value()

        periods = [time_periods_ms[ch] for ch in self.selected_channels if time_periods_ms[ch] > 0]
        period = min(periods) if periods else 0
        if period > 0:
            time_limit = (period / 1000) * num_cycles
        else:
            time_limit = 0.1 * num_cycles

        idx_limit = min(int(time_limit * self.acquisition_thread.sampling_rate), digital_data.shape[1])
        max_points = 10000
        if idx_limit > max_points:
            step = idx_limit // max_points
            plot_time_axis = time_axis[::step]
            plot_digital_data = digital_data[:, ::step]
        else:
            plot_time_axis = time_axis[:idx_limit]
            plot_digital_data = self.last_digital_data[:, :idx_limit]

        for i in range(16):
            if self.plot_curves[i] is not None:
                if i in self.selected_channels:
                    self.plot_curves[i].setData(plot_time_axis, plot_digital_data[i])
                    self.plot_widgets[i].setXRange(0, time_limit)
                else:
                    self.plot_curves[i].setData([], [])
        
        self.update_time_axis_plot()

    def closeEvent(self, event):
        self.acquisition_thread.stop()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = FrequencyGUI()
    window.show()
    sys.exit(app.exec_())




