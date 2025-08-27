#!/usr/bin/env python3
"""
Test script to verify GUI performance improvements
"""

import os
import sys
import threading
import time

from PyQt5.QtCore import QThread, QTimer, pyqtSignal
from PyQt5.QtWidgets import (
    QApplication,
    QLabel,
    QMainWindow,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


class PerformanceTest(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("GUI Performance Test")
        self.setGeometry(100, 100, 400, 300)

        # Create central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout()

        # Status label
        self.status_label = QLabel("Ready")
        layout.addWidget(self.status_label)

        # Test button
        self.test_button = QPushButton("Start Performance Test")
        self.test_button.clicked.connect(self.start_test)
        layout.addWidget(self.test_button)

        # Results label
        self.results_label = QLabel("")
        layout.addWidget(self.results_label)

        central_widget.setLayout(layout)

        # Performance counters
        self.frame_count = 0
        self.start_time = None

    def start_test(self):
        """Start a simple performance test"""
        self.frame_count = 0
        self.start_time = time.time()
        self.status_label.setText("Running performance test...")
        self.test_button.setEnabled(False)

        # Create a timer to simulate frame processing
        self.timer = QTimer()
        self.timer.timeout.connect(self.simulate_frame_processing)
        self.timer.start(100)  # 10 FPS

    def simulate_frame_processing(self):
        """Simulate frame processing to test performance"""
        self.frame_count += 1

        # Simulate some processing time
        time.sleep(0.01)  # 10ms processing time

        # Update status every 10 frames
        if self.frame_count % 10 == 0:
            elapsed_time = time.time() - self.start_time
            fps = self.frame_count / elapsed_time if elapsed_time > 0 else 0
            self.status_label.setText(
                f"Processed {self.frame_count} frames, FPS: {fps:.1f}"
            )

        # Stop after 100 frames
        if self.frame_count >= 100:
            self.timer.stop()
            elapsed_time = time.time() - self.start_time
            avg_fps = self.frame_count / elapsed_time if elapsed_time > 0 else 0
            self.results_label.setText(
                f"Test completed!\nTotal frames: {self.frame_count}\nAverage FPS: {avg_fps:.1f}\nElapsed time: {elapsed_time:.2f}s"
            )
            self.status_label.setText("Test completed")
            self.test_button.setEnabled(True)


def main():
    app = QApplication(sys.argv)
    window = PerformanceTest()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
