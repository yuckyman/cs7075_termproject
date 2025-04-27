import csv
import time
from datetime import datetime
import os
from pathlib import Path

class GestureLatencyLogger:
    def __init__(self, log_dir="metrics/runs"):
        # create log directory if it doesn't exist
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # create a new log file with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = self.log_dir / f"gesture_latency_{timestamp}.csv"
        
        # write header
        with open(self.log_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'gesture_timestamp',
                'control_timestamp',
                'latency_ms',
                'gesture_type',
                'control_params'
            ])
    
    def log_gesture(self, gesture_type, control_params):
        """Log when a gesture is recognized"""
        return {
            'gesture_timestamp': time.time(),
            'gesture_type': gesture_type,
            'control_params': str(control_params)
        }
    
    def log_control(self, gesture_data):
        """Log when control is executed and calculate latency"""
        control_timestamp = time.time()
        latency_ms = (control_timestamp - gesture_data['gesture_timestamp']) * 1000
        
        with open(self.log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                gesture_data['gesture_timestamp'],
                control_timestamp,
                latency_ms,
                gesture_data['gesture_type'],
                gesture_data['control_params']
            ])
        
        return latency_ms 