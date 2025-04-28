import csv
import time
from datetime import datetime
import os
from pathlib import Path
import psutil

class SystemMetricsLogger:
    def __init__(self, log_dir="metrics/runs"):
        # create log directory if it doesn't exist
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # create a new log file with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = self.log_dir / f"system_metrics_{timestamp}.csv"
        
        # write header
        with open(self.log_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'timestamp',
                'cpu_percent',
                'memory_percent',
                'cpu_temp'
            ])
    
    def log_metrics(self):
        """Log current system metrics"""
        timestamp = time.time()
        cpu_percent = psutil.cpu_percent()
        memory_percent = psutil.virtual_memory().percent
        
        # get CPU temperature if available (works on mac)
        try:
            cpu_temp = psutil.sensors_temperatures()['coretemp'][0].current
        except (KeyError, AttributeError):
            cpu_temp = 0  # not available
        
        with open(self.log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                timestamp,
                cpu_percent,
                memory_percent,
                cpu_temp
            ])
        
        return {
            'cpu_percent': cpu_percent,
            'memory_percent': memory_percent,
            'cpu_temp': cpu_temp
        }
    
    def close(self):
        """Clean up resources"""
        pass  # nothing to clean up on mac 