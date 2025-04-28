import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import glob
import os
import numpy as np
from scipy.signal import find_peaks

def find_extrema(x, y):
    """Find global maximum and minimum"""
    global_max_idx = np.argmax(y)
    global_min_idx = np.argmin(y)
    return np.array([global_max_idx]), np.array([global_min_idx])

def plot_latency_metrics(log_dir="metrics/runs"):
    """Plot latency metrics from gesture control csv files"""
    # get all csv files in the directory
    log_dir = Path(log_dir)
    latency_files = sorted(glob.glob(str(log_dir / "gesture_latency_*.csv")))
    system_files = sorted(glob.glob(str(log_dir / "system_metrics_*.csv")))
    
    if not latency_files:
        print("No metric files found!")
        return
    
    # read all csv files and combine them
    dfs = []
    for i, file in enumerate(latency_files, 1):
        df = pd.read_csv(file)
        df['run_id'] = f"Run {i}"
        
        # normalize timestamps to start from 0 for each run
        df['runtime'] = df['gesture_timestamp'] - df['gesture_timestamp'].min()
        
        # convert latency to microseconds
        df['latency_us'] = df['latency_ms'] * 1000
        
        dfs.append(df)
    
    combined_df = pd.concat(dfs, ignore_index=True)
    unique_runs = sorted(combined_df['run_id'].unique())
    colors = plt.cm.berlin(np.linspace(0, 1, len(unique_runs)))
    
    # Plot 1: Latency over time
    fig_latency = plt.figure(figsize=(12, 6))
    ax_latency = fig_latency.add_subplot(111)
    
    for i, run_id in enumerate(unique_runs):
        run_data = combined_df[combined_df['run_id'] == run_id]
        
        # sort by runtime to ensure correct line plot
        run_data = run_data.sort_values('runtime')
        x = run_data['runtime'].values
        y = run_data['latency_us'].values  # use microseconds
        
        # plot main line
        ax_latency.plot(x, y, "o-", alpha=0.8, label=f'{run_id}', color=colors[i], linewidth=2)
        
        # find and plot global maximum only
        max_idx, _ = find_extrema(x, y)
        
        # plot maximum
        y_offset = 0.1 * (y.max() - y.min())  # offset for annotations
        for idx in max_idx:
            ax_latency.annotate(f'max: {y[idx]:.1f}µs\nt: {x[idx]:.1f}s',
                            xy=(x[idx], y[idx]),
                            xytext=(x[idx], y.min() - y_offset * (2 if run_id == "Run 1" else 1)),
                            ha='center', va='top',
                            color=colors[i],
                            arrowprops=dict(arrowstyle='->', color=colors[i]))
    
    ax_latency.set_title("Gesture to Control Latency Over Time")
    ax_latency.set_xlabel("Runtime (s)")
    ax_latency.set_ylabel("Latency (µs)")
    ax_latency.grid(True)
    ax_latency.legend(title="Run ID")
    plt.tight_layout()
    plt.show()
    
    # Plot 2: Latency by gesture type
    fig_gesture = plt.figure(figsize=(12, 6))
    ax_gesture = fig_gesture.add_subplot(111)
    
    gesture_latency = combined_df.groupby(['gesture_type', 'run_id'])['latency_us'].agg(['mean', 'std'])
    gesture_latency = gesture_latency.reset_index()
    
    # pivot for grouped bar plot
    pivot_df = gesture_latency.pivot(index='gesture_type', columns='run_id', values='mean')
    pivot_std = gesture_latency.pivot(index='gesture_type', columns='run_id', values='std')
    
    pivot_df.plot(kind='bar', yerr=pivot_std, ax=ax_gesture)
    ax_gesture.set_title('Average Latency by Gesture Type and Run')
    ax_gesture.set_xlabel('Gesture Type')
    ax_gesture.set_ylabel('Latency (µs)')
    ax_gesture.grid(True)
    plt.tight_layout()
    plt.show()
    
    # Plot 3: Latency distribution
    fig_dist = plt.figure(figsize=(12, 6))
    ax_dist = fig_dist.add_subplot(111)
    
    for run_id, color in zip(unique_runs, colors):
        run_data = combined_df[combined_df['run_id'] == run_id]
        ax_dist.hist(run_data['latency_us'], bins=20, alpha=0.5, 
                    label=run_id, color=color)
    
    ax_dist.set_title('Latency Distribution by Run')
    ax_dist.set_xlabel('Latency (µs)')
    ax_dist.set_ylabel('Count')
    ax_dist.grid(True)
    ax_dist.legend(title='Run ID')
    plt.tight_layout()
    plt.show()
    
    # System metrics plots
    if system_files:
        system_df = pd.read_csv(system_files[-1])  # use most recent system metrics file
        system_df['runtime'] = system_df['timestamp'] - system_df['timestamp'].min()
        
        # CPU usage plot
        fig_cpu = plt.figure(figsize=(10, 6))
        ax_cpu = fig_cpu.add_subplot(111)
        ax_cpu.plot(system_df['runtime'], system_df['cpu_percent'], 'b-')
        ax_cpu.set_title('CPU Usage Over Time')
        ax_cpu.set_xlabel('Runtime (s)')
        ax_cpu.set_ylabel('CPU Usage (%)')
        ax_cpu.grid(True)
        ax_cpu.set_ylim(0, 100)
        plt.tight_layout()
        plt.show()
        
        # Memory usage plot
        fig_mem = plt.figure(figsize=(10, 6))
        ax_mem = fig_mem.add_subplot(111)
        ax_mem.plot(system_df['runtime'], system_df['memory_percent'], 'g-')
        ax_mem.set_title('Memory Usage Over Time')
        ax_mem.set_xlabel('Runtime (s)')
        ax_mem.set_ylabel('Memory Usage (%)')
        ax_mem.grid(True)
        ax_mem.set_ylim(0, 100)
        plt.tight_layout()
        plt.show()
        
        # CPU temperature plot (if available)
        if system_df['cpu_temp'].max() > 0:
            fig_temp = plt.figure(figsize=(10, 6))
            ax_temp = fig_temp.add_subplot(111)
            ax_temp.plot(system_df['runtime'], system_df['cpu_temp'], 'r-')
            ax_temp.set_title('CPU Temperature Over Time')
            ax_temp.set_xlabel('Runtime (s)')
            ax_temp.set_ylabel('Temperature (°C)')
            ax_temp.grid(True)
            
            # add a horizontal line at 80°C as a warning threshold
            ax_temp.axhline(y=80, color='orange', linestyle='--', alpha=0.5)
            ax_temp.text(0, 82, 'Warning Threshold', color='orange')
            plt.tight_layout()
            plt.show()
    
    # print statistics for each run
    print("\nLatency Statistics by Run:")
    for run_id in unique_runs:
        run_data = combined_df[combined_df['run_id'] == run_id]
        print(f"\n{run_id}:")
        print(f"  Runtime: {run_data['runtime'].max():.2f} seconds")
        print(f"  Average latency: {run_data['latency_us'].mean():.2f} µs")
        print(f"  Median latency: {run_data['latency_us'].median():.2f} µs")
        print(f"  Max latency: {run_data['latency_us'].max():.2f} µs")
        print(f"  Min latency: {run_data['latency_us'].min():.2f} µs")
        print(f"  Number of gestures: {len(run_data)}")
    
    # print overall statistics
    print("\nOverall Statistics:")
    print(f"Total number of runs: {len(unique_runs)}")
    print(f"Total number of gestures: {len(combined_df)}")
    print(f"Global average latency: {combined_df['latency_us'].mean():.2f} µs")
    print(f"Global median latency: {combined_df['latency_us'].median():.2f} µs")
    
    # print gesture type statistics
    print("\nLatency by gesture type:")
    print(combined_df.groupby('gesture_type')['latency_us'].agg(['mean', 'std', 'count']))
    
    # print system metrics statistics if available
    if system_files:
        print("\nSystem Metrics Statistics:")
        system_df = pd.read_csv(system_files[-1])
        print(f"Average CPU usage: {system_df['cpu_percent'].mean():.2f}%")
        print(f"Max CPU usage: {system_df['cpu_percent'].max():.2f}%")
        print(f"Average memory usage: {system_df['memory_percent'].mean():.2f}%")
        print(f"Max memory usage: {system_df['memory_percent'].max():.2f}%")
        if system_df['cpu_temp'].max() > 0:
            print(f"Average CPU temperature: {system_df['cpu_temp'].mean():.2f}°C")
            print(f"Max CPU temperature: {system_df['cpu_temp'].max():.2f}°C")

if __name__ == "__main__":
    plot_latency_metrics() 