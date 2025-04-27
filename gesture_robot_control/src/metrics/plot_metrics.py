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
    csv_files = sorted(glob.glob(str(log_dir / "gesture_latency_*.csv")))
    
    if not csv_files:
        print("No metric files found!")
        return
    
    # read all csv files and combine them
    dfs = []
    for i, file in enumerate(csv_files, 1):
        df = pd.read_csv(file)
        df['run_id'] = f"Run {i}"
        
        # normalize timestamps to start from 0 for each run
        df['runtime'] = df['gesture_timestamp'] - df['gesture_timestamp'].min()
        
        # convert latency to microseconds
        df['latency_us'] = df['latency_ms'] * 1000
        
        dfs.append(df)
    
    combined_df = pd.concat(dfs, ignore_index=True)
    
    # create figure with subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12))
    
    unique_runs = sorted(combined_df['run_id'].unique())
    colors = plt.cm.berlin(np.linspace(0, 1, len(unique_runs)))
    
    # increase space at bottom for annotations
    ax1.set_position([0.1, 0.7, 0.8, 0.25])  # [left, bottom, width, height]
    
    for i, run_id in enumerate(unique_runs):
        run_data = combined_df[combined_df['run_id'] == run_id]
        
        # sort by runtime to ensure correct line plot
        run_data = run_data.sort_values('runtime')
        x = run_data['runtime'].values
        y = run_data['latency_us'].values  # use microseconds
        
        # plot main line
        ax1.plot(x, y, "o-", alpha=0.8, label=f'Run {run_id}', color=colors[i], linewidth=2)
        
        # find and plot global maximum only
        max_idx, _ = find_extrema(x, y)
        
        # plot maximum
        y_offset = 0.1 * (y.max() - y.min())  # offset for annotations
        for idx in max_idx:
            ax1.annotate(f'max: {y[idx]:.1f}µs\nt: {x[idx]:.1f}s',
                        xy=(x[idx], y[idx]),
                        xytext=(x[idx], y.min() - y_offset * (2 if run_id == "Run 1" else 1)),
                        ha='center', va='top',
                        color=colors[i],
                        arrowprops=dict(arrowstyle='->', color=colors[i]))
    
    # for stacked subplots, it's cleaner to omit the x-axis label on upper plots
    ax1.set(title="Gesture to Control Latency Over Time", ylabel="Latency (µs)")
    ax1.grid(True)
    ax1.legend(title="Run ID")
    ax1.tick_params(labelbottom=False)
    
    # plot 2: latency by gesture type and run (in microseconds)
    gesture_latency = combined_df.groupby(['gesture_type', 'run_id'])['latency_us'].agg(['mean', 'std'])
    gesture_latency = gesture_latency.reset_index()
    
    # pivot for grouped bar plot
    pivot_df = gesture_latency.pivot(index='gesture_type', columns='run_id', values='mean')
    pivot_std = gesture_latency.pivot(index='gesture_type', columns='run_id', values='std')
    
    pivot_df.plot(kind='bar', yerr=pivot_std, ax=ax2)
    ax2.set_title('Average Latency by Gesture Type and Run')
    ax2.set_xlabel('Gesture Type')
    ax2.set_ylabel('Latency (µs)')
    ax2.grid(True)
    
    # plot 3: latency distribution by run (in microseconds)
    for run_id, color in zip(unique_runs, colors):
        run_data = combined_df[combined_df['run_id'] == run_id]
        ax3.hist(run_data['latency_us'], bins=20, alpha=0.5, 
                label=run_id, color=color)
    
    ax3.set_title('Latency Distribution by Run')
    ax3.set_xlabel('Latency (µs)')
    ax3.set_ylabel('Count')
    ax3.grid(True)
    ax3.legend(title='Run ID')
    
    # adjust layout and show
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

if __name__ == "__main__":
    plot_latency_metrics() 