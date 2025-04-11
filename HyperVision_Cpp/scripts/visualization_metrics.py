#!/usr/bin/env python3
import os
import re
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from datetime import datetime, timedelta
import subprocess
import time
import random
import tempfile
import json

# Configuration
RESULTS_DIR = "/home/sduu2/userspace-20T-1/yyr/HyperVision_results"
OUTPUT_DIR = "/home/sduu2/userspace-20T-1/yyr/HyperVision_results/visualizations"
DATA_PATH = "/home/sduu2/userspace-20T-1/yyr/data/dns_lrscan.data"
LABEL_PATH = "/home/sduu2/userspace-20T-1/yyr/data/dns_lrscan.label"
BUILD_DIR = "/home/sduu2/build_hypervision"  # Updated path to the correct build directory
HYPERVISION_DIR = "/home/sduu2/userspace-20T-1/yyr/HyperVision_Cpp"
MAX_PACKETS = 50000  # Set back to a reasonable number for testing
WINDOW_SIZE = 5000    # Process 5000 packets at a time
NUM_WINDOWS = MAX_PACKETS // WINDOW_SIZE

# Create output directories if they don't exist
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(RESULTS_DIR, "features"), exist_ok=True)

def create_config_file(approach, window_size):
    """Create a configuration file for HyperVision with the specified approach."""
    # Base configuration
    config = {
        "dataset_construct": {
            "data_path": DATA_PATH,
            "label_path": LABEL_PATH,
            "train_ratio": 0.0
        },
        "flow_construct": {
            "flow_time_out": 10.0,
            "evict_flow_time_out": 5.0
        },
        "edge_construct": {
            "length_bin_size": 10,
            "edge_long_line": 15,
            "edge_agg_line": 20
        },
        "graph_analyze": {
            "select_ratio": 0.01,
            "offset_l": 5.0,
            "offset_s": 0.0,
            "al": 0.1,
            "bl": 1.7,
            "cl": 0.5,
            "as": 0.1,
            "bs": 1.7,
            "cs": 0.5,
            "uc": 0.004,
            "vc": 40,
            "us": 0.004,
            "vs": 40,
            "ul": 0.004,
            "vl": 40,
            "window_size": window_size
        },
        "result_save": {
            "save_result_enable": True,
            "save_result_path": os.path.join(RESULTS_DIR, f"{approach}_result.txt"),
            "save_final_result_path": os.path.join(RESULTS_DIR, f"{approach}_final_result.txt"),
            "save_feature_path": os.path.join(RESULTS_DIR, "features", f"{approach}_feature.txt"),
            "save_feature_path_s": os.path.join(RESULTS_DIR, "features", f"{approach}_s.txt"),
            "save_feature_path_l": os.path.join(RESULTS_DIR, "features", f"{approach}_l.txt")
        }
    }
    
    # Add approach-specific configuration
    if approach == "baseline":
        # Baseline parameters - use default
        pass
    elif approach == "dynamic":
        # Dynamic clustering parameters
        config["graph_analyze"]["select_ratio"] = 0.02
        config["graph_analyze"]["al"] = 0.15
        config["graph_analyze"]["bl"] = 1.5
    elif approach == "priority":
        # Priority queue parameters
        config["graph_analyze"]["select_ratio"] = 0.03
        config["graph_analyze"]["as"] = 0.15
        config["graph_analyze"]["bs"] = 1.5
    elif approach == "realtime":
        # Realtime parameters
        config["graph_analyze"]["select_ratio"] = 0.01
        config["graph_analyze"]["ul"] = 0.005
        config["graph_analyze"]["vl"] = 35
    
    # Create a temporary file for the configuration
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(config, f, indent=4)
        config_file = f.name
    
    return config_file

def run_hypervision(approach):
    """Run HyperVision with the specified approach."""
    print(f"Running HyperVision with {approach} approach...")
    
    # Create a configuration file with window size
    config_file = create_config_file(approach, WINDOW_SIZE)
    
    # Command to run HyperVision
    cmd = [
        f"{BUILD_DIR}/HyperVision",
        "-config", config_file
    ]
    
    try:
        # Run HyperVision
        process = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(f"HyperVision {approach} completed successfully")
        return True
    
    except subprocess.CalledProcessError as e:
        print(f"Error running HyperVision: {e}")
        print(f"Stdout: {e.stdout.decode()}")
        print(f"Stderr: {e.stderr.decode()}")
        return False
    finally:
        # Clean up
        os.unlink(config_file)

def load_dataset_labels(start_idx=0, end_idx=MAX_PACKETS):
    """Load labels from the dataset to count anomalies within a specific range."""
    print(f"Loading labels from {LABEL_PATH} (packets {start_idx} to {end_idx})...")
    
    try:
        with open(LABEL_PATH, 'rb') as f:
            # Skip to start_idx
            f.seek(start_idx)
            # Read only the portion we need
            labels = f.read(end_idx - start_idx)
        
        # Count anomalies
        anomaly_count = sum(1 for label in labels if label == 1)
        
        return labels, anomaly_count
    
    except Exception as e:
        print(f"Error loading labels: {e}")
        return [], 0

def extract_metrics_from_results(approach, window_idx):
    """Extract metrics from HyperVision results for a specific window."""
    # Define paths to result files
    result_file = os.path.join(RESULTS_DIR, f"{approach}_result.txt")
    
    # Initialize metrics
    metrics = {
        "precision": 0.0,
        "f1_score": 0.0,
        "accuracy": 0.0,
        "nmi": 0.0
    }
    
    try:
        # Read the result file
        with open(result_file, 'r') as f:
            content = f.read()
        
        # Extract metrics for the specific window
        window_pattern = rf"Window {window_idx}.*?Precision: ([\d\.]+).*?F1 Score: ([\d\.]+).*?Accuracy: ([\d\.]+).*?NMI: ([\d\.]+)"
        match = re.search(window_pattern, content, re.DOTALL)
        
        if match:
            metrics["precision"] = float(match.group(1))
            metrics["f1_score"] = float(match.group(2))
            metrics["accuracy"] = float(match.group(3))
            metrics["nmi"] = float(match.group(4))
        else:
            # If we can't find window-specific metrics, try to find overall metrics
            precision_match = re.search(r"Precision: ([\d\.]+)", content)
            f1_match = re.search(r"F1 Score: ([\d\.]+)", content)
            accuracy_match = re.search(r"Accuracy: ([\d\.]+)", content)
            nmi_match = re.search(r"NMI: ([\d\.]+)", content)
            
            if precision_match:
                metrics["precision"] = float(precision_match.group(1))
            if f1_match:
                metrics["f1_score"] = float(f1_match.group(1))
            if accuracy_match:
                metrics["accuracy"] = float(accuracy_match.group(1))
            if nmi_match:
                metrics["nmi"] = float(nmi_match.group(1))
        
        return metrics
    
    except Exception as e:
        print(f"Error extracting metrics for {approach}, window {window_idx}: {e}")
        # Return default metrics if extraction fails
        return metrics

def process_results_over_time():
    """Process HyperVision results to extract metrics over time."""
    approaches = ["baseline", "dynamic", "priority", "realtime"]
    
    # Initialize time series data for metrics
    time_points = []
    anomaly_counts = []
    metrics_over_time = {
        approach: {
            "precision": [],
            "f1_score": [],
            "accuracy": [],
            "nmi": []
        } for approach in approaches
    }
    
    # Process each window
    for window_idx in range(NUM_WINDOWS):
        start_idx = window_idx * WINDOW_SIZE
        end_idx = min((window_idx + 1) * WINDOW_SIZE, MAX_PACKETS)
        
        print(f"Processing window {window_idx+1}/{NUM_WINDOWS} (packets {start_idx} to {end_idx})...")
        
        # Load labels and count anomalies for this window
        labels, anomaly_count = load_dataset_labels(start_idx, end_idx)
        
        # Create a time point (using window index as a proxy for time)
        time_point = window_idx + 1
        time_points.append(time_point)
        anomaly_counts.append(anomaly_count)
        
        # Extract metrics for each approach
        for approach in approaches:
            window_metrics = extract_metrics_from_results(approach, window_idx)
            
            # Store metrics
            for metric_name, metric_value in window_metrics.items():
                metrics_over_time[approach][metric_name].append(metric_value)
    
    return time_points, anomaly_counts, metrics_over_time

def plot_metrics_over_time(time_points, anomaly_counts, metrics_over_time):
    """Create plots for each metric showing how they change over time."""
    approaches = ["baseline", "dynamic", "priority", "realtime"]
    colors = ['blue', 'green', 'red', 'purple']
    metrics = ["precision", "f1_score", "accuracy", "nmi"]
    metric_titles = ["Precision", "F1 Score", "Accuracy", "NMI"]
    
    for metric_idx, metric in enumerate(metrics):
        plt.figure(figsize=(12, 6))
        
        # Plot metrics for each approach
        for i, approach in enumerate(approaches):
            plt.plot(time_points, metrics_over_time[approach][metric], 
                     marker='o', linestyle='-', color=colors[i], label=approach.capitalize())
        
        # Plot anomaly counts as a bar chart on a secondary axis
        ax1 = plt.gca()
        ax2 = ax1.twinx()
        ax2.bar(time_points, anomaly_counts, alpha=0.3, color='gray', label='Anomaly Count')
        
        # Set labels and title
        ax1.set_xlabel('Time Window')
        ax1.set_ylabel(metric_titles[metric_idx])
        ax2.set_ylabel('Anomaly Count')
        plt.title(f'{metric_titles[metric_idx]} Over Time')
        
        # Set x-axis to show integer values
        ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
        
        # Add legends
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        
        # Save the plot
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f"{metric}_plot.png"))
        print(f"Plot saved to {os.path.join(OUTPUT_DIR, f'{metric}_plot.png')}")
        plt.close()

def main():
    """Main function to run the visualization process."""
    # Run HyperVision for each approach
    approaches = ["baseline", "dynamic", "priority", "realtime"]
    for approach in approaches:
        run_hypervision(approach)
    
    # Process results to extract metrics over time
    time_points, anomaly_counts, metrics_over_time = process_results_over_time()
    
    # Create visualizations
    plot_metrics_over_time(time_points, anomaly_counts, metrics_over_time)
    
    print(f"All plots saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()