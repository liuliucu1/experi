#!/usr/bin/env python3
import os
import glob
import struct
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
import argparse
from pathlib import Path

# Function to read binary files
def read_bin_file(file_path):
    """
    Read binary files containing labels and scores.
    Based on the format in detector_main.hpp's do_save2 function.
    """
    labels = []
    scores = []
    
    try:
        with open(file_path, 'rb') as f:
            # Read label size
            label_size_bytes = f.read(8)  # size_t is 8 bytes
            if not label_size_bytes:
                return np.array([]), np.array([])
            
            label_size = struct.unpack('Q', label_size_bytes)[0]
            
            # Read labels (boolean values)
            for _ in range(label_size):
                label_byte = f.read(1)
                if not label_byte:
                    break
                label = struct.unpack('?', label_byte)[0]
                labels.append(1 if label else 0)
            
            # Read score size
            score_size_bytes = f.read(8)  # size_t is 8 bytes
            if not score_size_bytes:
                return np.array(labels), np.array([])
            
            score_size = struct.unpack('Q', score_size_bytes)[0]
            
            # Read scores (double values)
            score_bytes = f.read(score_size * 8)  # double is 8 bytes
            scores = list(struct.unpack(f'{score_size}d', score_bytes))
            
    except Exception as e:
        print(f"Error reading bin file {file_path}: {e}")
        return np.array([]), np.array([])
    
    return np.array(labels), np.array(scores)

def calculate_metrics(labels, scores, threshold=11):
    """Calculate F1, Accuracy, Precision, and Recall metrics."""
    if len(labels) == 0 or len(scores) == 0:
        return 0, 0, 0, 0
    
    # Convert scores to binary predictions using threshold
    predictions = (scores > threshold).astype(int)
    
    # Calculate metrics
    try:
        f1 = f1_score(labels, predictions)
        acc = accuracy_score(labels, predictions)
        precision = precision_score(labels, predictions)
        recall = recall_score(labels, predictions)
    except Exception as e:
        print(f"Error calculating metrics: {e}")
        return 0, 0, 0, 0
    
    return f1, acc, precision, recall

def count_anomalies(labels):
    """Count the number of anomalies (label=1) in the dataset."""
    return np.sum(labels == 1)

def process_directory(directory_path):
    """Process all bin files in a directory and calculate metrics."""
    # Find all bin files in the directory
    bin_files = sorted(glob.glob(os.path.join(directory_path, "window_*_results.bin")))
    
    if not bin_files:
        print(f"No bin files found in {directory_path}")
        return None
    
    # Initialize results dictionary
    results = {
        'window_nums': [],
        'f1_scores': [],
        'accuracies': [],
        'precisions': [],
        'recalls': [],
        'anomaly_counts': []
    }
    
    # Process each bin file
    for bin_file in bin_files:
        # Extract window number from filename
        window_num = int(os.path.basename(bin_file).split('_')[1])
        
        # Read labels and scores
        labels, scores = read_bin_file(bin_file)
        
        if len(labels) == 0 or len(scores) == 0:
            print(f"Warning: Empty data in {bin_file}")
            continue
        
        # Calculate metrics
        f1, acc, precision, recall = calculate_metrics(labels, scores)
        
        # Count anomalies
        anomaly_count = count_anomalies(labels)
        
        # Store results
        results['window_nums'].append(window_num)
        results['f1_scores'].append(f1)
        results['accuracies'].append(acc)
        results['precisions'].append(precision)
        results['recalls'].append(recall)
        results['anomaly_counts'].append(anomaly_count)
    
    return results

def create_metrics_visualization(results, output_dir, dataset_name):
    """Create visualizations for metrics."""
    if not results or len(results['window_nums']) == 0:
        print(f"No results to visualize for {dataset_name}")
        return
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Sort results by window number
    indices = np.argsort(results['window_nums'])
    window_nums = np.array(results['window_nums'])[indices]
    f1_scores = np.array(results['f1_scores'])[indices]
    accuracies = np.array(results['accuracies'])[indices]
    precisions = np.array(results['precisions'])[indices]
    recalls = np.array(results['recalls'])[indices]
    anomaly_counts = np.array(results['anomaly_counts'])[indices]
    
    # Create metrics plot
    plt.figure(figsize=(12, 8))
    plt.plot(window_nums, f1_scores, 'o-', label='F1 Score')
    plt.plot(window_nums, accuracies, 's-', label='Accuracy')
    plt.plot(window_nums, precisions, '^-', label='Precision')
    plt.plot(window_nums, recalls, 'D-', label='Recall')
    plt.xlabel('Window Number')
    plt.ylabel('Metric Value')
    plt.title(f'Performance Metrics by Window - {dataset_name}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, f'{dataset_name}_metrics.png'), dpi=300)
    plt.close()
    
    # Create anomaly count plot
    plt.figure(figsize=(12, 6))
    plt.bar(window_nums, anomaly_counts, color='crimson', alpha=0.7)
    plt.xlabel('Window Number')
    plt.ylabel('Anomaly Count')
    plt.title(f'Anomaly Count by Window - {dataset_name}')
    plt.grid(True, alpha=0.3, axis='y')
    plt.savefig(os.path.join(output_dir, f'{dataset_name}_anomalies.png'), dpi=300)
    plt.close()
    
    # Save metrics to CSV
    with open(os.path.join(output_dir, f'{dataset_name}_metrics.csv'), 'w') as f:
        f.write('Window,F1,Accuracy,Precision,Recall,AnomalyCount\n')
        for i in range(len(window_nums)):
            f.write(f'{window_nums[i]},{f1_scores[i]},{accuracies[i]},{precisions[i]},{recalls[i]},{anomaly_counts[i]}\n')

def main():
    """Main function to process all datasets."""
    parser = argparse.ArgumentParser(description='Visualize metrics from CIC2017 dataset.')
    parser.add_argument('--input-dir', type=str, 
                        default='/home/sduu2/userspace-20T-1/yyr/HyperVision_Cpp/analysis_output/cic2017',
                        help='Input directory containing CIC2017 dataset results')
    parser.add_argument('--output-dir', type=str, 
                        default='/home/sduu2/userspace-20T-1/yyr/HyperVision_Cpp/analysis_output/cic2017/visualizations',
                        help='Output directory for visualizations')
    args = parser.parse_args()
    
    # Get all subdirectories in the input directory
    subdirs = [d for d in os.listdir(args.input_dir) 
               if os.path.isdir(os.path.join(args.input_dir, d)) and d != 'visualizations']
    
    if not subdirs:
        print(f"No subdirectories found in {args.input_dir}")
        return
    
    print(f"Found {len(subdirs)} datasets: {', '.join(subdirs)}")
    
    # Process each subdirectory
    for subdir in subdirs:
        dataset_path = os.path.join(args.input_dir, subdir)
        dataset_output_dir = os.path.join(args.output_dir, subdir)
        
        print(f"Processing dataset: {subdir}")
        results = process_directory(dataset_path)
        
        if results:
            create_metrics_visualization(results, dataset_output_dir, subdir)
            print(f"Visualizations created for {subdir}")
    
    # Create combined visualization for all datasets
    create_combined_visualization(args.input_dir, subdirs, args.output_dir)

def create_combined_visualization(input_dir, datasets, output_dir):
    """Create combined visualizations for all datasets."""
    # Initialize combined results
    combined_results = {
        'datasets': [],
        'f1_scores': [],
        'accuracies': [],
        'precisions': [],
        'recalls': []
    }
    
    # Process each dataset
    for dataset in datasets:
        dataset_path = os.path.join(input_dir, dataset)
        results = process_directory(dataset_path)
        
        if results and len(results['window_nums']) > 0:
            # Calculate average metrics for this dataset
            avg_f1 = np.mean(results['f1_scores'])
            avg_acc = np.mean(results['accuracies'])
            avg_precision = np.mean(results['precisions'])
            avg_recall = np.mean(results['recalls'])
            
            # Store results
            combined_results['datasets'].append(dataset)
            combined_results['f1_scores'].append(avg_f1)
            combined_results['accuracies'].append(avg_acc)
            combined_results['precisions'].append(avg_precision)
            combined_results['recalls'].append(avg_recall)
    
    if len(combined_results['datasets']) == 0:
        print("No combined results to visualize")
        return
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create combined metrics plot
    plt.figure(figsize=(14, 10))
    
    x = np.arange(len(combined_results['datasets']))
    width = 0.2
    
    plt.bar(x - 1.5*width, combined_results['f1_scores'], width, label='F1 Score')
    plt.bar(x - 0.5*width, combined_results['accuracies'], width, label='Accuracy')
    plt.bar(x + 0.5*width, combined_results['precisions'], width, label='Precision')
    plt.bar(x + 1.5*width, combined_results['recalls'], width, label='Recall')
    
    plt.xlabel('Dataset')
    plt.ylabel('Average Metric Value')
    plt.title('Average Performance Metrics by Dataset')
    plt.xticks(x, combined_results['datasets'])
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')
    plt.savefig(os.path.join(output_dir, 'combined_metrics.png'), dpi=300)
    plt.close()
    
    # Save combined metrics to CSV
    with open(os.path.join(output_dir, 'combined_metrics.csv'), 'w') as f:
        f.write('Dataset,F1,Accuracy,Precision,Recall\n')
        for i in range(len(combined_results['datasets'])):
            f.write(f'{combined_results["datasets"][i]},{combined_results["f1_scores"][i]},{combined_results["accuracies"][i]},{combined_results["precisions"][i]},{combined_results["recalls"][i]}\n')

if __name__ == "__main__":
    main()