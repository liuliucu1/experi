import os
import numpy as np
import matplotlib.pyplot as plt
from glob import glob

def read_binary_file(filename):
    """Read binary file containing labels and scores."""
    with open(filename, 'rb') as f:
        # Read labels
        label_size = int.from_bytes(f.read(8), byteorder='little')
        labels = np.array([bool.from_bytes(f.read(1), byteorder='little') for _ in range(label_size)])
        
        # Read scores
        score_size = int.from_bytes(f.read(8), byteorder='little')
        scores = np.frombuffer(f.read(score_size * 8), dtype=np.float64)
        
    return labels, scores

def visualize_window_results(filename, output_dir):
    """Create visualizations for a single window's results."""
    labels, scores = read_binary_file(filename)
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    fig.suptitle(f'Results for {os.path.basename(filename)}')
    
    # Plot 1: Labels and Scores overlay
    ax1.plot(scores, label='Detection Scores', alpha=0.7)
    ax1.plot(labels.astype(int), label='True Labels', alpha=0.5)
    ax1.set_title('Detection Scores vs True Labels')
    ax1.set_xlabel('Sample Index')
    ax1.set_ylabel('Value')
    ax1.legend()
    ax1.grid(True)
    
    # Plot 2: ROC curve
    from sklearn.metrics import roc_curve, auc
    fpr, tpr, _ = roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)
    
    ax2.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
    ax2.plot([0, 1], [0, 1], 'k--')
    ax2.set_title('ROC Curve')
    ax2.set_xlabel('False Positive Rate')
    ax2.set_ylabel('True Positive Rate')
    ax2.legend()
    ax2.grid(True)
    
    # Adjust layout and save
    plt.tight_layout()
    output_path = os.path.join(output_dir, f'{os.path.splitext(os.path.basename(filename))[0]}_viz.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return output_path

def main():
    # Create output directory for visualizations
    results_dir = "/home/sduu2/userspace-20T-1/yyr/HyperVision_Cpp/analysis_output/cic2017/Friday"
    viz_output_dir = os.path.join(results_dir, "visualizations")
    os.makedirs(viz_output_dir, exist_ok=True)
    
    # Find all binary result files
    bin_files = glob(os.path.join(results_dir, "window_*.bin"))
    
    if not bin_files:
        print(f"No binary result files found in {results_dir}")
        return
    
    # Process each file
    for bin_file in sorted(bin_files):
        try:
            output_path = visualize_window_results(bin_file, viz_output_dir)
            print(f"Generated visualization: {output_path}")
        except Exception as e:
            print(f"Error processing {bin_file}: {str(e)}")

if __name__ == "__main__":
    main()
