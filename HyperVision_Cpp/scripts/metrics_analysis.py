import os
import struct
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, roc_curve


def read_binary_results(file_path):
    results = []
    with open(file_path, 'rb') as f:
        # Read label size
        label_size = struct.unpack('Q', f.read(8))[0]
        
        # Read labels
        labels = np.array([struct.unpack('?', f.read(1))[0] for _ in range(label_size)])
        
        # Read score size
        score_size = struct.unpack('Q', f.read(8))[0]
        
        # Read scores
        scores = np.array(struct.unpack(f'{score_size}d', f.read(8 * score_size)))
        
        results.append((labels, scores))
    return results


def calculate_metrics(labels, scores):
    acc = accuracy_score(labels, scores >11)
    f1 = f1_score(labels, scores >11)
    
    # Check if there are at least two classes in the labels
    if len(np.unique(labels)) < 2:
        auc = 0.5  # Default value for undefined AUC
        fpr, tpr = np.array([0, 1]), np.array([0, 1])  # Default ROC curve
    else:
        auc = roc_auc_score(labels, scores)
        fpr, tpr, _ = roc_curve(labels, scores)
    
    return acc, f1, auc, fpr, tpr


def plot_metrics(window_results, output_dir):
    acc_values = []
    f1_values = []
    auc_values = []
    
    # Create figure for metrics
    plt.figure(figsize=(20, 5))
    
    # Plot ACC, F1, AUC over windows
    plt.subplot(1, 4, 1)
    for i, (labels, scores) in enumerate(window_results):
        acc, f1, auc, _, _ = calculate_metrics(labels, scores)
        acc_values.append(acc)
        f1_values.append(f1)
        auc_values.append(auc)
    
    plt.plot(acc_values, label='Accuracy')
    plt.plot(f1_values, label='F1 Score')
    plt.plot(auc_values, label='AUC')
    plt.xlabel('Window Index')
    plt.ylabel('Score')
    plt.title('Metrics Over Windows')
    plt.legend()
    
    # Plot ROC Curve for last window
    plt.subplot(1, 4, 2)
    _, _, _, fpr, tpr = calculate_metrics(*window_results[-1])
    plt.plot(fpr, tpr, label='ROC Curve')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve (Last Window)')
    plt.legend()
    
    # Plot score distributions for each window
    plt.subplot(1, 4, 3)
    for i, (labels, scores) in enumerate(window_results):
        # Plot scores with color based on labels
        plt.scatter([i] * len(scores), scores, c=labels, cmap='coolwarm', alpha=0.5, s=10)
    plt.xlabel('Window Index')
    plt.ylabel('Score')
    plt.title('Score Distribution by Window')
    plt.colorbar(label='Label (0/1)')
    
    # Plot combined score distribution
    plt.subplot(1, 4, 4)
    all_scores = np.concatenate([scores for _, scores in window_results])
    all_labels = np.concatenate([labels for labels, _ in window_results])
    plt.hist([all_scores[all_labels == 0], all_scores[all_labels == 1]], 
             bins=50, stacked=True, color=['blue', 'red'], alpha=0.7)
    plt.xlabel('Score')
    plt.ylabel('Count')
    plt.title('Combined Score Distribution')
    plt.legend(['Label 0', 'Label 1'])
    
    # Save figure
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'metrics_visualization.png'))
    plt.close()


def analyze_results(results_dir, output_dir):
    window_results = []
    
    # Read all window result files
    for filename in sorted(os.listdir(results_dir)):
        if filename.endswith('_results.bin'):
            file_path = os.path.join(results_dir, filename)
            window_results.extend(read_binary_results(file_path))
    
    # Calculate and plot metrics
    plot_metrics(window_results, output_dir)


if __name__ == '__main__':
    results_dir = '/home/sduu2/userspace-20T-1/yyr/HyperVision_Cpp/analysis_output/evasion'  # Directory containing binary results
    output_dir = '/home/sduu2/userspace-20T-1/yyr/HyperVision_Cpp/analysis_output/evasion/visualizations'  # Directory to save visualizations
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    analyze_results(results_dir, output_dir)
