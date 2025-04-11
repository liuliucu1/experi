#!/usr/bin/env python3
import os
import struct
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import curve_fit
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches

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

# Gaussian distribution function
def gaussian(x, amp, mu, sigma):
    return amp * np.exp(-(x - mu)**2 / (2 * sigma**2))

# Heavy-tailed distribution (Pareto) function
def pareto(x, amp, alpha, xm):
    return amp * alpha * (xm**alpha) / (x**(alpha+1)) * (x >= xm)

# Function to fit distributions and plot
def analyze_score_distribution(file_path, bin_width=3, output_dir=None):
    """
    Analyze score distribution with Gaussian and heavy-tailed fitting.
    Creates a histogram with color-coded bars based on labels.
    """
    # Read labels and scores
    labels, scores = read_bin_file(file_path)
    
    if len(labels) == 0 or len(scores) == 0:
        print(f"Error: No data found in {file_path}")
        return
    
    # Print basic statistics
    print(f"Total entries: {len(scores)}")
    print(f"Score range: [{np.min(scores):.2f}, {np.max(scores):.2f}]")
    print(f"Mean score: {np.mean(scores):.2f}")
    print(f"Median score: {np.median(scores):.2f}")
    print(f"Standard deviation: {np.std(scores):.2f}")
    
    # Separate scores by label
    normal_scores = scores[labels == 0]
    anomaly_scores = scores[labels == 1]
    
    print(f"Normal entries: {len(normal_scores)}")
    print(f"Anomaly entries: {len(anomaly_scores)}")
    
    # Define bin edges based on bin_width
    min_score = np.floor(np.min(scores))
    max_score = np.ceil(np.max(scores))
    bin_edges = np.arange(min_score, max_score + bin_width, bin_width)
    
    # Create figure
    plt.figure(figsize=(14, 10))
    
    # Create histogram for all scores
    hist, bin_edges = np.histogram(scores, bins=bin_edges)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Create separate histograms for normal and anomaly scores
    hist_normal, _ = np.histogram(normal_scores, bins=bin_edges)
    hist_anomaly, _ = np.histogram(anomaly_scores, bins=bin_edges)
    
    # Plot stacked histogram with different colors for normal and anomaly
    plt.bar(bin_centers, hist_normal, width=bin_width*0.9, color='skyblue', 
            label='Normal (Label=0)', alpha=0.7)
    plt.bar(bin_centers, hist_anomaly, width=bin_width*0.9, bottom=hist_normal, 
            color='crimson', label='Anomaly (Label=1)', alpha=0.7)
    
    # Fit Gaussian distribution to all scores
    try:
        # Initial guess for parameters: amplitude, mean, standard deviation
        p0 = [np.max(hist), np.mean(scores), np.std(scores)]
        
        # Fit Gaussian
        popt_gauss, _ = curve_fit(gaussian, bin_centers, hist, p0=p0, maxfev=10000)
        
        # Generate fitted curve
        x_fit = np.linspace(min_score, max_score, 1000)
        y_fit_gauss = gaussian(x_fit, *popt_gauss)
        
        # Plot Gaussian fit
        plt.plot(x_fit, y_fit_gauss, 'g-', linewidth=2, 
                 label=f'Gaussian Fit (μ={popt_gauss[1]:.2f}, σ={popt_gauss[2]:.2f})')
        
        print(f"Gaussian fit parameters: amplitude={popt_gauss[0]:.2f}, mean={popt_gauss[1]:.2f}, std={popt_gauss[2]:.2f}")
    except Exception as e:
        print(f"Error fitting Gaussian distribution: {e}")
    
    # Fit heavy-tailed distribution (Pareto) to all scores
    try:
        # Filter out negative scores for Pareto fitting
        positive_scores = scores[scores > 0]
        if len(positive_scores) > 0:
            hist_pos, bin_edges_pos = np.histogram(positive_scores, bins=bin_edges[bin_edges > 0])
            bin_centers_pos = (bin_edges_pos[:-1] + bin_edges_pos[1:]) / 2
            
            if len(bin_centers_pos) > 3:  # Need at least 3 points for fitting
                # Initial guess for parameters: amplitude, alpha, xm (minimum x)
                p0 = [np.max(hist_pos), 1.5, np.min(positive_scores)]
                
                # Fit Pareto
                popt_pareto, _ = curve_fit(pareto, bin_centers_pos, hist_pos, p0=p0, 
                                          bounds=([0, 0.1, 0], [np.inf, 10, np.min(positive_scores)*2]),
                                          maxfev=10000)
                
                # Generate fitted curve
                x_fit_pos = np.linspace(np.min(positive_scores), max_score, 1000)
                y_fit_pareto = pareto(x_fit_pos, *popt_pareto)
                
                # Plot Pareto fit
                plt.plot(x_fit_pos, y_fit_pareto, 'r-', linewidth=2, 
                         label=f'Pareto Fit (α={popt_pareto[1]:.2f}, xm={popt_pareto[2]:.2f})')
                
                print(f"Pareto fit parameters: amplitude={popt_pareto[0]:.2f}, alpha={popt_pareto[1]:.2f}, xm={popt_pareto[2]:.2f}")
    except Exception as e:
        print(f"Error fitting Pareto distribution: {e}")
    
    # Add labels and title
    plt.xlabel('Score', fontsize=14)
    plt.ylabel('Number of Packets', fontsize=14)
    plt.title('Score Distribution with Gaussian and Heavy-tailed Fitting\n'
              f'File: {os.path.basename(file_path)}', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Add text box with statistics
    stats_text = (
        f"Total: {len(scores)}\n"
        f"Normal: {len(normal_scores)} ({len(normal_scores)/len(scores)*100:.1f}%)\n"
        f"Anomaly: {len(anomaly_scores)} ({len(anomaly_scores)/len(scores)*100:.1f}%)\n"
        f"Mean: {np.mean(scores):.2f}\n"
        f"Median: {np.median(scores):.2f}\n"
        f"Std Dev: {np.std(scores):.2f}"
    )
    plt.annotate(stats_text, xy=(0.02, 0.97), xycoords='axes fraction',
                 bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.8),
                 va='top', fontsize=12)
    
    # Save figure if output directory is provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f"{os.path.basename(file_path).replace('.bin', '')}_distribution.png")
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {output_file}")
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Analyze the Friday/window_10_results.bin file
    file_path = '/home/sduu2/userspace-20T-1/yyr/HyperVision_results/injection/charrdos_injection/window_13_results.bin'
    output_dir = '/home/sduu2/userspace-20T-1/yyr/HyperVision_results/injection/charrdos_injection/visualizations'
    
    analyze_score_distribution(file_path, bin_width=3, output_dir=output_dir)