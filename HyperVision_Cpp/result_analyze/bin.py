import os
import glob
import struct
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix, accuracy_score,f1_score,recall_score,precision_score

# -----------------------------------------------------------
# 1. 解析二进制文件
# -----------------------------------------------------------
def read_bin_file(file_path):
    with open(file_path, 'rb') as f:
        # Read the number of entries (4 bytes)
        num_entries = struct.unpack('I', f.read(4))[0]
        
        # Read labels and scores
        labels = []
        scores = []
        nan_count = 0  # Counter for NaN/infinite values
        total_count = 0  # Counter for total entries processed
        
        for _ in range(num_entries-2):
            try:
                # Read label (1 byte)
                label_data = f.read(1)
                if not label_data:
                    break
                
                label = struct.unpack('B', label_data)[0]
                
                # Read score (8 bytes double)
                score_data = f.read(8)
                if len(score_data) < 8:
                    break
                
                score = struct.unpack('d', score_data)[0]
                total_count += 1
                
                # Skip invalid scores (NaN, infinity)
                if not np.isfinite(score):
                    nan_count += 1
                    continue
                    
                labels.append(1 if label > 0 else 0)
                scores.append(score)
                
                # Skip padding byte if present
                if f.tell() < os.path.getsize(file_path):
                    f.read(1)
            except struct.error:
                break
    
    # Final validation
    if len(labels) != len(scores) or len(labels) == 0:
        return np.array([]), np.array([]), (total_count, nan_count)
        
    return np.array(labels), np.array(scores), (total_count, nan_count)

# -----------------------------------------------------------
# 2. 处理单个窗口文件
# -----------------------------------------------------------
def process_window(file_path, threshold=11):
    # 读取数据
    true_labels, scores, _ = read_bin_file(file_path)  # Ignore the statistics here as they're handled in analyze_results
    
    # Check if we have enough valid data
    if len(true_labels) < 2 or len(scores) < 2:
        print(f"Warning: Not enough valid data in {file_path}")
        # Return dummy values to avoid errors
        return np.array([0, 1]), np.array([0, 1]), 0.5, 0.5, 0.5, 0.5, 0.5, (0, 0, 0, 0)
    
    # Normalize scores to avoid extreme values
    scores_normalized = np.clip(scores, -1e10, 1e10)  # Clip extreme values
    
    # 生成预测标签
    pred_labels = (scores_normalized > threshold).astype(int)
    
    # 计算指标
    fpr, tpr, _ = roc_curve(true_labels, scores_normalized)
    roc_auc = auc(fpr, tpr)
    acc = accuracy_score(true_labels, pred_labels)
    tn, fp, fn, tp = confusion_matrix(true_labels, pred_labels).ravel()
    f1 = f1_score(true_labels, pred_labels)
    recall = recall_score(true_labels, pred_labels)
    precision = precision_score(true_labels, pred_labels)
    
    return fpr, tpr, roc_auc, acc, f1, recall, precision, (tp, fp, tn, fn)

# -----------------------------------------------------------
# 3. 绘制结果
# -----------------------------------------------------------
def plot_roc_curve(fpr, tpr, roc_auc, save_path, threshold):
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {roc_auc:.2f}')
    plt.plot([0, 1], [0, 1], 'k--', lw=2)  # 对角线
    
    # 标记阈值点
    threshold_idx = np.argmin(np.abs((fpr**2 + (1 - tpr)**2)))  # 找到最近点
    plt.scatter(fpr[threshold_idx], tpr[threshold_idx], 
                c='red', label=f'Threshold={threshold}')
    
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.savefig(save_path)
    plt.close()

def plot_score_dist(scores, true_labels, save_path):
    plt.figure(figsize=(10, 6))
    
    # Clip extreme values for visualization
    scores_clipped = np.clip(scores, np.percentile(scores, 1), np.percentile(scores, 99))
    
    # Use a limited number of points for plotting if there are too many
    max_points = 10000
    if len(scores_clipped) > max_points:
        indices = np.random.choice(len(scores_clipped), max_points, replace=False)
        scores_plot = scores_clipped[indices]
        labels_plot = true_labels[indices]
    else:
        scores_plot = scores_clipped
        labels_plot = true_labels
    
    plt.scatter(range(len(scores_plot)), scores_plot, c=labels_plot, cmap='viridis', alpha=0.7, s=5)
    plt.xlabel('Index')
    plt.ylabel('Score (clipped to 1-99 percentile)')
    plt.title('Score Distribution')
    plt.colorbar().set_label('True Label')
    
    try:
        plt.savefig(save_path)
    except Exception as e:
        print(f"Warning: Failed to save score distribution plot: {e}")
    
    plt.close()

# -----------------------------------------------------------
# 4. 主处理流程
# -----------------------------------------------------------
def analyze_results(bin_dir, output_dir):
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # 处理单个文件
        bin_file = bin_dir  # 直接使用输入的文件路径
        window_id = os.path.basename(bin_file).split('_')[-1].split('.')[0]  # 提取window_id
        
        # Read data first to check if it's valid
        true_labels, scores, (total_count, nan_count) = read_bin_file(bin_file)
        
        print(f"\nProcessing {bin_file}:")
        print(f"Total entries processed: {total_count}")
        print(f"Entries removed due to NaN/infinite values: {nan_count} ({(nan_count/total_count*100):.2f}% of total)")
        print(f"Valid entries remaining: {len(scores)} ({(len(scores)/total_count*100):.2f}% of total)")
        
        if len(true_labels) < 2 or len(scores) < 2:
            print(f"Warning: Not enough valid data in {bin_file}, skipping analysis")
            return
            
        # Process data for metrics
        fpr, tpr, roc_auc, acc, f1, recall, precision, (tp, fp, tn, fn) = process_window(bin_file, threshold=11)
        
        # 生成输出路径
        plot_path = os.path.join(output_dir, f"window_{window_id}_roc.png")
        
        # 绘制图像
        try:
            plot_roc_curve(fpr, tpr, roc_auc, plot_path, threshold=11)
        except Exception as e:
            print(f"Warning: Failed to create ROC curve: {e}")
        
        # 绘制分数分布图
        score_path = os.path.join(output_dir, f"window_{window_id}_score.png")
        try:
            plot_score_dist(scores, true_labels, score_path)
        except Exception as e:
            print(f"Warning: Failed to create score distribution plot: {e}")
            
        # 保存报告
        with open(os.path.join(output_dir, f"window_{window_id}_report.txt"), 'w') as f:
            f.write(f"\nWindow {window_id} Report:\n")
            f.write(f"- Data Quality:\n")
            f.write(f"  Total entries: {total_count}\n")
            f.write(f"  Invalid entries (NaN/inf): {nan_count} ({(nan_count/total_count*100):.2f}%)\n")
            f.write(f"  Valid entries: {len(scores)} ({(len(scores)/total_count*100):.2f}%)\n\n")
            f.write(f"- Performance Metrics:\n")
            f.write(f"  AUC: {roc_auc:.4f}\n")
            f.write(f"  Accuracy: {acc:.4f}\n")
            f.write(f"  F1 Score: {f1:.4f}\n")
            f.write(f"  Recall: {recall:.4f}\n")
            f.write(f"  Precision: {precision:.4f}\n")
            f.write(f"- Confusion Matrix:\n")
            f.write(f"  TP: {tp}  FP: {fp}\n")
            f.write(f"  FN: {fn}  TN: {tn}\n")
            f.write(f"- Data Stats:\n")
            f.write(f"  Total valid samples: {len(true_labels)}\n")
            f.write(f"  Positive samples: {np.sum(true_labels)}\n")
            f.write(f"  Negative samples: {len(true_labels) - np.sum(true_labels)}\n")
            f.write(f"  Score range: [{np.min(scores):.2f}, {np.max(scores):.2f}]\n")
    except Exception as e:
        print(f"Error processing {bin_dir}: {e}")

# -----------------------------------------------------------
# 执行主程序
# -----------------------------------------------------------
if __name__ == "__main__":
    # base_dir = "/home/sduu2/userspace-20T-1/yyr/HyperVision_results/delaying/charrdos_delaying"
    # output_base = "/home/sduu2/userspace-20T-1/yyr/HyperVision_results/delaying/charrdos_delaying/visualizations"
    analyze_results(bin_dir="/home/sduu2/userspace-20T-1/yyr/HyperVision_Cpp/results/multi_dataset_incremental/dataset_0/window_15_results.bin", output_dir="/home/sduu2/userspace-20T-1/yyr/HyperVision_Cpp/results/multi_dataset_incremental/dataset_0/visualizations")
    # for bin_file in glob.glob(os.path.join(base_dir, "dataset_1_window_*.bin")):
    #     file_name = os.path.basename(bin_file)
    #     output_dir = os.path.join(output_base, os.path.splitext(file_name)[0])
    #     analyze_results(bin_dir=bin_file, output_dir=output_dir)