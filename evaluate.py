import os
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, precision_recall_curve
import pandas as pd

from data_loader import FraudGTDataLoader
from sequence_generator import SequenceGenerator
from models import AccountRiskPredictor
from utils import (
    create_data_loaders, evaluate_model, print_metrics, 
    SequenceSelector, set_random_seed
)

def load_trained_model(model_path, device='cpu'):
    """載入訓練好的模型"""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    checkpoint = torch.load(model_path, map_location=device)
    
    # 從checkpoint重建模型
    model_config = checkpoint['model_config']
    model = AccountRiskPredictor(**model_config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    
    print(f"Model loaded from {model_path}")
    print(f"Trained for {checkpoint['epoch']} epochs")
    print(f"Best validation F1: {checkpoint['val_f1']:.4f}")
    
    return model, checkpoint

def detailed_evaluation(model, data, loader, sequence_selector, sequences_data, device='cpu'):
    """詳細評估模型"""
    model.eval()
    model = model.to(device)
    
    all_predictions = []
    all_labels = []
    all_node_ids = []
    
    with torch.no_grad():
        for sub_edge_index, subset, batch_size in loader:
            sub_edge_index = sub_edge_index.to(device)
            
            # 獲取序列
            in_sequences, in_lengths = sequence_selector.select(
                subset, sequences_data['in_sequences'], sequences_data['in_lengths']
            )
            out_sequences, out_lengths = sequence_selector.select(
                subset, sequences_data['out_sequences'], sequences_data['out_lengths']
            )
            
            in_sequences = in_sequences.to(device)
            out_sequences = out_sequences.to(device)
            
            # 模型預測
            risk_scores, _ = model(in_sequences, out_sequences, in_lengths, out_lengths, sub_edge_index)
            risk_probs = torch.sigmoid(risk_scores)
            
            # 收集結果
            all_predictions.append(risk_probs[:batch_size].cpu())
            all_labels.append(data.labels[subset[:batch_size]].cpu())
            all_node_ids.append(subset[:batch_size].cpu())
    
    # 合併結果
    predictions = torch.cat(all_predictions, 0).numpy()
    labels = torch.cat(all_labels, 0).numpy()
    node_ids = torch.cat(all_node_ids, 0).numpy()
    
    return predictions, labels, node_ids

def plot_evaluation_results(predictions, labels, save_dir=None):
    """繪製評估結果圖表"""
    plt.style.use('default')
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # ROC曲線
    from sklearn.metrics import roc_curve, auc
    fpr, tpr, _ = roc_curve(labels, predictions)
    roc_auc = auc(fpr, tpr)
    
    axes[0, 0].plot(fpr, tpr, color='darkorange', lw=2, 
                    label=f'ROC curve (AUC = {roc_auc:.3f})')
    axes[0, 0].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    axes[0, 0].set_xlim([0.0, 1.0])
    axes[0, 0].set_ylim([0.0, 1.05])
    axes[0, 0].set_xlabel('False Positive Rate')
    axes[0, 0].set_ylabel('True Positive Rate')
    axes[0, 0].set_title('ROC Curve')
    axes[0, 0].legend(loc="lower right")
    axes[0, 0].grid(True)
    
    # Precision-Recall曲線
    from sklearn.metrics import precision_recall_curve, average_precision_score
    precision, recall, _ = precision_recall_curve(labels, predictions)
    avg_precision = average_precision_score(labels, predictions)
    
    axes[0, 1].plot(recall, precision, color='blue', lw=2,
                    label=f'PR curve (AP = {avg_precision:.3f})')
    axes[0, 1].set_xlim([0.0, 1.0])
    axes[0, 1].set_ylim([0.0, 1.05])
    axes[0, 1].set_xlabel('Recall')
    axes[0, 1].set_ylabel('Precision')
    axes[0, 1].set_title('Precision-Recall Curve')
    axes[0, 1].legend(loc="lower left")
    axes[0, 1].grid(True)
    
    # 預測分佈直方圖
    axes[1, 0].hist(predictions[labels == 0], bins=50, alpha=0.7, label='Normal', color='blue', density=True)
    axes[1, 0].hist(predictions[labels == 1], bins=50, alpha=0.7, label='Risky', color='red', density=True)
    axes[1, 0].set_xlabel('Risk Score')
    axes[1, 0].set_ylabel('Density')
    axes[1, 0].set_title('Risk Score Distribution')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # 混淆矩陣
    optimal_threshold = 0.5
    fpr_opt, tpr_opt, thresholds = roc_curve(labels, predictions)
    optimal_idx = np.argmax(tpr_opt - fpr_opt)
    optimal_threshold = thresholds[optimal_idx]
    
    pred_binary = (predictions > optimal_threshold).astype(int)
    cm = confusion_matrix(labels, pred_binary)
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, 1])
    axes[1, 1].set_xlabel('Predicted')
    axes[1, 1].set_ylabel('Actual')
    axes[1, 1].set_title(f'Confusion Matrix (threshold={optimal_threshold:.3f})')
    
    plt.tight_layout()
    
    if save_dir:
        plt.savefig(os.path.join(save_dir, 'evaluation_plots.png'), dpi=300, bbox_inches='tight')
        print(f"Evaluation plots saved to {save_dir}")
    
    plt.show()

def threshold_analysis(predictions, labels, save_dir=None):
    """閾值分析"""
    thresholds = np.linspace(0, 1, 101)
    precisions = []
    recalls = []
    f1_scores = []
    
    for threshold in thresholds:
        pred_binary = (predictions > threshold).astype(int)
        
        tp = np.sum((pred_binary == 1) & (labels == 1))
        fp = np.sum((pred_binary == 1) & (labels == 0))
        fn = np.sum((pred_binary == 0) & (labels == 1))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)
    
    # 找到最佳閾值
    best_f1_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_f1_idx]
    best_f1 = f1_scores[best_f1_idx]
    
    # 繪圖
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    plt.plot(thresholds, precisions, label='Precision', color='blue')
    plt.plot(thresholds, recalls, label='Recall', color='red')
    plt.plot(thresholds, f1_scores, label='F1-Score', color='green')
    plt.axvline(x=best_threshold, color='black', linestyle='--', alpha=0.7, label=f'Best threshold: {best_threshold:.3f}')
    plt.xlabel('Threshold')
    plt.ylabel('Score')
    plt.title('Metrics vs Threshold')
    plt.legend()
    plt.grid(True)
    
    # 創建結果DataFrame
    results_df = pd.DataFrame({
        'threshold': thresholds,
        'precision': precisions,
        'recall': recalls,
        'f1_score': f1_scores
    })
    
    if save_dir:
        results_df.to_csv(os.path.join(save_dir, 'threshold_analysis.csv'), index=False)
        plt.savefig(os.path.join(save_dir, 'threshold_analysis.png'), dpi=300, bbox_inches='tight')
    
    plt.tight_layout()
    plt.show()
    
    print(f"Best threshold: {best_threshold:.3f}")
    print(f"Best F1-score: {best_f1:.3f}")
    print(f"Precision at best threshold: {precisions[best_f1_idx]:.3f}")
    print(f"Recall at best threshold: {recalls[best_f1_idx]:.3f}")
    
    return best_threshold, results_df

def analyze_predictions_by_account(predictions, labels, node_ids, data, save_dir=None):
    """按帳戶類型分析預測結果"""
    # 創建結果DataFrame
    results_df = pd.DataFrame({
        'node_id': node_ids,
        'true_label': labels,
        'risk_score': predictions,
        'predicted_label': (predictions > 0.5).astype(int)
    })
    
    # 添加預測正確性
    results_df['correct'] = (results_df['true_label'] == results_df['predicted_label'])
    
    # 統計分析
    print("=== Prediction Analysis by Account ===")
    print(f"Total accounts: {len(results_df)}")
    print(f"Accuracy: {results_df['correct'].mean():.3f}")
    print(f"True positive accounts: {((results_df['true_label'] == 1) & (results_df['predicted_label'] == 1)).sum()}")
    print(f"False positive accounts: {((results_df['true_label'] == 0) & (results_df['predicted_label'] == 1)).sum()}")
    print(f"True negative accounts: {((results_df['true_label'] == 0) & (results_df['predicted_label'] == 0)).sum()}")
    print(f"False negative accounts: {((results_df['true_label'] == 1) & (results_df['predicted_label'] == 0)).sum()}")
    
    # 風險分數統計
    print(f"\nRisk Score Statistics:")
    print(f"Normal accounts - Mean: {results_df[results_df['true_label'] == 0]['risk_score'].mean():.3f}, Std: {results_df[results_df['true_label'] == 0]['risk_score'].std():.3f}")
    print(f"Risky accounts - Mean: {results_df[results_df['true_label'] == 1]['risk_score'].mean():.3f}, Std: {results_df[results_df['true_label'] == 1]['risk_score'].std():.3f}")
    
    # 保存詳細結果
    if save_dir:
        results_df.to_csv(os.path.join(save_dir, 'account_predictions.csv'), index=False)
        
        # 保存錯誤案例分析
        false_positives = results_df[(results_df['true_label'] == 0) & (results_df['predicted_label'] == 1)]
        false_negatives = results_df[(results_df['true_label'] == 1) & (results_df['predicted_label'] == 0)]
        
        false_positives.to_csv(os.path.join(save_dir, 'false_positives.csv'), index=False)
        false_negatives.to_csv(os.path.join(save_dir, 'false_negatives.csv'), index=False)
        
        print(f"Detailed results saved to {save_dir}")
    
    return results_df

def main(args):
    """主評估函數"""
    print("=== FraudGT Model Evaluation ===")
    
    # 設置設備
    device = torch.device(f'cuda:{args.gpu}' if args.gpu >= 0 and torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 設置隨機種子
    set_random_seed(args.random_seed)
    
    # 載入資料
    print("Loading data...")
    data_loader = FraudGTDataLoader(args.data_dir)
    data = data_loader.load_data('data.pt')
    
    sequence_generator = SequenceGenerator(args.data_dir, args.max_sequence_length)
    sequences_data = sequence_generator.load_sequences()
    training_sequences = sequence_generator.prepare_for_training(sequences_data)
    
    # 載入模型
    print("Loading trained model...")
    model, checkpoint = load_trained_model(args.model_path, device)
    
    # 創建資料載入器
    train_loader, val_loader, test_loader = create_data_loaders(
        data, training_sequences, args.batch_size
    )
    
    sequence_selector = SequenceSelector()
    
    # 創建結果目錄
    if args.save_results:
        results_dir = os.path.join(args.data_dir, 'evaluation_results')
        os.makedirs(results_dir, exist_ok=True)
    else:
        results_dir = None
    
    # 評估各個分割
    print("\n=== Evaluating on all splits ===")
    
    # 訓練集評估
    train_predictions, train_labels, train_node_ids = detailed_evaluation(
        model, data, train_loader, sequence_selector, training_sequences, device
    )
    train_metrics = evaluate_model(model, data, train_loader, sequence_selector, training_sequences, device)
    print_metrics(train_metrics, "Training Set")
    
    # 驗證集評估
    val_predictions, val_labels, val_node_ids = detailed_evaluation(
        model, data, val_loader, sequence_selector, training_sequences, device
    )
    val_metrics = evaluate_model(model, data, val_loader, sequence_selector, training_sequences, device)
    print_metrics(val_metrics, "Validation Set")
    
    # 測試集評估
    test_predictions, test_labels, test_node_ids = detailed_evaluation(
        model, data, test_loader, sequence_selector, training_sequences, device
    )
    test_metrics = evaluate_model(model, data, test_loader, sequence_selector, training_sequences, device)
    print_metrics(test_metrics, "Test Set")
    
    # 詳細分析（使用測試集）
    print("\n=== Detailed Analysis on Test Set ===")
    
    # 繪製評估圖表
    if args.plot_results:
        plot_evaluation_results(test_predictions, test_labels, results_dir)
    
    # 閾值分析
    best_threshold, threshold_df = threshold_analysis(test_predictions, test_labels, results_dir)
    
    # 按帳戶分析
    account_results = analyze_predictions_by_account(
        test_predictions, test_labels, test_node_ids, data, results_dir
    )
    
    # 保存所有結果
    if args.save_results:
        all_results = {
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
            'test_metrics': test_metrics,
            'best_threshold': best_threshold,
            'threshold_analysis': threshold_df,
            'model_checkpoint': checkpoint,
            'test_predictions': test_predictions,
            'test_labels': test_labels,
            'test_node_ids': test_node_ids
        }
        
        results_path = os.path.join(results_dir, 'evaluation_results.pt')
        torch.save(all_results, results_path)
        print(f"All results saved to {results_path}")

def create_arg_parser():
    """創建參數解析器"""
    parser = argparse.ArgumentParser(description='Evaluate FraudGT Account Risk Prediction Model')
    
    parser.add_argument('--data_dir', type=str, default='./data/HI-Medium',
                       help='Data directory')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--max_sequence_length', type=int, default=32,
                       help='Maximum sequence length')
    parser.add_argument('--batch_size', type=int, default=128,
                       help='Batch size for evaluation')
    parser.add_argument('--gpu', type=int, default=0,
                       help='GPU device ID (-1 for CPU)')
    parser.add_argument('--random_seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--plot_results', action='store_true',
                       help='Generate evaluation plots')
    parser.add_argument('--save_results', action='store_true',
                       help='Save evaluation results')
    
    return parser

if __name__ == '__main__':
    parser = create_arg_parser()
    args = parser.parse_args()
    
    main(args)
