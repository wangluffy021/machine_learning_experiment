import numpy as np

def confusion_matrix(y_true, y_pred):
    """
    计算混淆矩阵
    
    参数:
    y_true: 真实标签
    y_pred: 预测结果
    
    返回:
    tp: 真阳性
    fp: 假阳性
    tn: 真阴性
    fn: 假阴性
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    
    return tp, fp, tn, fn

def calculate_metrics(tp, fp, tn, fn):
    """
    计算评估指标
    
    参数:
    tp: 真阳性
    fp: 假阳性
    tn: 真阴性
    fn: 假阴性
    
    返回:
    error_rate: 错误率
    precision:  查准率
    recall:     查全率
    f1:         F1值
    """
    total = tp + fp + tn + fn
    error_rate = (fp + fn) / total if total > 0 else 0
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return error_rate, precision, recall, f1

def evaluate_model(y_true, y_pred, model_name):
    """
    评估单个模型的性能
    
    参数:
    y_true:     真实标签
    y_pred:     预测结果
    model_name: 模型名称
    """
    tp, fp, tn, fn = confusion_matrix(y_true, y_pred)
    error_rate, precision, recall, f1 = calculate_metrics(tp, fp, tn, fn)
    
    print(f"\n{model_name}评估结果:")
    print("-" * 40)
    print("混淆矩阵:")
    
    # 使用表格形式输出混淆矩阵
    print("┌─────────────────┬─────────────────┬─────────────────┐")
    print("│                 │      正例       │      反例       │")
    print("├─────────────────┼─────────────────┼─────────────────┤")
    print(f"│      正例       │    TP: {tp:<7}  │    FN: {fn:<7}  │")
    print("├─────────────────┼─────────────────┼─────────────────┤")
    print(f"│      反例       │    FP: {fp:<7}  │    TN: {tn:<7}  │")
    print("└─────────────────┴─────────────────┴─────────────────┘")
    
    print("-" * 50)
    print(f"查准率: {precision:.4f}")
    print(f"查全率: {recall:.4f}")
    print(f"F1值: {f1:.4f}")

def load_data_from_csv(csv_file_path):
    """
    从CSV文件加载数据
    
    参数:
    csv_file_path: CSV文件路径
    
    返回:
    y_true:             真实标签
    predictions_dict:   字典，键为模型名称，值为该模型的预测结果
    """
    try:
        # 使用numpy读取CSV文件，跳过第一行（标题行）
        data = np.genfromtxt(csv_file_path, delimiter=',', skip_header=1)
        
        # 提取真实y值（第2列）
        y_true = data[:, 1]
        
        # 提取三种模型的预测值（第3-5列）
        model_names = [f"模型{i+1}" for i in range(3)]
        predictions_dict = {}
        
        for i, model_name in enumerate(model_names):
            predictions_dict[model_name] = data[:, i+2]
        
        return y_true, predictions_dict
    
    except Exception as e:
        print(f"加载CSV文件时出错: {e}")
        return None, None

# 主程序
if __name__ == "__main__":
    # 从CSV文件加载数据
    csv_file_path = "experiment_01_dataset_02.csv"
    print(f"正在从CSV文件加载数据: {csv_file_path}")
    
    y_true, predictions = load_data_from_csv(csv_file_path)
    
    if y_true is not None and predictions is not None:
        # 评估每个模型
        for model_name, y_pred in predictions.items():
            evaluate_model(y_true, y_pred, model_name)