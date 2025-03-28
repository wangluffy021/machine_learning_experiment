import numpy as np

def MSE(y_true, y_pred):
    """
    计算均方误差 (MSE)
        y_true: 真实标签
        y_pred: 预测结果
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return np.mean((y_true - y_pred) ** 2)

def MAE(y_true, y_pred):
    """
    计算平均绝对误差 (MAE)
        y_true: 真实标签
        y_pred: 预测结果
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return np.mean(np.abs(y_true - y_pred))

def RMSE(y_true, y_pred):
    """
    计算均方根误差 (RMSE)
        y_true: 真实标签
        y_pred: 预测结果
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return np.sqrt(MSE(y_true, y_pred))

def compare_models(y_true, predictions_dict):
    """
    比较多个模型在不同评估指标上的表现
        y_true: 真实标签
        predictions_dict: 字典，键为模型名称，值为该模型的预测结果
    """
    results = {}
    
    for model_name, y_pred in predictions_dict.items():
        mse = MSE(y_true, y_pred)
        mae = MAE(y_true, y_pred)
        rmse = RMSE(y_true, y_pred)
        
        results[model_name] = {
            'MSE': mse,
            'MAE': mae,
            'RMSE': rmse
        }
    
    return results

def print_comparison_results(results):
    """
    打印模型比较结果
    results: 字典，包含每个模型在各评估指标上的表现
    """
    print("模型评估结果比较:")
    print("-" * 60)
    print(f"{'模型名称':<15}{'MSE':<15}{'MAE':<15}{'RMSE':<15}")
    print("-" * 60)
    
    for model_name, metrics in results.items():
        print(f"{model_name:<15}{metrics['MSE']:<15.4f}{metrics['MAE']:<15.4f}{metrics['RMSE']:<15.4f}")

def load_data_from_csv(csv_file_path):
    """
    从CSV文件加载数据
    
    参数:
    csv_file_path: CSV文件路径
    
    返回:
    y_true: 真实标签
    predictions_dict: 字典，键为模型名称，值为该模型的预测结果
    """
    # 读取CSV文件
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

# 示例用法
if __name__ == "__main__":
    # 默认使用experiment_01_dataset_01.csv文件
    csv_file_path = "experiment_01_dataset_01.csv"
    
    print(f"正在从CSV文件加载数据: {csv_file_path}")
    
    # 从CSV文件加载数据
    y_true, predictions = load_data_from_csv(csv_file_path)
    
    if y_true is not None and predictions is not None:
        # 比较模型
        results = compare_models(y_true, predictions)
        
        # 打印比较结果
        print_comparison_results(results)
    