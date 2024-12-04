import os
import torch
import sys
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from scipy.stats import t, norm


def calculate_intervals(data, confidence=0.95):
    """
    计算置信区间、预测区间和概率误差范围
    :param data: 数据，形状为 (N, 12)
    :param confidence: 置信水平 (默认 95%)
    :return: 包含置信区间、预测区间和误差范围的结果
    """
    n = data.shape[0]
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    stderr = std / np.sqrt(n)

    # 置信区间 (confidence interval)
    z = norm.ppf((1 + confidence) / 2)  # 正态分布分位数
    ci_lower = mean - z * stderr
    ci_upper = mean + z * stderr

    # 预测区间 (prediction interval)
    t_val = t.ppf((1 + confidence) / 2, df=n - 1)  # t 分布分位数
    pred_interval_lower = mean - t_val * std * np.sqrt(1 + 1 / n)
    pred_interval_upper = mean + t_val * std * np.sqrt(1 + 1 / n)

    # 概率误差范围 (probabilistic error bounds)
    prob_range_lower = mean - z * std
    prob_range_upper = mean + z * std

    return {
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "pred_interval_lower": pred_interval_lower,
        "pred_interval_upper": pred_interval_upper,
        "prob_range_lower": prob_range_lower,
        "prob_range_upper": prob_range_upper,
    }

def extract_point_from_predictions(predictions, point_index):
    """
    从预测张量中提取指定点的值
    :param predictions: 模型的预测输出 (N, 12, 1, 64, 64)
    :param point_index: 感兴趣点的下标 (x, y)
    :return: 提取后的值，形状为 (N, 12)
    """
    x, y = point_index
    return predictions[..., x, y]  # 提取感兴趣点的值

def setup_project_root():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    # 将项目根目录添加到 sys.path
    sys.path.append(project_root)

def main():
    # 动态设置项目根目录
    setup_project_root()
    from omegaconf import OmegaConf
    import torch
    from models.SurrogateModel import SurrogateModel
    from utils.dataUtils import load_co2_h5_file, load_per_por_from_h5

    # 配置
    per_por_file_path = "dataSet/surrogate/dataset_per_por_4983.h5"
    co2_file_path = "dataSet/surrogate/dataset_co2_4983.h5"

    # 加载输入数据
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 检查是否有 GPU

    per_tensor, por_tensor = load_per_por_from_h5(per_por_file_path)
    per_tensor = per_tensor.unsqueeze(1).unsqueeze(2).to(device)
    por_tensor = por_tensor.unsqueeze(1).unsqueeze(2).to(device)
    per_tensor = torch.log(per_tensor + 1e-8)
    per_min, per_max = per_tensor.min(), per_tensor.max()
    per_tensor = (per_tensor - per_min) / (per_max - per_min)  # 归一化到 [0, 1]
    por_min, por_max = por_tensor.min(), por_tensor.max()
    por_tensor = (por_tensor - por_min) / (por_max - por_min)  # 归一化到 [0, 1]
    inputs_tensor = torch.cat((per_tensor, por_tensor), dim=2)

    # 加载目标数据
    co2_tensor = load_co2_h5_file(co2_file_path).to(device)
    target_tensor = co2_tensor.unsqueeze(2)

    # 创建数据集
    dataset = TensorDataset(inputs_tensor, target_tensor)
    data_loader = DataLoader(dataset, batch_size=64, shuffle=False)

    config_path = os.path.join("config", "SurrogateModel.yaml")
    task_name = 'task1'
    cfg = OmegaConf.load(config_path)

    cfg = cfg['tasks'][task_name]
    cfg_surrogate = cfg

    # Load the surrogate model structure
    model = SurrogateModel(config=cfg_surrogate).to(device)  # 将模型移到 GPU

    # Load the surrogate model weights
    checkpoint_path = cfg_surrogate.task_params.save_model_path
    model.load_model_weights(checkpoint_path)
    model.eval()  # 设置模型为评估模式

    # 存储所有预测结果
    all_predictions = []

    with torch.no_grad():  # 不需要梯度计算
        for inputs, _ in data_loader:
            inputs = inputs.to(device)  # 将输入移到 GPU
            predictions = model(inputs)  # 前向传播
            all_predictions.append(predictions.cpu().numpy())  # 将结果移到 CPU 以便后续处理

    # 合并所有预测为一个大张量
    all_predictions = np.concatenate(all_predictions, axis=0)

    print(all_predictions.shape)

    point_index = (3, 3)  # 感兴趣点的下标 (x, y)
    extracted_point_values = extract_point_from_predictions(all_predictions, point_index)

    # 统计分析
    mean_prediction = np.mean(extracted_point_values, axis=0)
    std_prediction = np.std(extracted_point_values, axis=0)
    min_prediction = np.min(extracted_point_values, axis=0)
    max_prediction = np.max(extracted_point_values, axis=0)

    # 打印统计结果
    print(f"Point Index: {point_index}")
    print(f"Mean Prediction:\n {mean_prediction}")
    print(f"Standard Deviation:\n {std_prediction}")
    print(f"Minimum Prediction:\n {min_prediction}")
    print(f"Maximum Prediction:\n {max_prediction}")

    # 可选：保存统计结果
    save_dir = "uncertainty_analysis_results"
    os.makedirs(save_dir, exist_ok=True)

    np.save(os.path.join(save_dir, f"mean_prediction_point_{point_index}.npy"), mean_prediction)
    np.save(os.path.join(save_dir, f"std_prediction_point_{point_index}.npy"), std_prediction)
    np.save(os.path.join(save_dir, f"min_prediction_point_{point_index}.npy"), min_prediction)
    np.save(os.path.join(save_dir, f"max_prediction_point_{point_index}.npy"), max_prediction)

    # 统计分析
    intervals = calculate_intervals(extracted_point_values, confidence=0.95)

    # 打印统计结果
    print(f"Point Index: {point_index}")
    print(f"Confidence Interval:\n Lower: {intervals['ci_lower']}\n Upper: {intervals['ci_upper']}")
    print(f"Prediction Interval:\n Lower: {intervals['pred_interval_lower']}\n Upper: {intervals['pred_interval_upper']}")
    print(f"Probabilistic Error Bounds:\n Lower: {intervals['prob_range_lower']}\n Upper: {intervals['prob_range_upper']}")

    # 可选：保存统计结果
    np.save(os.path.join(save_dir, f"confidence_interval_{point_index}.npy"), {
        "lower": intervals['ci_lower'],
        "upper": intervals['ci_upper']
    })
    np.save(os.path.join(save_dir, f"prediction_interval_{point_index}.npy"), {
        "lower": intervals['pred_interval_lower'],
        "upper": intervals['pred_interval_upper']
    })
    np.save(os.path.join(save_dir, f"prob_error_bounds_{point_index}.npy"), {
        "lower": intervals['prob_range_lower'],
        "upper": intervals['prob_range_upper']
    })

if __name__ == "__main__":
    main()

