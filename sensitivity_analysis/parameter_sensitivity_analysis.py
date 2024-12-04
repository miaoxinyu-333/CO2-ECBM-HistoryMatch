import os
import sys

# 获取项目根目录
def setup_project_root():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    # 将项目根目录添加到 sys.path
    sys.path.append(project_root)
    return project_root


def main():
    # Setup
    project_root = setup_project_root()
    from utils.fileUtils import extract_tensors_from_folder
    from utils.fileUtils import extract_tensors_from_txt
    import numpy as np
    from collections import defaultdict

    # 加载基准张量
    base_path = os.path.join(project_root, "logs", "sensitivity_analysis", "baseline.txt")
    base_tensor = extract_tensors_from_txt(base_path)
    baseline_tensor = np.array(base_tensor)

    print(f"Baseline tensor shape: {baseline_tensor.shape}")

    # 加载所有敏感性分析张量文件（剔除 baseline 文件）
    folder_path = os.path.join(project_root, "logs", "sensitivity_analysis")
    all_files = [f for f in os.listdir(folder_path) if f.endswith('.txt') and f != "baseline.txt"]

    # 按文件加载张量
    tensors = []
    file_names = []
    for file_name in all_files:
        file_path = os.path.join(folder_path, file_name)
        tensor = extract_tensors_from_txt(file_path)
        tensors.append(np.array(tensor))
        file_names.append(file_name)

    print(f"Loaded {len(tensors)} tensors for sensitivity analysis.")
    print(f"Tensor shapes: {[t.shape for t in tensors]}")

    # 计算敏感性分数并分组加和
    sensitivity_scores = defaultdict(float)  # 用于存储每个参数的累积敏感性分数

    for tensor, file_name in zip(tensors, file_names):
        # 计算与基准张量的相对变化率
        delta = np.sum(np.abs(tensor - baseline_tensor)) / np.sum(np.abs(baseline_tensor))

        # 提取参数名称（假设文件名格式为 param_name_value.txt）
        param_name = file_name.split('_')[0]

        # 累加敏感性分数
        sensitivity_scores[param_name] += delta

    # 输出每个参数的敏感性总分
    print("\n每个参数的敏感性总分：")
    for param_name, total_score in sensitivity_scores.items():
        print(f"参数 {param_name} 的累积敏感性分数: {total_score}")

    # 根据总分排序（从高到低）
    sorted_scores = sorted(sensitivity_scores.items(), key=lambda x: x[1], reverse=True)
    print("\n参数敏感性排序（从高到低）：")
    for rank, (param_name, total_score) in enumerate(sorted_scores, start=1):
        print(f"优先级 {rank}: 参数 {param_name}, 累积敏感性分数: {total_score}")


if __name__ == "__main__":
    main()
