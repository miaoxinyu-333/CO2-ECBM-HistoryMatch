import matplotlib.pyplot as plt
import numpy as np
import torch
import os
from numpy.random import default_rng

def visualize_results(given_obs, generated_data):
    plt.figure(figsize=(12, 6))
    plt.plot(given_obs.cpu().numpy(), label='Given Observations', color='blue')
    plt.plot(generated_data.cpu().numpy(), label='Generated Data', color='red')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend()
    plt.title('Comparison of Given Observations and Generated Data')
    plt.show(block=False)

def compare_images(inputs_tensor, reconstructed_data, num_samples=5, cmap='viridis', show_colorbar=True, save_fig=False, save_path='./'):
    """
    从样本中抽取几对样本，并对比原始图像和重构图像的各个通道。
    """
    n_samples, n_channels, _, _ = inputs_tensor.shape
    indices = torch.randperm(n_samples)[:num_samples]
    
    for i, idx in enumerate(indices):
        for j in range(n_channels):
            # 显示并（可选）保存原始图像
            fig, ax = plt.subplots(figsize=(5, 5))
            im1 = ax.imshow(inputs_tensor[idx, j].cpu().numpy(), cmap=cmap)
            ax.axis('off')
            if show_colorbar:
                fig.colorbar(im1, ax=ax)
            
            if save_fig:
                plt.savefig(f"{save_path}/sample_{idx}_original_channel_{j}.png")
            plt.show(block=False)

            # 显示并（可选）保存重构图像
            fig, ax = plt.subplots(figsize=(5, 5))
            im2 = ax.imshow(reconstructed_data[idx, j].cpu().numpy(), cmap=cmap)
            ax.axis('off')
            if show_colorbar:
                fig.colorbar(im2, ax=ax)
            
            if save_fig:
                plt.savefig(f"{save_path}/sample_{idx}_reconstructed_channel_{j}.png")
            plt.show(block=False)

def analyze_pca_results(low_dim_data: np.ndarray, explained_variance_ratio: np.ndarray, save_path=None):
    """
    分析PCA结果并绘制累计解释方差图。
    如果提供了保存路径，则将图像保存为文件。
    
    参数:
    - low_dim_data: 降维后的数据
    - explained_variance_ratio: 各主成分的解释方差比
    - save_path: 图像保存的路径 (可选)
    """
    # 统计每个变量的变化范围
    min_vals = low_dim_data.min(axis=0)
    max_vals = low_dim_data.max(axis=0)
    ranges = max_vals - min_vals

    print("PCA变量的变化范围：")
    
    for i, (min_val, max_val, var_range) in enumerate(zip(min_vals, max_vals, ranges)):
        print(f"变量 {i+1}: 最小值 = {min_val}, 最大值 = {max_val}, 变化范围 = {var_range}")
    
    print(f"Total explained variance: {np.sum(explained_variance_ratio)}")

    # 计算累计解释方差
    cumulative_variance_ratio = np.cumsum(explained_variance_ratio)

    # 可视化累计解释方差
    plt.figure(figsize=(8, 6))
    plt.plot(cumulative_variance_ratio, marker='o')
    plt.xlabel('Number of Components', fontsize=14)  # 调整字体大小
    plt.ylabel('Cumulative Explained Variance', fontsize=14)  # 调整字体大小
    plt.title('Cumulative Explained Variance by PCA Components', fontsize=16)  # 调整字体大小
    plt.grid()

    # 设置坐标轴刻度字体大小
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    # 保存图像
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')  # 保存图像并自动调整边距
    
    plt.show(block=False)



def save_image(image, file_path, cmap='viridis', show_axes=False, show_colorbar=False):
    """Helper function to save an image."""
    plt.figure(figsize=(5, 5))
    img = plt.imshow(image, cmap=cmap, interpolation='nearest', origin='lower')
    if show_colorbar:
        plt.colorbar(img)
    if not show_axes:
        plt.axis('off')
    plt.savefig(file_path, bbox_inches='tight', pad_inches=0)
    plt.close()

def visualize_differences_autoencoder(model, dataset, num_samples=5, device='cpu', seed=42, path='output_images'):
    model.eval()
    model = model.to(device)

    # 确保输出目录存在
    os.makedirs(path, exist_ok=True)
    
    rng = default_rng(seed)
    indices = rng.choice(len(dataset), size=num_samples, replace=False)

    for sample_idx, index in enumerate(indices):
        sample = dataset[index]
        inputs, targets = sample[0].unsqueeze(0).to(device), sample[1].unsqueeze(0).to(device)  # shape: (1, 1, 2, 32, 32)

        # 获取模型预测
        with torch.no_grad():
            outputs = model(inputs)  # shape: (1, 1, 2, 32, 32)

        # 比较 outputs 和 targets 之间的差异，并分别处理每个通道
        for channel in range(outputs.shape[2]):  # 遍历每个通道
            input_img = inputs[0, 0, channel].cpu().numpy()  # Convert to shape (32, 32)
            output_img = outputs[0, 0, channel].cpu().numpy()  # Convert to shape (32, 32)
            target_img = targets[0, 0, channel].cpu().numpy()  # Convert to shape (32, 32)
            diff_img = np.abs(output_img - target_img)

            # 保存图像
            save_image(input_img, os.path.join(path, f'input_image_sample_{sample_idx}_channel_{channel}.png'))
            save_image(output_img, os.path.join(path, f'output_image_sample_{sample_idx}_channel_{channel}.png'))
            save_image(target_img, os.path.join(path, f'target_image_sample_{sample_idx}_channel_{channel}.png'))
            save_image(diff_img, os.path.join(path, f'difference_image_sample_{sample_idx}_channel_{channel}.png'))

    print(f"Saved input, output, target, and difference images to {path}")

def visualize_differences_surrogate(model, dataset, time_steps=12, device='cpu', seed=42, path='output_images', show_axes=False, show_colorbar=False):
    # 检查 dataset 是否有效
    if not dataset:
        raise ValueError("Dataset is empty or None. Please provide a valid dataset.")
    
    model.eval()
    model = model.to(device)

    # 确保输出目录存在
    os.makedirs(path, exist_ok=True)

    # 创建一个随机数生成器实例并随机选择一个样本
    rng = default_rng(seed)
    index = rng.integers(len(dataset))
    sample = dataset[index]
    inputs, targets = sample[0].unsqueeze(0).to(device), sample[1].unsqueeze(0).to(device)
    
    # 获取模型预测
    with torch.no_grad():
        outputs = model(inputs)
    
    # 遍历每个时间步绘制并保存差异图像
    for i in range(time_steps):
        output_img = outputs[0, i, 0].cpu().numpy()
        target_img = targets[0, i, 0].cpu().numpy()
        diff_img = np.abs(output_img - target_img)

        # 保存图像
        time_label = (i+1) * 30
        save_image(output_img, os.path.join(path, f'output_time_{time_label}.png'), show_axes=show_axes, show_colorbar=show_colorbar)
        save_image(target_img, os.path.join(path, f'target_time_{time_label}.png'), show_axes=show_axes, show_colorbar=show_colorbar)
        save_image(diff_img, os.path.join(path, f'difference_time_{time_label}.png'), show_axes=show_axes, show_colorbar=show_colorbar)
    
    print(f"Saved output, target, and difference images to {path}")

def set_plotting_params():
    """设置全局绘图参数."""
    plt.rcParams.update({
        'font.size': 14,  # 字体大小
        'axes.labelsize': 16,  # 坐标轴标签字体大小
        'axes.titlesize': 18,  # 标题字体大小
        'legend.fontsize': 14,  # 图例字体大小
        'xtick.labelsize': 14,  # x轴刻度字体大小
        'ytick.labelsize': 14,  # y轴刻度字体大小
        'figure.dpi': 300  # 图像分辨率
    })


def plot_parameter_variance(initial_variance, final_variance, save_dir):
    """Plot initial vs final parameter variance."""
    os.makedirs(save_dir, exist_ok=True)  # 确保目标目录存在

    plt.figure(figsize=(15, 5))  # 调整尺寸为适合双栏的比例
    plt.plot(initial_variance.flatten(), label='Initial Parameter Variance', linestyle='--')
    plt.plot(final_variance.flatten(), label='Final Parameter Variance', linestyle='-')
    plt.xlabel('Parameter Index')
    plt.ylabel('Variance')
    plt.title('Parameter Variance: Initial vs Final')
    plt.legend(loc='upper right')
    plt.grid(True)

    save_path = os.path.join(save_dir, 'parameter_variance_initial_final.png')
    plt.savefig(save_path, bbox_inches='tight')  # 保存图像，确保所有内容都在边框内
    plt.show(block=False)


def plot_errors_over_iterations(mse_list, mae_list, num_iterations, save_dir):
    """Plot MSE and MAE over iterations."""
    os.makedirs(save_dir, exist_ok=True)  # 确保目标目录存在

    iterations = list(range(num_iterations))
    plt.figure(figsize=(15, 6))  # 调整尺寸为适合双栏的比例

    plt.subplot(1, 2, 1)
    plt.plot(iterations, mse_list, label='MSE', marker='o')
    plt.xlabel('Iteration')
    plt.ylabel('MSE')
    plt.title('Mean Squared Error Over Iterations')
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(iterations, mae_list, label='MAE', marker='o', color='orange')
    plt.xlabel('Iteration')
    plt.ylabel('MAE')
    plt.title('Mean Absolute Error Over Iterations')
    plt.grid(True)

    plt.tight_layout()

    save_path = os.path.join(save_dir, 'mse_mae_over_iterations.png')
    plt.savefig(save_path, bbox_inches='tight')  # 保存图像，确保所有内容都在边框内
    plt.show(block=False)


def plot_assimilation_process(all_outputs, given_observation, save_dir):
    """Plot the assimilation process across iterations."""
    os.makedirs(save_dir, exist_ok=True)  # 确保目标目录存在

    plt.figure(figsize=(15, 10))  # 调整尺寸为适合双栏的比例
    for i in range(all_outputs.shape[0]):
        plt.plot(all_outputs[i], label=f'Iteration {i+1}', linestyle='--', alpha=0.6)

    plt.plot(given_observation, label='Given Observations', linestyle='-', color='blue', linewidth=2)
    plt.plot(all_outputs[-1], label='Final Assimilated Prediction', linestyle='-', color='red', linewidth=2)
    plt.xlabel('Month')
    plt.ylabel('CO₂ Concentration')  # 修改为 CO₂ 浓度
    plt.title('Assimilation Process: Convergence of Predictions to Observations')

    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()

    save_path = os.path.join(save_dir, 'assimilation_process_plot_every_iteration.png')
    plt.savefig(save_path, bbox_inches='tight')  # 保存图像，确保所有内容都在边框内
    plt.close()  # 关闭图像，节省内存


# 函数1：绘制渗透率场
def plot_permeability_field(df_perm, seed, save_dir='plots'):
    permeability = df_perm['Permeability'].values
    plt.figure(figsize=(10, 8))
    plt.tricontourf(df_perm['X'], df_perm['Y'], permeability, levels=50, cmap='viridis')
    plt.colorbar(label='Permeability (mD)')
    plt.title('Permeability Field')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.axis('equal')
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f'permeability_field_seed_{seed}.png'))
    plt.close()  # 关闭图像以节省内存

# 函数2：绘制孔隙度场
def plot_porosity_field(df_poro, seed, save_dir='plots'):
    porosity = df_poro['Porosity'].values
    plt.figure(figsize=(10, 8))
    plt.tricontourf(df_poro['X'], df_poro['Y'], porosity, levels=50, cmap='plasma')
    plt.colorbar(label='Porosity (%)')
    plt.title('Porosity Field')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.axis('equal')
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f'porosity_field_seed_{seed}.png'))
    plt.close()  # 关闭图像以节省内存

# 函数3：绘制渗透率频率分布
def plot_permeability_hist(permeability, seed, save_dir='plots'):
    plt.figure(figsize=(10, 6))
    plt.hist(permeability, bins=50, color='blue', alpha=0.7)
    plt.title('Frequency Distribution of Permeability')
    plt.xlabel('Permeability (mD)')
    plt.ylabel('Frequency')
    plt.grid(True)
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f'permeability_hist_seed_{seed}.png'))
    plt.close()  # 关闭图像以节省内存

# 函数4：绘制孔隙度频率分布
def plot_porosity_hist(porosity, seed, save_dir='plots'):
    plt.figure(figsize=(10, 6))
    plt.hist(porosity, bins=50, color='orange', alpha=0.7)
    plt.title('Frequency Distribution of Porosity')
    plt.xlabel('Porosity (%)')
    plt.ylabel('Frequency')
    plt.grid(True)
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f'porosity_hist_seed_{seed}.png'))
    plt.close()  # 关闭图像以节省内存