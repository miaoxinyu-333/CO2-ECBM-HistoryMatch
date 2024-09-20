import sys
import os
import torch

# 获取项目根目录
def setup_project_root():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    # 将项目根目录添加到 sys.path
    sys.path.append(project_root)

def main():
    # 设置项目根目录
    setup_project_root()

    # 导入其他模块
    from config.PCAModelConfig import PCAModelConfig
    from models.PCAModel import PCAModel
    from utils.fileUtils import save_pca_model
    from utils.plotUtils import analyze_pca_results, compare_images
    from utils.metricsUtils import calculate_reconstruction_error
    from utils.fileUtils import save_images_to_hdf5
    from utils.dataUtils import load_datasets_from_h5

    # 加载配置文件
    config_path = os.path.join("config", "pca.yaml")
    config = PCAModelConfig(config_path)  # 使用你刚才创建的配置类

    # 加载数据
    input_field_per = config.input_field1
    directory_path_per = config.data_set_path
    data_tensor_per = load_datasets_from_h5(directory=directory_path_per)
    inputs_tensor_per = torch.tensor(data_tensor_per[input_field_per], dtype=torch.float32)

    input_field_por = config.input_field2
    directory_path_por = config.data_set_path
    data_tensor_por = load_datasets_from_h5(directory=directory_path_por)
    inputs_tensor_por = torch.tensor(data_tensor_por[input_field_por], dtype=torch.float32)

    print(f"Shape of inputs_tensor_per: {inputs_tensor_per.shape}")
    print(f"Shape of inputs_tensor_por: {inputs_tensor_por.shape}")

    # 预处理数据
    conversion_factor = 1.01325e15
    inputs_tensor_per = inputs_tensor_per * conversion_factor  # 将渗透率张量转换为 mD

    n_samples, _, height, width = inputs_tensor_per.shape  # 期望的形状: (4970, 1, 32, 32)

    # 将每个通道的数据展平成二维数组
    tensor_reshaped_per = inputs_tensor_per.view(n_samples, height * width).numpy()
    tensor_reshaped_por = inputs_tensor_por.view(n_samples, height * width).numpy()

    # 对每个通道单独进行 PCA
    latent_dim = config.latent_dim
    pca_model_per = PCAModel(n_components=latent_dim)
    pca_model_por = PCAModel(n_components=latent_dim)

    pca_model_per.fit(tensor_reshaped_per)
    pca_model_por.fit(tensor_reshaped_por)

    # 保存 PCA 模型
    save_pca_model(pca_model_per, config.save_model_path_per)
    save_pca_model(pca_model_por, config.save_model_path_por)

    # 低维表示和重构
    low_dim_data_per = pca_model_per.transform(tensor_reshaped_per)
    reconstructed_data_per = pca_model_per.inverse_transform(low_dim_data_per, 1, height, width)

    low_dim_data_por = pca_model_por.transform(tensor_reshaped_por)
    reconstructed_data_por = pca_model_por.inverse_transform(low_dim_data_por, 1, height, width)

    # 将原始数据和重构数据合并
    origin_data = torch.cat((inputs_tensor_per, inputs_tensor_por), dim=1)
    reconstructed_data = torch.cat((reconstructed_data_per, reconstructed_data_por), dim=1)

    # 分析 PCA 结果和重构误差
    analyze_pca_results(low_dim_data_per, pca_model_per.explained_variance_ratio_, save_path=config.path_to_save_figures_per)
    analyze_pca_results(low_dim_data_por, pca_model_por.explained_variance_ratio_, save_path=config.path_to_save_figures_por)
    reconstruction_error_per = calculate_reconstruction_error(inputs_tensor_per, reconstructed_data_per)
    reconstruction_error_por = calculate_reconstruction_error(inputs_tensor_por, reconstructed_data_por)
    print(f"Reconstruction error for permeability channel: {reconstruction_error_per}")
    print(f"Reconstruction error for porosity channel: {reconstruction_error_por}")

    # 比较原始图像和重构图像
    compare_images(origin_data, reconstructed_data, num_samples=1, cmap='viridis', show_colorbar=False, save_fig=True, save_path=config.path_to_save_figures_differences)

    # 保存重构后的图像到 HDF5 文件
    save_images_to_hdf5(inputs_tensor_per, reconstructed_data_per, file_path=config.path_to_save_images_per)
    save_images_to_hdf5(inputs_tensor_por, reconstructed_data_por, file_path=config.path_to_save_images_por)

if __name__ == "__main__":
    main()
