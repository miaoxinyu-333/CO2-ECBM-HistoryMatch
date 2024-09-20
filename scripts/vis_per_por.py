import os
import sys

# 获取项目根目录
def setup_project_root():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    # 将项目根目录添加到 sys.path
    sys.path.append(project_root)

# 主函数调用
def main():
    from utils.plotUtils import plot_permeability_field
    from utils.plotUtils import plot_permeability_hist
    from utils.plotUtils import plot_porosity_field
    from utils.plotUtils import plot_porosity_hist
    from utils.plotUtils import set_plotting_params
    import pandas as pd
    # 调用全局绘图参数设置
    set_plotting_params()
    
    # 设置文件夹路径
    output_folder_perm = 'dataSet/permeability_datasets'
    output_folder_poro = 'dataSet/porosity_datasets'

    # 读取一个CSV文件作为示例（例如第一个seed的文件）
    seed = 4000
    csv_file_path_perm = os.path.join(output_folder_perm, f'{seed}.csv')
    csv_file_path_poro = os.path.join(output_folder_poro, f'{seed}.csv')

    # 读取渗透率和孔隙度数据
    df_perm = pd.read_csv(csv_file_path_perm)
    df_poro = pd.read_csv(csv_file_path_poro)

    # 提取渗透率和孔隙度数据
    permeability = df_perm['Permeability'].values
    porosity = df_poro['Porosity'].values

    # 调用绘图函数 
    plot_permeability_field(df_perm, seed)
    plot_porosity_field(df_poro, seed)
    plot_permeability_hist(permeability, seed)
    plot_porosity_hist(porosity, seed)

# 执行主函数
if __name__ == '__main__':
    main()