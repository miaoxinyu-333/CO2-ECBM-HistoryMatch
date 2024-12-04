import h5py

def check_h5_shape(h5_filename):
    # 打开 HDF5 文件
    with h5py.File(h5_filename, 'r') as h5f:
        # 查看文件中所有的数据集名称
        print(f"Available datasets in {h5_filename}: {list(h5f.keys())}")
        
        # 获取 'data' 数据集
        if 'data' in h5f:
            data = h5f['data']  # 获取数据集
            print(f"The shape of 'data' dataset: {data.shape}")
        else:
            print("'data' dataset not found in the HDF5 file.")

# 示例：检查 'output_data.h5' 文件的 shape
h5_filename = 'CO2_data.h5'  # 请替换为你生成的 HDF5 文件名
check_h5_shape(h5_filename)
