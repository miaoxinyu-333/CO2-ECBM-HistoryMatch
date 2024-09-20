from sklearn.decomposition import PCA
import torch

class PCAModel:
    def __init__(self, n_components):
        """
        初始化PCA模型。

        Args:
            n_components (int): 降维后的维度数。
        """
        self.pca = PCA(n_components=n_components, random_state=12)

    def fit(self, data):
        """
        训练PCA模型。

        Args:
            data (np.ndarray): 训练数据，形状为 (样本数, 通道数, 高度, 宽度)。
        """
        data_reshaped = data.reshape(data.shape[0], -1)
        self.pca.fit(data_reshaped)

    def transform(self, data):
        """
        将数据降维到低维空间。

        Args:
            data (np.ndarray): 输入数据，形状为 (样本数, 通道数, 高度, 宽度)。

        Returns:
            np.ndarray: 降维后的数据。
        """
        data_reshaped = data.reshape(data.shape[0], -1)
        return self.pca.transform(data_reshaped)

    def inverse_transform(self, data, n_channels, height, width):
        """
        将降维后的数据重构回原始维度。

        Args:
            data (np.ndarray): 低维数据。
            n_channels (int): 原始数据的通道数。
            height (int): 原始数据的高度。
            width (int): 原始数据的宽度。

        Returns:
            tensor: 重构后的数据，形状为 (样本数, 通道数, 高度, 宽度)。
        """
        reconstructed_data = self.pca.inverse_transform(data)
        n_samples = data.shape[0]
        return torch.tensor(reconstructed_data.reshape(n_samples, n_channels, height, width))


    @property
    def explained_variance_ratio_(self):
        """
        返回解释方差比率。

        Returns:
            np.ndarray: 解释方差比率。
        """
        return self.pca.explained_variance_ratio_

    def forward(self, data):
        """
        将数据降维到低维空间。

        Args:
            data (np.ndarray): 输入数据。

        Returns:
            np.ndarray: 降维后的数据。
        """
        return self.transform(data)
