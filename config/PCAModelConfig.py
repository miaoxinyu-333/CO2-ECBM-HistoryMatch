import yaml

class PCAModelConfig:
    def __init__(self, config_path):
        # 加载 YAML 配置文件
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)

        # Task parameters
        task_params = config.get('task_params', {})
        self.task_id = task_params.get('task_id')
        self.task_name = task_params.get('task_name')
        self.data_set_path = task_params.get('data_set_path')
        self.input_field1 = task_params.get('input_field1')
        self.input_field2 = task_params.get('input_field2')
        self.target_field = task_params.get('target_field')
        self.logger_path = task_params.get('logger_path')
        self.logger_name = task_params.get('logger_name')
        self.save_model_path_per = task_params.get('save_model_path_per')
        self.save_model_path_por = task_params.get('save_model_path_por')
        self.path_to_save_images_per = task_params.get('path_to_save_images_per')
        self.path_to_save_images_por = task_params.get('path_to_save_images_por')
        self.path_to_save_figures_per = task_params.get('path_to_save_figures_per')
        self.path_to_save_figures_por = task_params.get('path_to_save_figures_por')
        self.path_to_save_figures_differences = task_params.get('path_to_save_figures_differences')

        # Model parameters
        model_params = config.get('model_params', {})
        self.latent_dim = model_params.get('latent_dim')

    def __repr__(self):
        return f"<PCA_Model_Config task_name={self.task_name}, latent_dim={self.latent_dim}>"
