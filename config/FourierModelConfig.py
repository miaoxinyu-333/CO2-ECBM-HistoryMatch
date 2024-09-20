import yaml

class Config:
    def __init__(self, config):
        # 初始化各类参数对象
        self.task_params = TaskParams(config['task_params'])
        self.model_params = ModelParams(config['model_params'])
        self.optimizer_params = OptimizerParams(config['optimizer_params'])
        self.scheduler_params = SchedulerParams(config['scheduler_params'])
        self.training_params = TrainingParams(config['training_params'])  # 添加训练参数
        self.loss_function = config.get('loss_function', 'MSELoss')

    def to_dict(self):
        # 返回嵌套字典结构
        return {
            'task_params': self.task_params.__dict__,
            'model_params': self.model_params.__dict__,
            'optimizer_params': self.optimizer_params.__dict__,
            'scheduler_params': self.scheduler_params.__dict__,
            'training_params': self.training_params.__dict__,  # 返回训练参数
            'loss_function': self.loss_function
        }

class TrainingParams:
    def __init__(self, training_params):
        self.batch_size = training_params.get('batch_size', 32)  # 设置默认值
        self.max_epochs = training_params.get('max_epochs', 50)

# 子类同样使用字典转换，或使用 __dict__
class TaskParams:
    def __init__(self, task_params):
        self.task_id = task_params.get('task_id')
        self.task_name = task_params.get('task_name')
        self.data_set_path = task_params.get('data_set_path', "")
        self.input_field1 = task_params.get('input_field1', "")
        self.input_field2 = task_params.get('input_field2', "")
        self.target_field = task_params.get('target_field', "")
        self.logger_path = task_params.get('logger_path', "")
        self.logger_name = task_params.get('logger_name', "")
        self.save_model_path = task_params.get('save_model_path', "")

class ModelParams:
    def __init__(self, model_params):
        self.name = model_params.get('name', '')
        self.n_input_scalar_components = model_params.get('n_input_scalar_components', 0)
        self.n_input_vector_components = model_params.get('n_input_vector_components', 0)
        self.n_output_scalar_components = model_params.get('n_output_scalar_components', 0)
        self.n_output_vector_components = model_params.get('n_output_vector_components', 0)
        self.time_history = model_params.get('time_history', 0)
        self.time_future = model_params.get('time_future', 0)
        self.hidden_channels = model_params.get('hidden_channels', 0)
        self.activation = model_params.get('activation', 'gelu')
        self.modes1 = model_params.get('modes1', 16)
        self.modes2 = model_params.get('modes2', 16)
        self.norm = model_params.get('norm', True)
        self.ch_mults = model_params.get('ch_mults', [1, 2, 2, 2])
        self.is_attn = model_params.get('is_attn', [False, True, True, False])
        self.mid_attn = model_params.get('mid_attn', True)
        self.n_blocks = model_params.get('n_blocks', 2)
        self.n_fourier_layers = model_params.get('n_fourier_layers', 2)
        self.mode_scaling = model_params.get('mode_scaling', True)
        self.use1x1 = model_params.get('use1x1', False)

class OptimizerParams:
    def __init__(self, optimizer_params):
        self.lr = optimizer_params.get('lr', 1e-4)
        self.weight_decay = optimizer_params.get('weight_decay', 0)

class SchedulerParams:
    def __init__(self, scheduler_params):
        self.warmup_epochs = scheduler_params.get('warmup_epochs', 0)
        self.max_epochs = scheduler_params.get('max_epochs', 0)
        self.eta_min = scheduler_params.get('eta_min', 1e-7)

def load_config(file_path, task_name):
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
        return Config(config['tasks'][task_name])