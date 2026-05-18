import argparse
import yaml
import os

from dotenv import load_dotenv
from utils.argument_utils import ArgumentUtils


class ProjectConfig:
    """
    统一处理整个项目的配置，整个项目的配置对象设置为单例
    """
    _instance = None  # 当前这个类的实例

    def __init__(self):
        self._config = None
        self._args = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(ProjectConfig, cls).__new__(cls)
        return cls._instance

    def initialize(self):
        """
        初始化所有的项目配置
        :return:
        """

        # 环境变量配置的初始化 - 从 .env 文件加载
        load_dotenv()

        # 命令行参数配置的初始化
        if self._args is None:
            arg_utils = ArgumentUtils()
            self._args = arg_utils.parse_arg()

        # YAML 文件配置的初始化 ： 如果YAML中的配置和命令行参数冲突，以命令参数为准
        if self._config is None:
            with open(self._args.config, 'r') as f:
                config = yaml.safe_load(f)

            # 只覆盖用户明确传入的命令行参数（排除默认值）
            overridden_config = {}
            args_dict = vars(self._args)
            
            # 获取命令行参数的默认值（从ArgumentParser对象获取）
            defaults = {}
            for action in arg_utils.parser._actions:
                if hasattr(action, 'default') and action.default is not argparse.SUPPRESS:
                    defaults[action.dest] = action.default
            
            # 只覆盖非默认值的命令行参数
            for key, value in args_dict.items():
                if key in config and value is not None:
                    # 检查是否是默认值，如果是默认值则不覆盖配置文件
                    if key in defaults and value == defaults[key]:
                        continue
                    overridden_config[key] = value

            config.update(overridden_config)  # 把命令的参数覆盖config文件里面的

            # 从环境变量读取 api_key，如果存在则覆盖配置
            env_api_key = os.getenv('API_KEY')
            if env_api_key:
                config['api_key'] = env_api_key

            self._config = config

    def __getattr__(self, item):
        # 当访问当前对象实例属性时自动调用
        if self._config and item in self._config:
            return self._config[item]
        raise AttributeError(f'项目配置中没有一个属性：{item}')


if __name__ == '__main__':
    o1 = ProjectConfig()
    o1.initialize()
    print(o1.model_name)
    print(o1.model)
