import platform


mini = (platform.node() == 'p50')  # NOTE(rpradeep): True if on personal laptop

class DataModuleConfig:
    batch_size: int = 8 if mini else 32
    dev_trial: bool = True
    use_test_split: bool = False


class ModelConfig:
    use_manual_seed: bool = False
    use_centered_convolutions = False


def __print_config(config_class):
    print('')
    name = config_class.__name__
    print(f'{name}:')
    print('-' * (len(name) + 1))
    for k, v in vars(config_class).items():
        if k.startswith('__'): continue
        print(f'{k:>40}:    {v}')
    print('')

__print_config(DataModuleConfig)
__print_config(ModelConfig)
