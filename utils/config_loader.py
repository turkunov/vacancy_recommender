import json

def load_config(config_path: str) -> dict:
    """
    :params:
        config_path: str
        - путь к конфигу
    
    :returns:
        словарь с ключами "a_size", "a2idx", "skills2idx", "debug"
    """
    config_contents = None
    with open(config_path, 'r', encoding='utf-8') as f:
        config_contents = json.load(f)
    convert_to_int = ['a2idx', 'skills2idx']
    for key in convert_to_int:
        config_contents[key] = {int(idx): a for idx, a in config_contents[key].items()}
    config_contents['idx2skills'] = {a: idx for idx, a in config_contents['skills2idx'].items()}
    return config_contents