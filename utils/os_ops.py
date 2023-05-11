import json

def json_read(filepath: str):
    with open(filepath,'r',encoding='utf-8') as f:
        return json.load(f)