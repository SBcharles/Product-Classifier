import json


def read_json_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as json_file:
        data_dict = json.load(json_file)

    return data_dict
