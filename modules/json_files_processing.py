import json


def read_json_files(json_file):
    with open(json_file, 'r', encoding='utf-8') as json_readf:
        loaded_data = json.load(json_readf)

    return loaded_data


def write_json_files(json_file_path, input_data_callback):

    target_data = input_data_callback()

    with open(json_file_path, 'w', encoding='utf-8') as fp_write_json:
        json.dump(
            target_data,
            fp_write_json, 
            indent='\t', 
            ensure_ascii=False
        )



if __name__ == '__main__':
    print("hello, world~!!")





