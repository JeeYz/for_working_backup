import sys
sys.path.append("../../../for_working_backup")


json_file = "/home/pncdl/DeepLearning/CWtraindata/PnC_Solution_CW_all_1102/$$whole_data_info.json"

from modules import json_files_processing as jsonfp


result = jsonfp.read_json_files(json_file)


for one in result:
    
    temp_filename = one['files'][0]

    elements_of_name = temp_filename.split('/')

    elements_of_name[-4] = 'PnC_Solution_CW_all_1102_16k'

    new_filename = "/".join(elements_of_name)

    one['files'][0] = new_filename



target_json_file = "/home/pncdl/DeepLearning/CWtraindata/PnC_Solution_CW_all_1102_16k/$$whole_data_info.json"


jsonfp.write_json_files(target_json_file, result)



