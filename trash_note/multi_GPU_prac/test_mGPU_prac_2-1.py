
#%%
import numpy as np
import tensorflow as tf


## global command
command_tflite_file = 'D:\\test_tflite_mGPU_prac_2-1.tflite'
# command_tflite_file = 'D:\\test_tflite_mGPU_prac_1.tflite'
command_interpreter = tf.lite.Interpreter(model_path=command_tflite_file)
command_interpreter.allocate_tensors()
command_input_details = command_interpreter.get_input_details()[0]
command_output_details = command_interpreter.get_output_details()[0]


temp_data = np.random.randn(64000)*100
# temp_data = np.random.randn(504, 127)*100
# temp_data = np.expand_dims(temp_data, axis=-1)
temp_data = np.expand_dims(temp_data, axis=0)
# test_data = np.array(temp_data, dtype=np.float32)
#test_data = np.array(test_data, dtype=np.float32)
#test_data = np.expand_dims(test_data, axis=-1)
# print(test_data.shape)
# print(test_data)

test_data = np.array(temp_data, dtype=np.float32)

command_interpreter.set_tensor(command_input_details['index'], test_data)
command_interpreter.invoke()
predictions = command_interpreter.get_tensor(command_output_details['index'])


print(predictions)

