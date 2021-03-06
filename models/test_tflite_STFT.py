
#%%
import numpy as np
import tensorflow as tf


## global command
command_tflite_file = 'D:\\test_tflite_STFT.tflite'
command_interpreter = tf.lite.Interpreter(model_path=command_tflite_file)
command_interpreter.allocate_tensors()
command_input_details = command_interpreter.get_input_details()[0]
command_output_details = command_interpreter.get_output_details()[0]


##
def standardization_func(data):
    return (data-np.mean(data))/np.std(data)


temp_data = np.random.randn(32000)*100

test_data = standardization_func(temp_data)
test_data = np.array([test_data], dtype=np.float32)
#test_data = np.array(test_data, dtype=np.float32)
#test_data = np.expand_dims(test_data, axis=-1)
print(test_data.shape)
print(test_data)

command_interpreter.set_tensor(command_input_details['index'], test_data)
command_interpreter.invoke()
predictions = command_interpreter.get_tensor(command_output_details['index'])


print(predictions)
# a = np.argmax(predictions)

# print('\n')
# print(predictions[0])

# print("label : {label}, \tprediction : {pred}".format(label=a, pred=predictions[0][a]))
# print("label : {label}, \tprediction : {pred}".format(label=a, pred='None'))










# %%
