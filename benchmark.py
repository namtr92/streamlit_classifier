import onnxruntime
import numpy as np
import time
import os

model_name = 'deit_tiny'

model_path = os.path.join(os.path.dirname(__file__), 'models', f'{model_name}_pad_int8.onnx')

sess_options = onnxruntime.SessionOptions()
sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL
ort_session = onnxruntime.InferenceSession(
    model_path,
    sess_options,
    providers=['TensorrtExecutionProvider'],
    provider_options=[{'device_id': '0',
                       'trt_int8_enable': True,
                       'trt_engine_cache_enable': True
                      }])
input_name = ort_session.get_inputs()[0].name
output_name = ort_session.get_outputs()[0].name
#benchmark
input_data = np.random.random_sample((4, 3, 224, 224)).astype(np.float32)

for i in range(100):
    start = time.time()
    ort_session.run([output_name], {input_name: input_data})
    end = time.time()
    print(f'Inference time: {(end-start)*1000} ms')
#Inference time: 0.002 seconds 


