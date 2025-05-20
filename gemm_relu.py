import onnx
from onnx import helper, TensorProto, numpy_helper
import numpy as np

# Model input: shape [1, 4]
input_tensor = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 4])
output_tensor = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 4])

# Weights and biases for two Gemm layers
W1 = np.eye(4).astype(np.float32) * 0.5     # Shape [4, 4]
B1 = np.zeros(4, dtype=np.float32)          # Shape [4]
W2 = np.eye(4).astype(np.float32) * 1.5
B2 = np.ones(4, dtype=np.float32)

W1_tensor = numpy_helper.from_array(W1, name='W1')
B1_tensor = numpy_helper.from_array(B1, name='B1')
W2_tensor = numpy_helper.from_array(W2, name='W2')
B2_tensor = numpy_helper.from_array(B2, name='B2')

# Nodes
gemm1 = helper.make_node('Gemm', inputs=['input', 'W1', 'B1'], outputs=['gemm1_out'])
relu1 = helper.make_node('Relu', inputs=['gemm1_out'], outputs=['relu1_out'])

gemm2 = helper.make_node('Gemm', inputs=['relu1_out', 'W2', 'B2'], outputs=['gemm2_out'])
relu2 = helper.make_node('Relu', inputs=['gemm2_out'], outputs=['output'])

# Create the graph
graph = helper.make_graph(
    [gemm1, relu1, gemm2, relu2],
    'GemmReluTwice',
    [input_tensor],
    [output_tensor],
    initializer=[W1_tensor, B1_tensor, W2_tensor, B2_tensor]
)

# Create the model
model = helper.make_model(graph, producer_name='onnx-gemm-relu-chain')
onnx.checker.check_model(model)

# Save the model
onnx.save(model, 'gemm_relu.onnx')