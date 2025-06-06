import onnx
from onnx import helper, TensorProto, numpy_helper
import numpy as np

# Define input/output tensor types
input_tensor = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 4])
output_tensor = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 4])

# Create weights and biases as initializers (constants in the model)
W1 = np.eye(4, dtype=np.float32) * 0.5
B1 = np.zeros((4,), dtype=np.float32)
W2 = np.eye(4, dtype=np.float32) * 1.5
B2 = np.ones((4,), dtype=np.float32)

W1_init = numpy_helper.from_array(W1, name='W1')
B1_init = numpy_helper.from_array(B1, name='B1')
W2_init = numpy_helper.from_array(W2, name='W2')
B2_init = numpy_helper.from_array(B2, name='B2')

# Define nodes
nodes = [
    helper.make_node('Gemm', ['input', 'W1', 'B1'], ['x1'], name='Gemm1'),
    helper.make_node('Relu', ['x1'], ['x2'], name='ReLU1'),
    helper.make_node('Gemm', ['x2', 'W2', 'B2'], ['x3'], name='Gemm2'),
    helper.make_node('Relu', ['x3'], ['output'], name='ReLU2'),
]

# Create the graph
graph_def = helper.make_graph(
    nodes=nodes,
    name='GemmReluTwice',
    inputs=[input_tensor],
    outputs=[output_tensor],
    initializer=[W1_init, B1_init, W2_init, B2_init],
)

# Create the model
model_def = helper.make_model(graph_def, producer_name='onnx-gemm-relu-chain')
onnx.checker.check_model(model_def)

# Save it
onnx.save(model_def, 'gemm_relu.onnx')