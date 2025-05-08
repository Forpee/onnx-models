import onnx
from onnx import helper, TensorProto
import numpy as np

# Define quantization params
scale = 0.1
zero_point = 128

# Inputs
input1 = helper.make_tensor_value_info("input1", TensorProto.UINT8, [None])
input2 = helper.make_tensor_value_info("input2", TensorProto.UINT8, [None])

# Outputs
output_add = helper.make_tensor_value_info("output_add", TensorProto.UINT8, [None])
output_mul = helper.make_tensor_value_info("output_mul", TensorProto.UINT8, [None])

# Quantization scale and zero point as constants
scale_tensor = helper.make_tensor("scale", TensorProto.FLOAT, [], [scale])
zero_point_tensor = helper.make_tensor("zero_point", TensorProto.UINT8, [], [zero_point])

# QLinearAdd node
qlinear_add_node = helper.make_node(
    "QLinearAdd",
    inputs=[
        "input1", "scale", "zero_point",
        "input2", "scale", "zero_point",
        "scale", "zero_point"
    ],
    outputs=["output_add"]
)

# QLinearMul node
qlinear_mul_node = helper.make_node(
    "QLinearMul",
    inputs=[
        "input1", "scale", "zero_point",
        "input2", "scale", "zero_point",
        "scale", "zero_point"
    ],
    outputs=["output_mul"]
)

# Graph
graph_def = helper.make_graph(
    nodes=[qlinear_add_node, qlinear_mul_node],
    name="QuantizedAddMulGraph",
    inputs=[input1, input2],
    outputs=[output_add, output_mul],
    initializer=[scale_tensor, zero_point_tensor]
)

# Model
model_def = helper.make_model(graph_def, producer_name="quantized-add-mul")
onnx.save(model_def, "quantized_add_mul.onnx")