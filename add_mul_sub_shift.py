import onnx
from onnx import helper, TensorProto

# ----- Float32 inputs for arithmetic -----
float_input1 = helper.make_tensor_value_info("float_input1", TensorProto.FLOAT, [None])
float_input2 = helper.make_tensor_value_info("float_input2", TensorProto.FLOAT, [None])

# ----- Int32 inputs for bit shifts -----
int_input1 = helper.make_tensor_value_info("int_input1", TensorProto.INT32, [None])
int_input2 = helper.make_tensor_value_info("int_input2", TensorProto.INT32, [None])

# ----- Outputs -----
output_add = helper.make_tensor_value_info("output_add", TensorProto.FLOAT, [None])
output_mul = helper.make_tensor_value_info("output_mul", TensorProto.FLOAT, [None])
output_sub = helper.make_tensor_value_info("output_sub", TensorProto.FLOAT, [None])
output_shl = helper.make_tensor_value_info("output_shl", TensorProto.INT32, [None])
output_shr = helper.make_tensor_value_info("output_shr", TensorProto.INT32, [None])

# ----- Arithmetic Nodes -----
node_add = helper.make_node("Add", ["float_input1", "float_input2"], ["output_add"])
node_mul = helper.make_node("Mul", ["float_input1", "float_input2"], ["output_mul"])
node_sub = helper.make_node("Sub", ["float_input1", "float_input2"], ["output_sub"])

# ----- BitShift Nodes -----
node_shl = helper.make_node(
    "BitShift",
    ["int_input1", "int_input2"],
    ["output_shl"],
    direction="LEFT"
)

node_shr = helper.make_node(
    "BitShift",
    ["int_input1", "int_input2"],
    ["output_shr"],
    direction="RIGHT"
)

# ----- Create the Graph -----
graph_def = helper.make_graph(
    nodes=[node_add, node_mul, node_sub, node_shl, node_shr],
    name="AddMulSubBitShiftGraph",
    inputs=[float_input1, float_input2, int_input1, int_input2],
    outputs=[output_add, output_mul, output_sub, output_shl, output_shr],
)

# ----- Create and Save the Model -----
model_def = helper.make_model(graph_def, producer_name="arith-bitshift-model")
onnx.save(model_def, "add_mul_sub_shift.onnx")