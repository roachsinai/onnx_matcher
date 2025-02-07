import onnx_matcher
from onnx import helper, numpy_helper
import onnx
import numpy as np

name = "flu_model_simp_2d_3d_sim.onnx"  # 替换为你的模型文件名
model = onnx.load(name)

# 定义替换策略函数
def merge_gemm(model, i, subgraph):
    gemm1, gemm2 = subgraph[0], subgraph[1]

    def get_v_by_attr_name(op, name):
        for item in op.attribute:
            if item.name == name:
                if item.type == 1:
                    return item.f
                if item.type == 2:
                    return item.i
        return None

    # Extract attributes for the first GEMM
    v = get_v_by_attr_name(gemm1, 'alpha')
    alpha1 = v if v is not None else 1.0
    v = get_v_by_attr_name(gemm1, 'beta')
    beta1 = v if v is not None else 1.0
    v = get_v_by_attr_name(gemm1, 'tranA')
    transA1 = v if v is not None else 0
    v = get_v_by_attr_name(gemm1, 'transB')
    transB1 = v if v is not None else 0
    W1_name = gemm1.input[1]
    b1_name = gemm1.input[2] if len(gemm1.input) > 2 else None

    # Extract attributes for the second GEMM
    v = get_v_by_attr_name(gemm2, 'alpha')
    alpha2 = v if v is not None else 1.0
    v = get_v_by_attr_name(gemm2, 'beta')
    beta2 = v if v is not None else 1.0
    v = get_v_by_attr_name(gemm2, 'transA')
    transA2 = v if v is not None else 0
    v = get_v_by_attr_name(gemm2, 'transB')
    transB2 = v if v is not None else 0
    W2_name = gemm2.input[1]
    b2_name = gemm2.input[2] if len(gemm2.input) > 2 else None

    # Helper function to get initializer by name
    def get_initializer(name):
        for initializer in model.graph.initializer:
            if initializer.name == name:
                return numpy_helper.to_array(initializer)
        raise ValueError(f"Initializer {name} not found in the model.")

    # Retrieve weights and biases
    W1 = get_initializer(W1_name)
    W2 = get_initializer(W2_name)
    b1 = get_initializer(b1_name) if b1_name else None
    b2 = get_initializer(b2_name) if b2_name else None

    # Handle transposes if necessary
    if transB1:
        W1 = W1.T
    if transB2:
        W2 = W2.T

    # Compute the new weights: W_new = W1 @ W2
    W_new = np.matmul(W1, W2)

    # Compute the new bias
    if b1 is not None:
        b_new = alpha2 * beta1 * np.matmul(b1, W2)
    else:
        b_new = np.zeros(W_new.shape[0], dtype=W_new.dtype)

    if b2 is not None:
        b_new = b_new + beta2 * b2
    else:
        # If there's no b2, beta2 is multiplied by zero implicitly
        pass

    # Define names for new weights and biases
    W_new_name = W1_name + "_" + W2_name + "_fused"
    b_new_name = (b1_name + "_" if b1_name else "fused_") + \
                 (b2_name if b2_name else "b2_missing") + "_fused"

    W_new = W_new.transpose(1, 0)
    # Add new weights to the model's initializers
    W_new_initializer = numpy_helper.from_array(W_new, name=W_new_name)
    model.graph.initializer.append(W_new_initializer)

    # Add new biases to the model's initializers
    b_new_initializer = numpy_helper.from_array(b_new, name=b_new_name)
    model.graph.initializer.append(b_new_initializer)

    # Determine the input tensor name (input of gemm1)
    input_name = gemm1.input[0]

    # Determine the output tensor name (output of gemm2)
    output_name = gemm2.output[0]

    # Create the new fused GEMM node
    new_gemm = helper.make_node(
        'Gemm',
        inputs=[input_name, W_new_name, b_new_name],
        outputs=[output_name],
        alpha=alpha1 * alpha2,
        beta=1.0,  # Since biases are already combined
        transA=transA1,  # Assuming transA remains the same as gemm1
        transB=1
    )

    return [new_gemm], []

# 定义子图模式
subgraph_matcher = onnx_matcher.Matcher(
    """
    Gemm(?, ?)
    Gemm(?, ?)
    """
)

# 打印所有匹配的子图
subgraph_matcher.print_match(model)

# 使用替换策略函数进行替换
num_replaced_graph = subgraph_matcher.replace(model, merge_gemm)

print(f"Done for replace {num_replaced_graph} nodes.")
onnx.save(model, "merged_gemm_model.onnx")