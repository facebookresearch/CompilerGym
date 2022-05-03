func @matmul(%a: tensor<${M}x${K}xf32> {linalg.buffer_layout = affine_map<(i, j)[s0, s1] -> (i, j)>},
             %b: tensor<${K}x${N}xf32> {linalg.buffer_layout = affine_map<(i, j)[s0, s1] -> (i, j)>},
             %c: tensor<${M}x${N}xf32> {linalg.buffer_layout = affine_map<(i, j)[s0, s1] -> (i, j)>})
  -> tensor<${M}x${N}xf32> attributes { passthrough = [["target-cpu", "${TARGET_CPU}"], ["prefer-vector-width", "${VECTOR_WIDTH}"]]}
{
  %f0 = constant 0.0 : f32
  %f1 = linalg.fill(%f0, %c) : f32, tensor<${M}x${N}xf32> -> tensor<${M}x${N}xf32>
  %d = linalg.matmul ins(%a, %b : tensor<${M}x${K}xf32>, tensor<${K}x${N}xf32>)
    outs(%f1: tensor<${M}x${N}xf32>) -> tensor<${M}x${N}xf32>
  return %d: tensor<${M}x${N}xf32>
}
