func.func @mm(%A: tensor<64x64xf32>, %B: tensor<64x64xf32>, %C: tensor<64x64xf32>) -> tensor<64x64xf32> {
  %0 = linalg.matmul ins(%A, %B : tensor<64x64xf32>, tensor<64x64xf32>) outs(%C : tensor<64x64xf32>) -> tensor<64x64xf32>
  return %0 : tensor<64x64xf32>
}
