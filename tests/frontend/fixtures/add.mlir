func.func @add(%a: tensor<1024xf32>, %b: tensor<1024xf32>, %init: tensor<1024xf32>) -> tensor<1024xf32> {
  %c = linalg.add ins(%a, %b : tensor<1024xf32>, tensor<1024xf32>) outs(%init : tensor<1024xf32>) -> tensor<1024xf32>
  return %c : tensor<1024xf32>
}
