# [API Reference](../API.md) - ConvolutionLayers

## Constructors

### FastConvolution1D

```

ConvolutionLayers.FastConvolution1D{tensor: tensor, weightTensor: tensor, strideDimensionSize: number}: AutomaticDifferentiationTensor

```

#### Parameters:

* tensor: The tensor that will be used as inputs.

* weightTensor: The tensor that will be used as the weights.

* strideDimensionSize: The dimension size for the stride. [Default: 1]

#### Returns:

* AutomaticDifferentiationTensor: The automatic differentiation tensor that is created as a result of calling this function.

### FastConvolution2D

```

ConvolutionLayers.FastConvolution2D{tensor: tensor, weightTensor: tensor, strideDimensionSizeArray: {number}}: AutomaticDifferentiationTensor

```

#### Parameters:

* tensor: The tensor that will be used as inputs.

* weightTensor: The tensor that will be used as the weights.

* strideDimensionSizeArray: The dimension size for the stride. The index of the array represents the dimension and the value represents the size for that particular dimension. [Default: {1, 1}]

#### Returns:

* AutomaticDifferentiationTensor: The automatic differentiation tensor that is created as a result of calling this function.

### FastConvolution3D

```

ConvolutionLayers.FastConvolution3D{tensor: tensor, weightTensor: tensor, strideDimensionSizeArray: {number}}: AutomaticDifferentiationTensor

```

#### Parameters:

* tensor: The tensor that will be used as inputs.

* weightTensor: The tensor that will be used as the weights.

* strideDimensionSizeArray: The dimension size for the stride. The index of the array represents the dimension and the value represents the size for that particular dimension. [Default: {1, 1, 1}]

#### Returns:

* AutomaticDifferentiationTensor: The automatic differentiation tensor that is created as a result of calling this function.

### Convolution1D

```

ConvolutionLayers.Convolution1D{tensor: tensor, weightTensor: tensor, strideDimensionSize: number}: AutomaticDifferentiationTensor

```

#### Parameters:

* tensor: The tensor that will be used as inputs.

* weightTensor: The tensor that will be used as the weights.

* strideDimensionSize: The dimension size for the stride. [Default: 1]

#### Returns:

* AutomaticDifferentiationTensor: The automatic differentiation tensor that is created as a result of calling this function.

### Convolution2D

```

ConvolutionLayers.Convolution2D{tensor: tensor, weightTensor: tensor, strideDimensionSizeArray: {number}}: AutomaticDifferentiationTensor

```

#### Parameters:

* tensor: The tensor that will be used as inputs.

* weightTensor: The tensor that will be used as the weights.

* strideDimensionSizeArray: The dimension size for the stride. The index of the array represents the dimension and the value represents the size for that particular dimension. [Default: {1, 1}]

#### Returns:

* AutomaticDifferentiationTensor: The automatic differentiation tensor that is created as a result of calling this function.

### Convolution3D

```

ConvolutionLayers.Convolution3D{tensor: tensor, weightTensor: tensor, strideDimensionSizeArray: {number}}: AutomaticDifferentiationTensor

```

#### Parameters:

* tensor: The tensor that will be used as inputs.

* weightTensor: The tensor that will be used as the weights.

* strideDimensionSizeArray: The dimension size for the stride. The index of the array represents the dimension and the value represents the size for that particular dimension. [Default: {1, 1, 1}]

#### Returns:

* AutomaticDifferentiationTensor: The automatic differentiation tensor that is created as a result of calling this function.
