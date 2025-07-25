# [API Reference](../API.md) - PoolingLayers

## Constructors

### FastAveragePooling1D

```

PoolingLayers.FastAveragePooling1D{tensor: tensor, kernelDimensionSize: number, strideDimensionSize: number}: AutomaticDifferentiationTensor

```

#### Parameters:

* tensor: The tensor that will be used as inputs.

* kernelDimensionSize: The dimension size for the kernel. [Default: 2]

* strideDimensionSize: The dimension size for the stride. [Default: 1]

#### Returns:

* AutomaticDifferentiationTensor: The automatic differentiation tensor that is created as a result of calling this function.

### FastAveragePooling2D

```

PoolingLayers.FastAveragePooling2D{tensor: tensor,  kernelDimensionSize: {number}, strideDimensionSizeArray: {number}}: AutomaticDifferentiationTensor

```

#### Parameters:

* tensor: The tensor that will be used as inputs.

* kernelDimensionSizeArray: The dimension size for the stride. The index of the array represents the dimension and the value represents the size for that particular dimension. [Default: {2, 2}]

* strideDimensionSizeArray: The dimension size for the stride. The index of the array represents the dimension and the value represents the size for that particular dimension. [Default: {1, 1}]

#### Returns:

* AutomaticDifferentiationTensor: The automatic differentiation tensor that is created as a result of calling this function.

### FastAveragePooling3D

```

PoolingLayers.FastAveragePooling3D{tensor: tensor,  kernelDimensionSizeArray: {number}, strideDimensionSizeArray: {number}}: AutomaticDifferentiationTensor

```

#### Parameters:

* tensor: The tensor that will be used as inputs.

* kernelDimensionSizeArray: The dimension size for the stride. The index of the array represents the dimension and the value represents the size for that particular dimension. [Default: {2, 2, 2}]

* strideDimensionSizeArray: The dimension size for the stride. The index of the array represents the dimension and the value represents the size for that particular dimension. [Default: {1, 1, 1}]

#### Returns:

* AutomaticDifferentiationTensor: The automatic differentiation tensor that is created as a result of calling this function.

### FastMaximumPooling1D

```

PoolingLayers.FastMaximumPooling1D{tensor: tensor, kernelDimensionSize: number, strideDimensionSize: number}: AutomaticDifferentiationTensor

```

#### Parameters:

* tensor: The tensor that will be used as inputs.

* kernelDimensionSize: The dimension size for the kernel. [Default: 2]

* strideDimensionSize: The dimension size for the stride. [Default: 1]

#### Returns:

* AutomaticDifferentiationTensor: The automatic differentiation tensor that is created as a result of calling this function.

### FastMaximumPooling2D

```

PoolingLayers.FastMaximumPooling2D{tensor: tensor,  kernelDimensionSize: {number}, strideDimensionSizeArray: {number}}: AutomaticDifferentiationTensor

```

#### Parameters:

* tensor: The tensor that will be used as inputs.

* kernelDimensionSizeArray: The dimension size for the stride. The index of the array represents the dimension and the value represents the size for that particular dimension. [Default: {2, 2}]

* strideDimensionSizeArray: The dimension size for the stride. The index of the array represents the dimension and the value represents the size for that particular dimension. [Default: {1, 1}]

#### Returns:

* AutomaticDifferentiationTensor: The automatic differentiation tensor that is created as a result of calling this function.

### FastMaximumPooling3D

```

PoolingLayers.FastMaximumPooling3D{tensor: tensor,  kernelDimensionSizeArray: {number}, strideDimensionSizeArray: {number}}: AutomaticDifferentiationTensor

```

#### Parameters:

* tensor: The tensor that will be used as inputs.

* kernelDimensionSizeArray: The dimension size for the stride. The index of the array represents the dimension and the value represents the size for that particular dimension. [Default: {2, 2, 2}]

* strideDimensionSizeArray: The dimension size for the stride. The index of the array represents the dimension and the value represents the size for that particular dimension. [Default: {1, 1, 1}]

#### Returns:

* AutomaticDifferentiationTensor: The automatic differentiation tensor that is created as a result of calling this function.

### FastMinimumPooling1D

```

PoolingLayers.FastMinimumPooling1D{tensor: tensor, kernelDimensionSize: number, strideDimensionSize: number}: AutomaticDifferentiationTensor

```

#### Parameters:

* tensor: The tensor that will be used as inputs.

* kernelDimensionSize: The dimension size for the kernel. [Default: 2]

* strideDimensionSize: The dimension size for the stride. [Default: 1]

#### Returns:

* AutomaticDifferentiationTensor: The automatic differentiation tensor that is created as a result of calling this function.

### FastMinimumPooling2D

```

PoolingLayers.FastMinimumPooling2D{tensor: tensor,  kernelDimensionSize: {number}, strideDimensionSizeArray: {number}}: AutomaticDifferentiationTensor

```

#### Parameters:

* tensor: The tensor that will be used as inputs.

* kernelDimensionSizeArray: The dimension size for the stride. The index of the array represents the dimension and the value represents the size for that particular dimension. [Default: {2, 2}]

* strideDimensionSizeArray: The dimension size for the stride. The index of the array represents the dimension and the value represents the size for that particular dimension. [Default: {1, 1}]

#### Returns:

* AutomaticDifferentiationTensor: The automatic differentiation tensor that is created as a result of calling this function.

### FastMinimumPooling3D

```

PoolingLayers.FastMinimumPooling3D{tensor: tensor,  kernelDimensionSizeArray: {number}, strideDimensionSizeArray: {number}}: AutomaticDifferentiationTensor

```

#### Parameters:

* tensor: The tensor that will be used as inputs.

* kernelDimensionSizeArray: The dimension size for the stride. The index of the array represents the dimension and the value represents the size for that particular dimension. [Default: {2, 2, 2}]

* strideDimensionSizeArray: The dimension size for the stride. The index of the array represents the dimension and the value represents the size for that particular dimension. [Default: {1, 1, 1}]

#### Returns:

* AutomaticDifferentiationTensor: The automatic differentiation tensor that is created as a result of calling this function.

### FastMaximumUnpoolingPooling1D

```

PoolingLayers.FastMaximumUnpoolingPooling1D{tensor: tensor, kernelDimensionSize: number, strideDimensionSize: number, unpoolingMethod: string}: AutomaticDifferentiationTensor

```

#### Parameters:

* tensor: The tensor that will be used as inputs.

* kernelDimensionSize: The dimension size for the kernel. [Default: 2]

* strideDimensionSize: The dimension size for the stride. [Default: 1]

* unpoolingMethod: The unpooling method that determines how the transformed tensor is generated. Available options are:

	* NearestNeighbour (Default)

	* BedOfNails

#### Returns:

* AutomaticDifferentiationTensor: The automatic differentiation tensor that is created as a result of calling this function.

### FastMaximumUnpoolingPooling2D

```

PoolingLayers.FastMaximumUnpoolingPooling2D{tensor: tensor,  kernelDimensionSize: {number}, strideDimensionSizeArray: {number}: unpoolingMethod: string}: AutomaticDifferentiationTensor

```

#### Parameters:

* tensor: The tensor that will be used as inputs.

* kernelDimensionSizeArray: The dimension size for the stride. The index of the array represents the dimension and the value represents the size for that particular dimension. [Default: {2, 2}]

* strideDimensionSizeArray: The dimension size for the stride. The index of the array represents the dimension and the value represents the size for that particular dimension. [Default: {1, 1}]

* unpoolingMethod: The unpooling method that determines how the transformed tensor is generated. Available options are:

	* NearestNeighbour (Default)

	* BedOfNails

#### Returns:

* AutomaticDifferentiationTensor: The automatic differentiation tensor that is created as a result of calling this function.

### FastMaximumUnpoolingPooling3D

```

PoolingLayers.FastMaximumUnpoolingPooling3D{tensor: tensor,  kernelDimensionSizeArray: {number}, strideDimensionSizeArray: {number}, unpoolingMethod: string}: AutomaticDifferentiationTensor

```

#### Parameters:

* tensor: The tensor that will be used as inputs.

* kernelDimensionSizeArray: The dimension size for the stride. The index of the array represents the dimension and the value represents the size for that particular dimension. [Default: {2, 2, 2}]

* strideDimensionSizeArray: The dimension size for the stride. The index of the array represents the dimension and the value represents the size for that particular dimension. [Default: {1, 1, 1}]

* unpoolingMethod: The unpooling method that determines how the transformed tensor is generated. Available options are:

	* NearestNeighbour (Default)

	* BedOfNails

#### Returns:

* AutomaticDifferentiationTensor: The automatic differentiation tensor that is created as a result of calling this function.

### AveragePooling1D

```

PoolingLayers.AveragePooling1D{tensor: tensor, kernelDimensionSize: number, strideDimensionSize: number}: AutomaticDifferentiationTensor

```

#### Parameters:

* tensor: The tensor that will be used as inputs.

* kernelDimensionSize: The dimension size for the kernel. [Default: 2]

* strideDimensionSize: The dimension size for the stride. [Default: 1]

#### Returns:

* AutomaticDifferentiationTensor: The automatic differentiation tensor that is created as a result of calling this function.

### AveragePooling2D

```

PoolingLayers.AveragePooling2D{tensor: tensor,  kernelDimensionSize: {number}, strideDimensionSizeArray: {number}}: AutomaticDifferentiationTensor

```

#### Parameters:

* tensor: The tensor that will be used as inputs.

* kernelDimensionSizeArray: The dimension size for the stride. The index of the array represents the dimension and the value represents the size for that particular dimension. [Default: {2, 2}]

* strideDimensionSizeArray: The dimension size for the stride. The index of the array represents the dimension and the value represents the size for that particular dimension. [Default: {1, 1}]

#### Returns:

* AutomaticDifferentiationTensor: The automatic differentiation tensor that is created as a result of calling this function.

### AveragePooling3D

```

PoolingLayers.AveragePooling3D{tensor: tensor,  kernelDimensionSizeArray: {number}, strideDimensionSizeArray: {number}}: AutomaticDifferentiationTensor

```

#### Parameters:

* tensor: The tensor that will be used as inputs.

* kernelDimensionSizeArray: The dimension size for the stride. The index of the array represents the dimension and the value represents the size for that particular dimension. [Default: {2, 2, 2}]

* strideDimensionSizeArray: The dimension size for the stride. The index of the array represents the dimension and the value represents the size for that particular dimension. [Default: {1, 1, 1}]

#### Returns:

* AutomaticDifferentiationTensor: The automatic differentiation tensor that is created as a result of calling this function.

### MaximumPooling1D

```

PoolingLayers.MaximumPooling1D{tensor: tensor, kernelDimensionSize: number, strideDimensionSize: number}: AutomaticDifferentiationTensor

```

#### Parameters:

* tensor: The tensor that will be used as inputs.

* kernelDimensionSize: The dimension size for the kernel. [Default: 2]

* strideDimensionSize: The dimension size for the stride. [Default: 1]

#### Returns:

* AutomaticDifferentiationTensor: The automatic differentiation tensor that is created as a result of calling this function.

### MaximumPooling2D

```

PoolingLayers.MaximumPooling2D{tensor: tensor,  kernelDimensionSize: {number}, strideDimensionSizeArray: {number}}: AutomaticDifferentiationTensor

```

#### Parameters:

* tensor: The tensor that will be used as inputs.

* kernelDimensionSizeArray: The dimension size for the stride. The index of the array represents the dimension and the value represents the size for that particular dimension. [Default: {2, 2}]

* strideDimensionSizeArray: The dimension size for the stride. The index of the array represents the dimension and the value represents the size for that particular dimension. [Default: {1, 1}]

#### Returns:

* AutomaticDifferentiationTensor: The automatic differentiation tensor that is created as a result of calling this function.

### MaximumPooling3D

```

PoolingLayers.MaximumPooling3D{tensor: tensor,  kernelDimensionSizeArray: {number}, strideDimensionSizeArray: {number}}: AutomaticDifferentiationTensor

```

#### Parameters:

* tensor: The tensor that will be used as inputs.

* kernelDimensionSizeArray: The dimension size for the stride. The index of the array represents the dimension and the value represents the size for that particular dimension. [Default: {2, 2, 2}]

* strideDimensionSizeArray: The dimension size for the stride. The index of the array represents the dimension and the value represents the size for that particular dimension. [Default: {1, 1, 1}]

#### Returns:

* AutomaticDifferentiationTensor: The automatic differentiation tensor that is created as a result of calling this function.

### MinimumPooling1D

```

PoolingLayers.MinimumPooling1D{tensor: tensor, kernelDimensionSize: number, strideDimensionSize: number}: AutomaticDifferentiationTensor

```

#### Parameters:

* tensor: The tensor that will be used as inputs.

* kernelDimensionSize: The dimension size for the kernel. [Default: 2]

* strideDimensionSize: The dimension size for the stride. [Default: 1]

#### Returns:

* AutomaticDifferentiationTensor: The automatic differentiation tensor that is created as a result of calling this function.

### MinimumPooling2D

```

PoolingLayers.MinimumPooling2D{tensor: tensor,  kernelDimensionSize: {number}, strideDimensionSizeArray: {number}}: AutomaticDifferentiationTensor

```

#### Parameters:

* tensor: The tensor that will be used as inputs.

* kernelDimensionSizeArray: The dimension size for the stride. The index of the array represents the dimension and the value represents the size for that particular dimension. [Default: {2, 2}]

* strideDimensionSizeArray: The dimension size for the stride. The index of the array represents the dimension and the value represents the size for that particular dimension. [Default: {1, 1}]

#### Returns:

* AutomaticDifferentiationTensor: The automatic differentiation tensor that is created as a result of calling this function.

### MinimumPooling3D

```

PoolingLayers.MinimumPooling3D{tensor: tensor,  kernelDimensionSizeArray: {number}, strideDimensionSizeArray: {number}}: AutomaticDifferentiationTensor

```

#### Parameters:

* tensor: The tensor that will be used as inputs.

* kernelDimensionSizeArray: The dimension size for the stride. The index of the array represents the dimension and the value represents the size for that particular dimension. [Default: {2, 2, 2}]

* strideDimensionSizeArray: The dimension size for the stride. The index of the array represents the dimension and the value represents the size for that particular dimension. [Default: {1, 1, 1}]

#### Returns:

* AutomaticDifferentiationTensor: The automatic differentiation tensor that is created as a result of calling this function.
