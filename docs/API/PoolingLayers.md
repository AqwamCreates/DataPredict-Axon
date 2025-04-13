# [API Reference](../API.md) - PoolingLayers

## Functions

### FastAveragePooling1D

```

PoolingLayers.FastAveragePooling1D{tensor: tensor, strideDimensionSize: number}: AutomaticDifferentiationTensor

```

#### Parameters:

* tensor: The tensor that will be used as inputs.

* strideDimensionSize: The dimension size for the stride. [Default: 1]

#### Returns:

* AutomaticDifferentiationTensor: The automatic differentiation tensor that is created as a result of calling this function.

### FastAveragePooling2D

```

PoolingLayers.FastAveragePooling2D{tensor: tensor, strideDimensionSizeArray: {number}}: AutomaticDifferentiationTensor

```

#### Parameters:

* tensor: The tensor that will be used as inputs.

* strideDimensionSizeArray: The dimension size for the stride. The index of the array represents the dimension and the value represents the size for that particular dimension. [Default: {1, 1}]

#### Returns:

* AutomaticDifferentiationTensor: The automatic differentiation tensor that is created as a result of calling this function.

### FastAveragePooling3D

```

PoolingLayers.FastAveragePooling3D{tensor: tensor, strideDimensionSizeArray: {number}}: AutomaticDifferentiationTensor

```

#### Parameters:

* tensor: The tensor that will be used as inputs.

* strideDimensionSizeArray: The dimension size for the stride. The index of the array represents the dimension and the value represents the size for that particular dimension. [Default: {1, 1, 1}]

#### Returns:

* AutomaticDifferentiationTensor: The automatic differentiation tensor that is created as a result of calling this function.

### FastMaximumPooling1D

```

PoolingLayers.FastMaximumPooling1D{tensor: tensor, strideDimensionSize: number}: AutomaticDifferentiationTensor

```

#### Parameters:

* tensor: The tensor that will be used as inputs.

* strideDimensionSize: The dimension size for the stride. [Default: 1]

#### Returns:

* AutomaticDifferentiationTensor: The automatic differentiation tensor that is created as a result of calling this function.

### FastMaximumPooling2D

```

PoolingLayers.FastMaximumPooling2D{tensor: tensor, strideDimensionSizeArray: {number}}: AutomaticDifferentiationTensor

```

#### Parameters:

* tensor: The tensor that will be used as inputs.

* strideDimensionSizeArray: The dimension size for the stride. The index of the array represents the dimension and the value represents the size for that particular dimension. [Default: {1, 1}]

#### Returns:

* AutomaticDifferentiationTensor: The automatic differentiation tensor that is created as a result of calling this function.

### FastMaximumPooling3D

```

PoolingLayers.FastMaximumPooling3D{tensor: tensor, strideDimensionSizeArray: {number}}: AutomaticDifferentiationTensor

```

#### Parameters:

* tensor: The tensor that will be used as inputs.

* strideDimensionSizeArray: The dimension size for the stride. The index of the array represents the dimension and the value represents the size for that particular dimension. [Default: {1, 1, 1}]

#### Returns:

* AutomaticDifferentiationTensor: The automatic differentiation tensor that is created as a result of calling this function.

### FastMinimumPooling1D

```

PoolingLayers.FastMinimumPooling1D{tensor: tensor, strideDimensionSize: number}: AutomaticDifferentiationTensor

```

#### Parameters:

* tensor: The tensor that will be used as inputs.

* strideDimensionSize: The dimension size for the stride. [Default: 1]

#### Returns:

* AutomaticDifferentiationTensor: The automatic differentiation tensor that is created as a result of calling this function.

### FastMinimumPooling2D

```

PoolingLayers.FastMinimumPooling2D{tensor: tensor, strideDimensionSizeArray: {number}}: AutomaticDifferentiationTensor

```

#### Parameters:

* tensor: The tensor that will be used as inputs.

* strideDimensionSizeArray: The dimension size for the stride. The index of the array represents the dimension and the value represents the size for that particular dimension. [Default: {1, 1}]

#### Returns:

* AutomaticDifferentiationTensor: The automatic differentiation tensor that is created as a result of calling this function.

### FastMinimumPooling3D

```

PoolingLayers.FastMinimumPooling3D{tensor: tensor, strideDimensionSizeArray: {number}}: AutomaticDifferentiationTensor

```

#### Parameters:

* tensor: The tensor that will be used as inputs.

* strideDimensionSizeArray: The dimension size for the stride. The index of the array represents the dimension and the value represents the size for that particular dimension. [Default: {1, 1, 1}]

#### Returns:

* AutomaticDifferentiationTensor: The automatic differentiation tensor that is created as a result of calling this function.