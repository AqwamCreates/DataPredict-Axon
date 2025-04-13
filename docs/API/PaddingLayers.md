# [API Reference](../API.md) - PaddingLayers

## Functions

### FastZeroPadding

```

PaddingLayers.FastZeroPadding{tensor: tensor, headPaddingDimensionSizeArray: {number}, tailPaddingDimensionSizeArray: {number}}: AutomaticDifferentiationTensor

```

#### Parameters:

* tensor: The tensor that will be padded.

* headPaddingDimensionSizeArray: The dimension size array for padding to be placed in front of the input tensor. Make note that when adding the padding sizes to the array, the padding starts from the final dimension.

* tailPaddingDimensionSizeArray: The dimension size array for padding to be placed at the back of the input tensor. Make note that when adding the padding sizes to the array, the padding starts from the final dimension.

#### Returns:

* AutomaticDifferentiationTensor: The automatic differentiation tensor that is created as a result of calling this function.

### FastConstantPadding

```

PaddingLayers.FastConstantPadding{tensor: tensor, headPaddingDimensionSizeArray: {number}, tailPaddingDimensionSizeArray: {number}, value: number}: AutomaticDifferentiationTensor

```

#### Parameters:

* tensor: The tensor that will be padded.

* headPaddingDimensionSizeArray: The dimension size array for padding to be placed in front of the input tensor. Make note that when adding the padding sizes to the array, the padding starts from the final dimension.

* tailPaddingDimensionSizeArray: The dimension size array for padding to be placed at the back of the input tensor. Make note that when adding the padding sizes to the array, the padding starts from the final dimension.

* value: The value to fill the padding with. [Default: 0]

#### Returns:

* AutomaticDifferentiationTensor: The automatic differentiation tensor that is created as a result of calling this function.

### FastCircularPadding

```

PaddingLayers.FastCircularPadding{tensor: tensor, headPaddingDimensionSizeArray: {number}, tailPaddingDimensionSizeArray: {number}, value: number}: AutomaticDifferentiationTensor

```

#### Parameters:

* tensor: The tensor that will be padded.

* headPaddingDimensionSizeArray: The dimension size array for padding to be placed in front of the input tensor. Make note that when adding the padding sizes to the array, the padding starts from the final dimension.

* tailPaddingDimensionSizeArray: The dimension size array for padding to be placed at the back of the input tensor. Make note that when adding the padding sizes to the array, the padding starts from the final dimension.

#### Returns:

* AutomaticDifferentiationTensor: The automatic differentiation tensor that is created as a result of calling this function.

### FastReflectionPadding

```

PaddingLayers.FastReflectionPadding{tensor: tensor, headPaddingDimensionSizeArray: {number}, tailPaddingDimensionSizeArray: {number}, value: number}: AutomaticDifferentiationTensor

```

#### Parameters:

* tensor: The tensor that will be padded.

* headPaddingDimensionSizeArray: The dimension size array for padding to be placed in front of the input tensor. Make note that when adding the padding sizes to the array, the padding starts from the final dimension.

* tailPaddingDimensionSizeArray: The dimension size array for padding to be placed at the back of the input tensor. Make note that when adding the padding sizes to the array, the padding starts from the final dimension.

#### Returns:

* AutomaticDifferentiationTensor: The automatic differentiation tensor that is created as a result of calling this function.

### FastReplicationPadding

```

PaddingLayers.FastReplicationPadding{tensor: tensor, headPaddingDimensionSizeArray: {number}, tailPaddingDimensionSizeArray: {number}, value: number}: AutomaticDifferentiationTensor

```

#### Parameters:

* tensor: The tensor that will be padded.

* headPaddingDimensionSizeArray: The dimension size array for padding to be placed in front of the input tensor. Make note that when adding the padding sizes to the array, the padding starts from the final dimension.

* tailPaddingDimensionSizeArray: The dimension size array for padding to be placed at the back of the input tensor. Make note that when adding the padding sizes to the array, the padding starts from the final dimension.

#### Returns:

* AutomaticDifferentiationTensor: The automatic differentiation tensor that is created as a result of calling this function.