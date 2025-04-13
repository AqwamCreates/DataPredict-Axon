# [API Reference](../API.md) - ActivationLayers

## Functions

### FastSigmoid

```

ActivationLayers.FastSigmoid{tensor: tensor}: AutomaticDifferentiationTensor

```

#### Parameters:

* tensor: The tensor that will be transformed.

#### Returns:

* AutomaticDifferentiationTensor: The automatic differentiation tensor that is created as a result of calling this function.

### FastBinaryStep

```

ActivationLayers.FastBinaryStep{tensor: tensor}: AutomaticDifferentiationTensor

```

#### Parameters:

* tensor: The tensor that will be transformed.

#### Returns:

* AutomaticDifferentiationTensor: The automatic differentiation tensor that is created as a result of calling this function.

### FastRectifiedLinearUnit

```

ActivationLayers.FastRectifiedLinearUnit{tensor: tensor}: AutomaticDifferentiationTensor

```

#### Parameters:

* tensor: The tensor that will be transformed.

#### Returns:

* AutomaticDifferentiationTensor: The automatic differentiation tensor that is created as a result of calling this function.

### FastLeakyRectifiedLinearUnit

```

ActivationLayers.FastLeakyRectifiedLinearUnit{tensor: tensor, negativeSlopeFactor: number}: AutomaticDifferentiationTensor

```

#### Parameters:

* tensor: The tensor that will be transformed.

* negativeSlopeFactor: The value to be multiplied with negative input values. [Default: 0.01]

#### Returns:

* AutomaticDifferentiationTensor: The automatic differentiation tensor that is created as a result of calling this function.

### FastExponentLinearUnit

```

ActivationLayers.FastExponentLinearUnit{tensor: tensor, negativeSlopeFactor: number}: AutomaticDifferentiationTensor

```

#### Parameters:

* tensor: The tensor that will be transformed.

* negativeSlopeFactor: The value to be multiplied with negative input values. [Default: 0.01]

#### Returns:

* AutomaticDifferentiationTensor: The automatic differentiation tensor that is created as a result of calling this function.

### FastSigmoidLinearUnit

```

ActivationLayers.FastSigmoidLinearUnit{tensor: tensor}: AutomaticDifferentiationTensor

```

#### Parameters:

* tensor: The tensor that will be transformed.

#### Returns:

* AutomaticDifferentiationTensor: The automatic differentiation tensor that is created as a result of calling this function.

### FastGaussian

```

ActivationLayers.FastGaussian{tensor: tensor}: AutomaticDifferentiationTensor

```

#### Parameters:

* tensor: The tensor that will be transformed.

#### Returns:

* AutomaticDifferentiationTensor: The automatic differentiation tensor that is created as a result of calling this function.

### FastMish

```

ActivationLayers.FastMish{tensor: tensor}: AutomaticDifferentiationTensor

```

#### Parameters:

* tensor: The tensor that will be transformed.

#### Returns:

* AutomaticDifferentiationTensor: The automatic differentiation tensor that is created as a result of calling this function.

### FastTanh

```

ActivationLayers.FastTanh{tensor: tensor}: AutomaticDifferentiationTensor

```

#### Parameters:

* tensor: The tensor that will be transformed.

#### Returns:

* AutomaticDifferentiationTensor: The automatic differentiation tensor that is created as a result of calling this function.

### FastSoftmax

```

ActivationLayers.FastSoftmax{tensor: tensor, dimension: number}: AutomaticDifferentiationTensor

```

#### Parameters:

* tensor: The tensor that will be transformed.

* dimension: The dimension at which the exponent values are summed. [Default: 1]

#### Returns:

* AutomaticDifferentiationTensor: The automatic differentiation tensor that is created as a result of calling this function.