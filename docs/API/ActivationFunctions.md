# [API Reference](../API.md) - ActivationFunctions

## Functions

### FastSigmoid

```

ActivationFunctions.FastSigmoid{tensor: tensor}: AutomaticDifferentiationTensor

```

#### Parameters

* tensor: The tensor that will be transformed.

#### Returns

* AutomaticDifferentiationTensor: The automatic differentiation tensor that is created as a result of calling this function.

### FastBinaryStep

```

ActivationFunctions.FastBinaryStep{tensor: tensor}: AutomaticDifferentiationTensor

```

#### Parameters

* tensor: The tensor that will be transformed.

#### Returns

* AutomaticDifferentiationTensor: The automatic differentiation tensor that is created as a result of calling this function.

### FastRectifiedLinearUnit

```

ActivationFunctions.FastRectifiedLinearUnit{tensor: tensor}: AutomaticDifferentiationTensor

```

#### Parameters

* tensor: The tensor that will be transformed.

#### Returns

* AutomaticDifferentiationTensor: The automatic differentiation tensor that is created as a result of calling this function.

### FastLeakyRectifiedLinearUnit

```

ActivationFunctions.FastLeakyRectifiedLinearUnit{tensor: tensor, negativeSlopeFactor: number}: AutomaticDifferentiationTensor

```

#### Parameters

* tensor: The tensor that will be transformed.

* negativeSlopeFactor: The value to be multiplied with negative input values. [Default: 0.01]

#### Returns

* AutomaticDifferentiationTensor: The automatic differentiation tensor that is created as a result of calling this function.

### FastExponentLinearUnit

```

ActivationFunctions.FastExponentLinearUnit{tensor: tensor, negativeSlopeFactor: number}: AutomaticDifferentiationTensor

```

#### Parameters

* tensor: The tensor that will be transformed.

* negativeSlopeFactor: The value to be multiplied with negative input values. [Default: 0.01]

#### Returns

* AutomaticDifferentiationTensor: The automatic differentiation tensor that is created as a result of calling this function.

### FastSigmoidLinearUnit

```

ActivationFunctions.FastSigmoidLinearUnit{tensor: tensor}: AutomaticDifferentiationTensor

```

#### Parameters

* tensor: The tensor that will be transformed.

#### Returns

* AutomaticDifferentiationTensor: The automatic differentiation tensor that is created as a result of calling this function.

### FastGaussian

```

ActivationFunctions.FastGaussian{tensor: tensor}: AutomaticDifferentiationTensor

```

#### Parameters

* tensor: The tensor that will be transformed.

#### Returns

* AutomaticDifferentiationTensor: The automatic differentiation tensor that is created as a result of calling this function.

### FastMish

```

ActivationFunctions.FastMish{tensor: tensor}: AutomaticDifferentiationTensor

```

#### Parameters

* tensor: The tensor that will be transformed.

#### Returns

* AutomaticDifferentiationTensor: The automatic differentiation tensor that is created as a result of calling this function.

### FastTanh

```

ActivationFunctions.FastTanh{tensor: tensor}: AutomaticDifferentiationTensor

```

#### Parameters

* tensor: The tensor that will be transformed.

#### Returns

* AutomaticDifferentiationTensor: The automatic differentiation tensor that is created as a result of calling this function.

### FastSoftmax

```

ActivationFunctions.FastSoftmax{tensor: tensor, dimension: number}: AutomaticDifferentiationTensor

```

#### Parameters

* tensor: The tensor that will be transformed.

* dimension: The dimension at which the exponent values are summed. [Default: 1]

#### Returns

* AutomaticDifferentiationTensor: The automatic differentiation tensor that is created as a result of calling this function.