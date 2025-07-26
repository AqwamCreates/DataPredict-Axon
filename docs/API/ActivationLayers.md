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

### FastStableSoftmax

```

ActivationLayers.FastStableSoftmax{tensor: tensor, dimension: number}: AutomaticDifferentiationTensor

```

#### Parameters:

* tensor: The tensor that will be transformed.

* dimension: The dimension at which the exponent values are summed. [Default: 1]

#### Returns:

* AutomaticDifferentiationTensor: The automatic differentiation tensor that is created as a result of calling this function.

### Sigmoid

```

ActivationLayers.Sigmoid{tensor: tensor}: AutomaticDifferentiationTensor

```

#### Parameters:

* tensor: The tensor that will be transformed.

#### Returns:

* AutomaticDifferentiationTensor: The automatic differentiation tensor that is created as a result of calling this function.

### BinaryStep

```

ActivationLayers.BinaryStep{tensor: tensor}: AutomaticDifferentiationTensor

```

#### Parameters:

* tensor: The tensor that will be transformed.

#### Returns:

* AutomaticDifferentiationTensor: The automatic differentiation tensor that is created as a result of calling this function.

### RectifiedLinearUnit

```

ActivationLayers.RectifiedLinearUnit{tensor: tensor}: AutomaticDifferentiationTensor

```

#### Parameters:

* tensor: The tensor that will be transformed.

#### Returns:

* AutomaticDifferentiationTensor: The automatic differentiation tensor that is created as a result of calling this function.

### LeakyRectifiedLinearUnit

```

ActivationLayers.LeakyRectifiedLinearUnit{tensor: tensor, negativeSlopeFactor: number}: AutomaticDifferentiationTensor

```

#### Parameters:

* tensor: The tensor that will be transformed.

* negativeSlopeFactor: The value to be multiplied with negative input values. [Default: 0.01]

#### Returns:

* AutomaticDifferentiationTensor: The automatic differentiation tensor that is created as a result of calling this function.

### ExponentLinearUnit

```

ActivationLayers.ExponentLinearUnit{tensor: tensor, negativeSlopeFactor: number}: AutomaticDifferentiationTensor

```

#### Parameters:

* tensor: The tensor that will be transformed.

* negativeSlopeFactor: The value to be multiplied with negative input values. [Default: 0.01]

#### Returns:

* AutomaticDifferentiationTensor: The automatic differentiation tensor that is created as a result of calling this function.

### SigmoidLinearUnit

```

ActivationLayers.SigmoidLinearUnit{tensor: tensor}: AutomaticDifferentiationTensor

```

#### Parameters:

* tensor: The tensor that will be transformed.

#### Returns:

* AutomaticDifferentiationTensor: The automatic differentiation tensor that is created as a result of calling this function.

### Gaussian

```

ActivationLayers.Gaussian{tensor: tensor}: AutomaticDifferentiationTensor

```

#### Parameters:

* tensor: The tensor that will be transformed.

#### Returns:

* AutomaticDifferentiationTensor: The automatic differentiation tensor that is created as a result of calling this function.

### Mish

```

ActivationLayers.Mish{tensor: tensor}: AutomaticDifferentiationTensor

```

#### Parameters:

* tensor: The tensor that will be transformed.

#### Returns:

* AutomaticDifferentiationTensor: The automatic differentiation tensor that is created as a result of calling this function.

### Tanh

```

ActivationLayers.Tanh{tensor: tensor}: AutomaticDifferentiationTensor

```

#### Parameters:

* tensor: The tensor that will be transformed.

#### Returns:

* AutomaticDifferentiationTensor: The automatic differentiation tensor that is created as a result of calling this function.

### Softmax

```

ActivationLayers.Softmax{tensor: tensor, dimension: number}: AutomaticDifferentiationTensor

```

#### Parameters:

* tensor: The tensor that will be transformed.

* dimension: The dimension at which the exponent values are summed. [Default: 1]

#### Returns:

* AutomaticDifferentiationTensor: The automatic differentiation tensor that is created as a result of calling this function.

### StableSoftmax

```

ActivationLayers.StableSoftmax{tensor: tensor, dimension: number}: AutomaticDifferentiationTensor

```

#### Parameters:

* tensor: The tensor that will be transformed.

* dimension: The dimension at which the exponent values are summed. [Default: 1]

#### Returns:

* AutomaticDifferentiationTensor: The automatic differentiation tensor that is created as a result of calling this function.
