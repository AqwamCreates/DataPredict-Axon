# [API Reference](../API.md) - AutomaticDifferentiationTensor

## Constructors

### new()

```

AutomaticDifferentiationTensor.new{tensor: tensor, PartialFirstDerivativeFunction: function, inputTensorArray: {tensor}}: AutomaticDifferentiationTensor

```

#### Parameters:

* tensor: The tensor that is stored inside the automatic differentiation tensor.

* PartialFirstDerivativeFunction: The function ths is involved in creating the automatic differentiation tensor. 

* inputTensorArray: An array containing the tensors that are involved in creating the automatic differentiation tensor.

#### Returns:

* AutomaticDifferentiationTensor: The generated automatic differentiation tensor object.

### coerce()

```

AutomaticDifferentiationTensor.coerce{tensor: tensor}: AutomaticDifferentiationTensor

```

#### Parameters:

* tensor: The tensor that will be coerced.

#### Returns:

* AutomaticDifferentiationTensor: The generated automatic differentiation tensor object.

### stack()

```

AutomaticDifferentiationTensor.stack{tensor: tensor}: AutomaticDifferentiationTensor

```

#### Parameters:

* ...: A list of tensors that will be stacked to form a new automatic differentiation tensor object.

#### Returns:

* AutomaticDifferentiationTensor: The generated automatic differentiation tensor object.

### createTensor()

```

AutomaticDifferentiationTensor.createTensor{dimensionSizeArray: {number}, allValues: number}: AutomaticDifferentiationTensor

```

#### Parameters:

* dimensionSizeArray: The dimension size array for the automatic differentiation tensor. 

* allValues: The values to be set for the automatic differentiation tensor.

### createRandomNormalTensor()

```

AutomaticDifferentiationTensor.createRandomNormalTensor{dimensionSizeArray: {number}, mean: number, standardDeviation: number}: AutomaticDifferentiationTensor

```

#### Parameters:

* dimensionSizeArray: The dimension size array for the automatic differentiation tensor. 

* mean: The mean for the generated values.

* standardDeviation: The standard deviation for the generated values.

#### Returns:

* AutomaticDifferentiationTensor: The generated automatic differentiation tensor object.

### createRandomUniformTensor()

```

AutomaticDifferentiationTensor.createRandomNormalTensor{dimensionSizeArray: {number}, minimumValue: number, maximumValue: number}: AutomaticDifferentiationTensor

```

#### Parameters:

* dimensionSizeArray: The dimension size array for the automatic differentiation tensor. 

* minimumValue: The minimum value for the generated values.

* maximumValue: The maximum value for the generated values.

#### Returns:

* AutomaticDifferentiationTensor: The generated automatic differentiation tensor object.

### radian()

```

AutomaticDifferentiationTensor.radian{tensor: tensor}: AutomaticDifferentiationTensor

```

#### Parameters:

* tensor: The tensor that is used by the automatic differentiation tensor.

#### Returns:

* AutomaticDifferentiationTensor: The generated automatic differentiation tensor object.

### degree()

```

AutomaticDifferentiationTensor.degree{tensor: tensor}: AutomaticDifferentiationTensor

```

#### Parameters:

* tensor: The tensor that is used by the automatic differentiation tensor.

#### Returns:

* AutomaticDifferentiationTensor: The generated automatic differentiation tensor object.

### sin()

```

AutomaticDifferentiationTensor.sin{tensor: tensor}: AutomaticDifferentiationTensor

```

#### Parameters:

* tensor: The tensor that is used by the automatic differentiation tensor.

#### Returns:

* AutomaticDifferentiationTensor: The generated automatic differentiation tensor object.

### cos()

```

AutomaticDifferentiationTensor.cos{tensor: tensor}: AutomaticDifferentiationTensor

```

#### Parameters:

* tensor: The tensor that is used by the automatic differentiation tensor.

#### Returns:

* AutomaticDifferentiationTensor: The generated automatic differentiation tensor object.

### tan()

```

AutomaticDifferentiationTensor.tan{tensor: tensor}: AutomaticDifferentiationTensor

```

#### Parameters:

* tensor: The tensor that is used by the automatic differentiation tensor.

#### Returns:

* AutomaticDifferentiationTensor: The generated automatic differentiation tensor object.

### exponent()

```

AutomaticDifferentiationTensor.exponent{tensor: tensor}: AutomaticDifferentiationTensor

```

#### Parameters:

* tensor: The tensor that is used by the automatic differentiation tensor.

#### Returns:

* AutomaticDifferentiationTensor: The generated automatic differentiation tensor object.

### logarithm()

```

AutomaticDifferentiationTensor.logarithm{numberTensor: tensor, baseTensor: tensor}: AutomaticDifferentiationTensor

```

#### Parameters:

* numberTensor: The number tensor that is used by the automatic differentiation tensor.

* baseTensor: The base tensor that is used by the automatic differentiation tensor.

#### Returns:

* AutomaticDifferentiationTensor: The generated automatic differentiation tensor object.

### clamp()

```

AutomaticDifferentiationTensor.clamp{tensor: tensor}: AutomaticDifferentiationTensor

```

#### Parameters:

* tensor: The tensor that is used by the automatic differentiation tensor.

* upperBondTensor: The upper bound tensor that is stored inside the automatic differentiation tensor.

* lowerBondTensor: The lower bound tensor that is stored inside the automatic differentiation tensor.

#### Returns:

* AutomaticDifferentiationTensor: The generated automatic differentiation tensor object.

### maximum()

```

AutomaticDifferentiationTensor.maximum{...: tensor}: AutomaticDifferentiationTensor

```

#### Parameters:

* tensor: The tensor that is used by the automatic differentiation tensor.

#### Returns:

* AutomaticDifferentiationTensor: The generated automatic differentiation tensor object.

### minimum()

```

AutomaticDifferentiationTensor.minimum{...: tensor}: AutomaticDifferentiationTensor

```

#### Parameters:

* tensor: The tensor that is used by the automatic differentiation tensor.

#### Returns:

* AutomaticDifferentiationTensor: The generated automatic differentiation tensor object.

### add()

```

AutomaticDifferentiationTensor.add{...: tensor}: AutomaticDifferentiationTensor

```

#### Parameters:

* tensor: The tensor that is used by the automatic differentiation tensor.

#### Returns:

* AutomaticDifferentiationTensor: The generated automatic differentiation tensor object.

### subtract()

```

AutomaticDifferentiationTensor.subtract{...: tensor}: AutomaticDifferentiationTensor

```

#### Parameters:

* tensor: The tensor that is used by the automatic differentiation tensor.

#### Returns:

* AutomaticDifferentiationTensor: The generated automatic differentiation tensor object.

### multiply()

```

AutomaticDifferentiationTensor.multiply{...: tensor}: AutomaticDifferentiationTensor

```

#### Parameters:

* tensor: The tensor that is used by the automatic differentiation tensor.

#### Returns:

* AutomaticDifferentiationTensor: The generated automatic differentiation tensor object.

### divide()

```

AutomaticDifferentiationTensor.divide{...: tensor}: AutomaticDifferentiationTensor

```

#### Parameters:

* tensor: The tensor that is used by the automatic differentiation tensor.

#### Returns:

* AutomaticDifferentiationTensor: The generated automatic differentiation tensor object.

### concatenate()

```

AutomaticDifferentiationTensor.concatenate{...: tensor, dimension: number}: AutomaticDifferentiationTensor

```

#### Parameters:

* tensor: The tensor that is used by the automatic differentiation tensor.

#### Returns:

* AutomaticDifferentiationTensor: The generated automatic differentiation tensor object.

## Arithmetic Functions

### findMinimumValue()

```

AutomaticDifferentiationTensor:findMinimumValue{}: AutomaticDifferentiationTensor

```

#### Returns:

* AutomaticDifferentiationTensor: The generated automatic differentiation tensor object.

### findMaximumValue()

```

AutomaticDifferentiationTensor:findMaximumValue{}: AutomaticDifferentiationTensor

```

#### Returns:

* AutomaticDifferentiationTensor: The generated automatic differentiation tensor object.

### sum()

```

AutomaticDifferentiationTensor:sum{dimension: number}: AutomaticDifferentiationTensor

```

### Parameters:

* dimension: The dimension of calculating the sum along an axis. Can be empty. [Default: None]

#### Returns:

* AutomaticDifferentiationTensor: The generated automatic differentiation tensor object.

### unaryMinus()

```

AutomaticDifferentiationTensor:unaryMinus{}: AutomaticDifferentiationTensor

```

#### Returns:

* AutomaticDifferentiationTensor: The generated automatic differentiation tensor object.

### absolute()

```

AutomaticDifferentiationTensor:absolute{}: AutomaticDifferentiationTensor

```

#### Returns:

* AutomaticDifferentiationTensor: The generated automatic differentiation tensor object.

### power()

```

AutomaticDifferentiationTensor:power{otherTensor: tensor}: AutomaticDifferentiationTensor

```

### Parameters:

* otherTensor: The tensor to be used as an exponent.

#### Returns:

* AutomaticDifferentiationTensor: The generated automatic differentiation tensor object.

### dotProduct()

```

AutomaticDifferentiationTensor:dotProduct{otherTensor: tensor}: AutomaticDifferentiationTensor

```

### Parameters:

* otherTensor: The tensor to be used for dot product operation.

#### Returns:

* AutomaticDifferentiationTensor: The generated automatic differentiation tensor object.

### transpose()

```

AutomaticDifferentiationTensor:transpose{dimensionArray: {number}}: AutomaticDifferentiationTensor

```

### Parameters:

* dimensionArray: An array containing the dimensions to transpose the automatic differentiation tensor.

#### Returns:

* AutomaticDifferentiationTensor: The generated automatic differentiation tensor object.

### flatten()

```

AutomaticDifferentiationTensor:flatten{dimensionArray: {number}}: AutomaticDifferentiationTensor

```

### Parameters:

* dimensionArray: An array containing the dimensions to flatten the automatic differentiation tensor.

#### Returns:

* AutomaticDifferentiationTensor: The generated automatic differentiation tensor object.

### reshape()

```

AutomaticDifferentiationTensor:reshape{dimensionSizeArray: {number}}: AutomaticDifferentiationTensor

```

### Parameters:

* dimensionSizeArray: An array containing the dimension sizes to reshape the automatic differentiation tensor.

#### Returns:

* AutomaticDifferentiationTensor: The generated automatic differentiation tensor object.

### permute()

```

AutomaticDifferentiationTensor:permute{dimensionArray: {number}}: AutomaticDifferentiationTensor

```

### Parameters:

* dimensionArray: An array containing the dimensions to permute the automatic differentiation tensor.

#### Returns:

* AutomaticDifferentiationTensor: The generated automatic differentiation tensor object.

### mean()

```

AutomaticDifferentiationTensor:mean{dimension: number}: AutomaticDifferentiationTensor

```

### Parameters:

* dimension: The dimension of calculating the mean along an axis. Can be empty. [Default: None]

#### Returns:

* AutomaticDifferentiationTensor: The generated automatic differentiation tensor object.

### standardDeviation()

```

AutomaticDifferentiationTensor:standardDeviation{dimension: number}: AutomaticDifferentiationTensor

```

### Parameters:

* dimension: The dimension of calculating the standard deviation along an axis. Can be empty. [Default: None]

#### Returns:

* AutomaticDifferentiationTensor: The generated automatic differentiation tensor object.

### zScoreNormalization()

```

AutomaticDifferentiationTensor:zScoreNormalization{dimension: number}: AutomaticDifferentiationTensor

```

### Parameters:

* dimension: The dimension of calculating the z-score normalization along an axis. Can be empty. [Default: None]

#### Returns:

* AutomaticDifferentiationTensor: The generated automatic differentiation tensor object.

### expandDimensionSizes()

```

AutomaticDifferentiationTensor:expandDimensionSizes{targetDimensionSizeArray: {number}}: AutomaticDifferentiationTensor

```

### Parameters:

* targetDimensionSizeArray: The target dimension sizes to add to the automatic differentiation tensor.

#### Returns:

* AutomaticDifferentiationTensor: The generated automatic differentiation tensor object.

### expandNumberOfDimensions()

```

AutomaticDifferentiationTensor:expandNumberOfDimensions{dimensionSizeToAddArray: {number}}: AutomaticDifferentiationTensor

```

### Parameters:

* dimensionSizeToAddArray: The dimension size to add to the automatic differentiation tensor.

#### Returns:

* AutomaticDifferentiationTensor: The generated automatic differentiation tensor object.

## Non-Arithmetic Functions

### getDimensionSizeArray()

```

AutomaticDifferentiationTensor:getDimensionSizeArray{}: number
```

#### Returns:

* dimensionSizeArray: The dimension size array for the automatic differentiation tensor.

### differentiate()

```

AutomaticDifferentiationTensors:differentiate{firstDerivativeTensor: tensor}

```

#### Parameters:

* firstDerivativeTensor: The tensor to be used for calculating chain rule first derivative tensors.

### copy()

```

AutomaticDifferentiationTensor:copy{}: AutomaticDifferentiationTensor

```

#### Returns:

* AutomaticDifferentiationTensor: The generated automatic differentiation tensor object.

### getTensor()

```

AutomaticDifferentiationTensor:getTensor{doNotDeepCopy: boolean}: tensor

```

#### Parameters:

* doNotDeepCopy: Set whether or not to deep copy the tensor.

#### Returns:

* tensor: The tensor that is stored inside the automatic differentiation tensor.

### setTensor()

```

AutomaticDifferentiationTensor:setTensor{tensor: tensor, doNotDeepCopy: boolean}

```

#### Parameters:

* tensor: The tensor that is will be stored inside the automatic differentiation tensor.

* doNotDeepCopy: Set whether or not to deep copy the tensor.

### getTotalFirstDerivativeTensor()

```

AutomaticDifferentiationTensor:getTotalFirstDerivativeTensor{doNotDeepCopy: boolean}: tensor

```

#### Parameters:

* doNotDeepCopy: Set whether or not to deep copy the total first derivative tensor.

#### Returns:

* tensor: The total first derivative tensor that is stored inside the automatic differentiation tensor.

### setTotalFirstDerivativeTensor()

```

AutomaticDifferentiationTensor:setTotalFirstDerivativeTensor{tensor: tensor, doNotDeepCopy: boolean}

```

#### Parameters:

* tensor: The total first derivative tensor that is will be stored inside the automatic differentiation tensor.

* doNotDeepCopy: Set whether or not to deep copy the total first derivative tensor.

### destroy()

```

AutomaticDifferentiationTensor:destroy{areDescendantsDestroyed: boolean, destroyFirstInputTensor: boolean}

```

#### Parameters:

* areDescendantsDestroyed: Set whether or not to destroy descendants of the automatic differentiation tensor.

* destroyFirstInputTensor: Set whether or not to destroy the very first tensors that are used as inputs.


### isAutomaticDifferentiationTensor()

```

AutomaticDifferentiationTensor:isAutomaticDifferentiationTensor{}: boolean

```

#### Returns:

* isAutomaticDifferentiationTensor: A boolean that indicates if the object is an automatic differentiation tensor object.

