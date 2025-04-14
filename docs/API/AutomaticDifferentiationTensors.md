# [API Reference](../API.md) - AutomaticDifferentiationTensors

## Constructors

### new()

```

AutomaticDifferentiationTensors.new{tensor: tensor, PartialFirstDerivativeFunction: function, inputTensorArray: {tensor}}: AutomaticDifferentiationTensor

```

#### Parameters:

* tensor: The tensor that is stored inside the automatic differentiation tensor.

* PartialFirstDerivativeFunction: The function ths is involved in creating the automatic differentiation tensor. 

* inputTensorArray: An array containing the tensors that are involved in creating the automatic differentiation tensor.

#### Returns:

* AutomaticDifferentiationTensor: The generated automatic differentiation tensor object.

### createTensor()

```

AutomaticDifferentiationTensors.createTensor{dimensionSizeArray: {number}, allValues: number}: AutomaticDifferentiationTensor

```

#### Parameters:

* dimensionSizeArray: The dimension size array for the automatic differentiation tensor. 

* allValues: The values to be set for the automatic differentiation tensor.

### createRandomNormalTensor()

```

AutomaticDifferentiationTensors.createRandomNormalTensor{dimensionSizeArray: {number}, mean: number, standardDeviation: number}: AutomaticDifferentiationTensor

```

#### Parameters:

* dimensionSizeArray: The dimension size array for the automatic differentiation tensor. 

* mean: The mean for the generated values.

* standardDeviation: The standard deviation for the generated values.

#### Returns:

* AutomaticDifferentiationTensor: The generated automatic differentiation tensor object.

### createRandomUniformTensor()

```

AutomaticDifferentiationTensors.createRandomNormalTensor{dimensionSizeArray: {number}, minimumValue: number, maximumValue: number}: AutomaticDifferentiationTensor

```

#### Parameters:

* dimensionSizeArray: The dimension size array for the automatic differentiation tensor. 

* minimumValue: The minimum value for the generated values.

* maximumValue: The maximum value for the generated values.

#### Returns:

* AutomaticDifferentiationTensor: The generated automatic differentiation tensor object.

### radian()

```

AutomaticDifferentiationTensors.radian{tensor: tensor}: AutomaticDifferentiationTensor

```

#### Parameters:

* tensor: The tensor that is used by the automatic differentiation tensor.

#### Returns:

* AutomaticDifferentiationTensor: The generated automatic differentiation tensor object.

### degree()

```

AutomaticDifferentiationTensors.degree{tensor: tensor}: AutomaticDifferentiationTensor

```

#### Parameters:

* tensor: The tensor that is used by the automatic differentiation tensor.

#### Returns:

* AutomaticDifferentiationTensor: The generated automatic differentiation tensor object.

### sin()

```

AutomaticDifferentiationTensors.sin{tensor: tensor}: AutomaticDifferentiationTensor

```

#### Parameters:

* tensor: The tensor that is used by the automatic differentiation tensor.

#### Returns:

* AutomaticDifferentiationTensor: The generated automatic differentiation tensor object.

### cos()

```

AutomaticDifferentiationTensors.cos{tensor: tensor}: AutomaticDifferentiationTensor

```

#### Parameters:

* tensor: The tensor that is used by the automatic differentiation tensor.

#### Returns:

* AutomaticDifferentiationTensor: The generated automatic differentiation tensor object.

### tan()

```

AutomaticDifferentiationTensors.tan{tensor: tensor}: AutomaticDifferentiationTensor

```

#### Parameters:

* tensor: The tensor that is used by the automatic differentiation tensor.

#### Returns:

* AutomaticDifferentiationTensor: The generated automatic differentiation tensor object.

### exponent()

```

AutomaticDifferentiationTensors.exponent{tensor: tensor}: AutomaticDifferentiationTensor

```

#### Parameters:

* tensor: The tensor that is used by the automatic differentiation tensor.

#### Returns:

* AutomaticDifferentiationTensor: The generated automatic differentiation tensor object.

### logarithm()

```

AutomaticDifferentiationTensors.logarithm{numberTensor: tensor, baseTensor: tensor}: AutomaticDifferentiationTensor

```

#### Parameters:

* numberTensor: The number tensor that is used by the automatic differentiation tensor.

* baseTensor: The base tensor that is used by the automatic differentiation tensor.

#### Returns:

* AutomaticDifferentiationTensor: The generated automatic differentiation tensor object.

### clamp()

```

AutomaticDifferentiationTensors.clamp{tensor: tensor}: AutomaticDifferentiationTensor

```

#### Parameters:

* tensor: The tensor that is used by the automatic differentiation tensor.

* upperBondTensor: The upper bound tensor that is stored inside the automatic differentiation tensor.

* lowerBondTensor: The lower bound tensor that is stored inside the automatic differentiation tensor.

#### Returns:

* AutomaticDifferentiationTensor: The generated automatic differentiation tensor object.

### maximum()

```

AutomaticDifferentiationTensors.maximum{...: tensor}: AutomaticDifferentiationTensor

```

#### Parameters:

* tensor: The tensor that is used by the automatic differentiation tensor.

#### Returns:

* AutomaticDifferentiationTensor: The generated automatic differentiation tensor object.

### minimum()

```

AutomaticDifferentiationTensors.minimum{...: tensor}: AutomaticDifferentiationTensor

```

#### Parameters:

* tensor: The tensor that is used by the automatic differentiation tensor.

#### Returns:

* AutomaticDifferentiationTensor: The generated automatic differentiation tensor object.

### add()

```

AutomaticDifferentiationTensors.add{...: tensor}: AutomaticDifferentiationTensor

```

#### Parameters:

* tensor: The tensor that is used by the automatic differentiation tensor.

#### Returns:

* AutomaticDifferentiationTensor: The generated automatic differentiation tensor object.

### subtract()

```

AutomaticDifferentiationTensors.subtract{...: tensor}: AutomaticDifferentiationTensor

```

#### Parameters:

* tensor: The tensor that is used by the automatic differentiation tensor.

#### Returns:

* AutomaticDifferentiationTensor: The generated automatic differentiation tensor object.

### multiply()

```

AutomaticDifferentiationTensors.multiply{...: tensor}: AutomaticDifferentiationTensor

```

#### Parameters:

* tensor: The tensor that is used by the automatic differentiation tensor.

#### Returns:

* AutomaticDifferentiationTensor: The generated automatic differentiation tensor object.

### divide()

```

AutomaticDifferentiationTensors.divide{...: tensor}: AutomaticDifferentiationTensor

```

#### Parameters:

* tensor: The tensor that is used by the automatic differentiation tensor.

#### Returns:

* AutomaticDifferentiationTensor: The generated automatic differentiation tensor object.

### concatenate()

```

AutomaticDifferentiationTensors.concatenate{...: tensor, dimension: number}: AutomaticDifferentiationTensor

```

#### Parameters:

* tensor: The tensor that is used by the automatic differentiation tensor.

#### Returns:

* AutomaticDifferentiationTensor: The generated automatic differentiation tensor object.

## Arithmetic Functions

### findMinimumValue()

```

AutomaticDifferentiationTensors:findMinimumValue{}: AutomaticDifferentiationTensor

```

#### Returns:

* AutomaticDifferentiationTensor: The generated automatic differentiation tensor object.

### findMaximumValue()

```

AutomaticDifferentiationTensors:findMaximumValue{}: AutomaticDifferentiationTensor

```

#### Returns:

* AutomaticDifferentiationTensor: The generated automatic differentiation tensor object.

### sum()

```

AutomaticDifferentiationTensors:sum{dimension: number}: AutomaticDifferentiationTensor

```

### Parameters:

* dimension: The dimension of calculating the sum along an axis. Can be empty. [Default: None]

#### Returns:

* AutomaticDifferentiationTensor: The generated automatic differentiation tensor object.

### unaryMinus()

```

AutomaticDifferentiationTensors:unaryMinus{}: AutomaticDifferentiationTensor

```

#### Returns:

* AutomaticDifferentiationTensor: The generated automatic differentiation tensor object.

### absolute()

```

AutomaticDifferentiationTensors:absolute{}: AutomaticDifferentiationTensor

```

#### Returns:

* AutomaticDifferentiationTensor: The generated automatic differentiation tensor object.

### power()

```

AutomaticDifferentiationTensors:power{otherTensor: tensor}: AutomaticDifferentiationTensor

```

### Parameters:

* otherTensor: The tensor to be used as an exponent.

#### Returns:

* AutomaticDifferentiationTensor: The generated automatic differentiation tensor object.

### dotProduct()

```

AutomaticDifferentiationTensors:dotProduct{otherTensor: tensor}: AutomaticDifferentiationTensor

```

### Parameters:

* otherTensor: The tensor to be used for dot product operation.

#### Returns:

* AutomaticDifferentiationTensor: The generated automatic differentiation tensor object.

### transpose()

```

AutomaticDifferentiationTensors:transpose{dimensionArray: {number}}: AutomaticDifferentiationTensor

```

### Parameters:

* dimensionArray: An array containing the dimensions to transpose the automatic differentiation tensor.

#### Returns:

* AutomaticDifferentiationTensor: The generated automatic differentiation tensor object.

### flatten()

```

AutomaticDifferentiationTensors:flatten{dimensionArray: {number}}: AutomaticDifferentiationTensor

```

### Parameters:

* dimensionArray: An array containing the dimensions to flatten the automatic differentiation tensor.

#### Returns:

* AutomaticDifferentiationTensor: The generated automatic differentiation tensor object.

### reshape()

```

AutomaticDifferentiationTensors:reshape{dimensionSizeArray: {number}}: AutomaticDifferentiationTensor

```

### Parameters:

* dimensionSizeArray: An array containing the dimension sizes to reshape the automatic differentiation tensor.

#### Returns:

* AutomaticDifferentiationTensor: The generated automatic differentiation tensor object.

### permute()

```

AutomaticDifferentiationTensors:permute{dimensionArray: {number}}: AutomaticDifferentiationTensor

```

### Parameters:

* dimensionArray: An array containing the dimensions to permute the automatic differentiation tensor.

#### Returns:

* AutomaticDifferentiationTensor: The generated automatic differentiation tensor object.

### mean()

```

AutomaticDifferentiationTensors:mean{dimension: number}: AutomaticDifferentiationTensor

```

### Parameters:

* dimension: The dimension of calculating the mean along an axis. Can be empty. [Default: None]

#### Returns:

* AutomaticDifferentiationTensor: The generated automatic differentiation tensor object.

### standardDeviation()

```

AutomaticDifferentiationTensors:standardDeviation{dimension: number}: AutomaticDifferentiationTensor

```

### Parameters:

* dimension: The dimension of calculating the standard deviation along an axis. Can be empty. [Default: None]

#### Returns:

* AutomaticDifferentiationTensor: The generated automatic differentiation tensor object.

### zScoreNormalization()

```

AutomaticDifferentiationTensors:zScoreNormalization{dimension: number}: AutomaticDifferentiationTensor

```

### Parameters:

* dimension: The dimension of calculating the z-score normalization along an axis. Can be empty. [Default: None]

#### Returns:

* AutomaticDifferentiationTensor: The generated automatic differentiation tensor object.

### expandDimensionSizes()

```

AutomaticDifferentiationTensors:expandDimensionSizes{targetDimensionSizeArray: {number}}: AutomaticDifferentiationTensor

```

### Parameters:

* targetDimensionSizeArray: The target dimension sizes to add to the automatic differentiation tensor.

#### Returns:

* AutomaticDifferentiationTensor: The generated automatic differentiation tensor object.

### expandNumberOfDimensions()

```

AutomaticDifferentiationTensors:expandNumberOfDimensions{dimensionSizeToAddArray: {number}}: AutomaticDifferentiationTensor

```

### Parameters:

* dimensionSizeToAddArray: The dimension size to add to the automatic differentiation tensor.

#### Returns:

* AutomaticDifferentiationTensor: The generated automatic differentiation tensor object.

## Non-Arithmetic Functions

```

AutomaticDifferentiationTensors:differentiate{firstDerivativeTensor: tensor}

```

#### Parameters:

* firstDerivativeTensor: The tensor to be used for calculating chain rule first derivative tensors.

### copy()

```

AutomaticDifferentiationTensors:copy{}: AutomaticDifferentiationTensor

```

#### Returns:

* AutomaticDifferentiationTensor: The generated automatic differentiation tensor object.

### getTensor()

```

AutomaticDifferentiationTensors:getTensor{doNotDeepCopy: boolean}: tensor

```

#### Parameters:

* doNotDeepCopy: Set whether or not to deep copy the tensor.

#### Returns:

* tensor: The tensor that is stored inside the automatic differentiation tensor.

### setTensor()

```

AutomaticDifferentiationTensors:setTensor{tensor: tensor, doNotDeepCopy: boolean}

```

#### Parameters:

* tensor: The tensor that is will be stored inside the automatic differentiation tensor.

* doNotDeepCopy: Set whether or not to deep copy the tensor.

### getTotalFirstDerivativeTensor()

```

AutomaticDifferentiationTensors:getTotalFirstDerivativeTensor{doNotDeepCopy: boolean}: tensor

```

#### Parameters:

* doNotDeepCopy: Set whether or not to deep copy the total first derivative tensor.

#### Returns:

* tensor: The total first derivative tensor that is stored inside the automatic differentiation tensor.

### setTotalFirstDerivativeTensor()

```

AutomaticDifferentiationTensors:setTotalFirstDerivativeTensor{tensor: tensor, doNotDeepCopy: boolean}

```

#### Parameters:

* tensor: The total first derivative tensor that is will be stored inside the automatic differentiation tensor.

* doNotDeepCopy: Set whether or not to deep copy the total first derivative tensor.

### destroy()

```

AutomaticDifferentiationTensors:destroy{areDescendantsDestroyed: boolean, destroyFirstInputTensor: boolean}

```

#### Parameters:

* areDescendantsDestroyed: Set whether or not to destroy descendants of the automatic differentiation tensor.

* destroyFirstInputTensor: Set whether or not to destroy the very first tensors that are used as inputs.


### isAutomaticDifferentiationTensor()

```

AutomaticDifferentiationTensors:isAutomaticDifferentiationTensor{}: boolean

```

#### Returns:

* isAutomaticDifferentiationTensor: A boolean that indicates if the object is an automatic differentiation tensor object.

