# [API Reference](../API.md) - WeightContainer

## Constructors

### new()

```

WeightContainer.new{TensorAndOptimizerArray: {tensor, number, Optimizer}, updateWeightTensorInPlace: boolean}: WeightContainer

```

#### Parameters:

* TensorAndOptimizerArray: An array containing the ADTensor, the learningRate and the optimizer.

* updateWeightTensorInPlace: Set whether or not to update the weight tensor in place. If true, updates the weight tensor directly for better performance by avoiding new table creation and reducing memory usage. Not supported for scalar values. [Default: true]

#### Returns:

* WeightContainer: The generated WeightContainer object.

## Functions

### gradientDescent()

```

WeightContainer:gradientDescent{}

```

### gradientAscent()

```

WeightContainer:gradientAscent{}

```

### getTensorArray()

```

WeightContainer:getTensorArray{doNotDeepCopy: boolean}: {tensor}

```

#### Parameters:

* doNotDeepCopy: Set whether or not to deep copy the tensor. [Default: false]

#### Returns

* tensorArray: The array containing the tensors from the automatic differentiation tensors.

### setTensorArray()

```

WeightContainer:setTensorArray{tensorArray: {tensor}, doNotDeepCopy: boolean}

```

#### Parameters:

* tensorArray: The array containing the tensors to be loaded to automatic differentiation tensors.

* doNotDeepCopy: Set whether or not to deep copy the tensor. [Default: false]
