# [API Reference](../API.md) - Optimizers

## Constructors

### new()

```

Optimizer.new{CalculateFunction: function, optimizerInternalParameterArray: {}, LearningRateValueScheduler: LearningRateValueScheduler}: Optimizer

```

#### Parameters:

* CalculateFunction: The tensor that will be transformed.

* LearningRateValueScheduler: The value scheduler object to be used by the learning rate.

* optimizerInternalParameterArray: The optimizer internal parameters that is used by the optimizer.

#### Returns:

* Optimizer: The generated optimizer object.

### AdaptiveGradient()

```

Optimizer.AdaptiveGradient{LearningRateValueScheduler: LearningRateValueScheduler, optimizerInternalParameterArray: {}}: Optimizer

```

#### Parameters:

* optimizerInternalParameterArray: The optimizer internal parameters that is used by the optimizer.

* LearningRateValueScheduler: The value scheduler object to be used by the learning rate.

#### Returns:

* Optimizer: The generated optimizer object.

### AdaptiveGradientDelta()

```

Optimizer.AdaptiveGradientDelta{decayRate: number, epsilon: number, LearningRateValueScheduler: LearningRateValueScheduler, optimizerInternalParameterArray: {}}: Optimizer

```

#### Parameters:

* decayRate: The value that controls the rate of decay. [Default: 0.9]

* epsilon: The value to ensure that the numbers are not divided by zero. [Default: 10^-7]

* LearningRateValueScheduler: The value scheduler object to be used by the learning rate.

* optimizerInternalParameterArray: The optimizer internal parameters that is used by the optimizer.

#### Returns:

* Optimizer: The generated optimizer object.

### AdaptiveMomentEstimation()

```

Optimizer.AdaptiveMomentEstimation{beta1: number, beta2: number, epsilon: number, LearningRateValueScheduler: LearningRateValueScheduler, optimizerInternalParameterArray: {}}: Optimizer

```

#### Parameters:

* beta1: The decay rate of the moving average of the first moment of the gradients. [Default: 0.9]

* beta2: The decay rate of the moving average of the squared gradients. [Default: 0.999]

* epsilon: The value to ensure that the numbers are not divided by zero. [Default: 10^-7]

* LearningRateValueScheduler: The value scheduler object to be used by the learning rate.

* optimizerInternalParameterArray: The optimizer internal parameters that is used by the optimizer.

#### Returns:

* Optimizer: The generated optimizer object.

### AdaptiveMomentEstimationMaximum()

```

Optimizer.AdaptiveMomentEstimationMaximum{beta1: number, beta2: number, epsilon: number, LearningRateValueScheduler: LearningRateValueScheduler, optimizerInternalParameterArray: {}}: Optimizer

```

#### Parameters:

* beta1: The decay rate of the moving average of the first moment of the gradients. [Default: 0.9]

* beta2: The decay rate of the moving average of the squared gradients. [Default: 0.999]

* epsilon: The value to ensure that the numbers are not divided by zero. [Default: 10^-7]

* LearningRateValueScheduler: The value scheduler object to be used by the learning rate.

* optimizerInternalParameterArray: The optimizer internal parameters that is used by the optimizer.

#### Returns:

* Optimizer: The generated optimizer object.

### Gravity()

```

Optimizer.Gravity{initialStepSize: number, movingAverage: number, LearningRateValueScheduler: LearningRateValueScheduler, optimizerInternalParameterArray: {}}: Optimizer

```

#### Parameters:

* initialStepSize: The value to set the initial velocity during the first iteration. [Default: 0.01]

* movingAverage: The value that controls the smoothing of gradients during training. [Default: 0.9]

* LearningRateValueScheduler: The value scheduler object to be used by the learning rate.

* optimizerInternalParameterArray: The optimizer internal parameters that is used by the optimizer.

#### Returns:

* Optimizer: The generated optimizer object.

### Momentum()

```

Optimizer.Momentum{decayRate: number, LearningRateValueScheduler: LearningRateValueScheduler, optimizerInternalParameterArray: {}}: Optimizer

```

#### Parameters:

* decayRate: The value that controls the rate of decay. [Default: 0.9]

* LearningRateValueScheduler: The value scheduler object to be used by the learning rate.

* optimizerInternalParameterArray: The optimizer internal parameters that is used by the optimizer.

#### Returns:

* Optimizer: The generated optimizer object.

### NesterovAcceleratedAdaptiveMomentEstimation()

```

Optimizer.NesterovAcceleratedAdaptiveMomentEstimation{beta1: number, beta2: number, epsilon: number, LearningRateValueScheduler: LearningRateValueScheduler, optimizerInternalParameterArray: {}}: Optimizer

```

#### Parameters:

* beta1: The decay rate of the moving average of the first moment of the gradients. [Default: 0.9]

* beta2: The decay rate of the moving average of the squared gradients. [Default: 0.999]

* epsilon: The value to ensure that the numbers are not divided by zero. [Default: 10^-7]

* LearningRateValueScheduler: The value scheduler object to be used by the learning rate.

* optimizerInternalParameterArray: The optimizer internal parameters that is used by the optimizer.

#### Returns:

* Optimizer: The generated optimizer object.

### RootMeanSquarePropagation()

```

Optimizer.RootMeanSquarePropagation{beta: number, epsilon: number, LearningRateValueScheduler: LearningRateValueScheduler, optimizerInternalParameterArray: {}}: Optimizer

```

#### Parameters:

* beta: The value that controls the exponential decay rate for the moving average of squared gradients. [Default:  0.9]

* epsilon: The value to ensure that the numbers are not divided by zero. [Default: 10^-7]

* LearningRateValueScheduler: The value scheduler object to be used by the learning rate.

* optimizerInternalParameterArray: The optimizer internal parameters that is used by the optimizer.

#### Returns:

* Optimizer: The generated optimizer object.

### LearningRateStepDecay()

```

Optimizer.LearningRateStepDecay{timeStepToDecay: number, decayRate: number, LearningRateValueScheduler: LearningRateValueScheduler, optimizerInternalParameterArray: {}}: Optimizer

```

#### Parameters:

* timeStepToDecay: The number of time steps to decay the learning rate. [Default: 100]

* decayRate: The value that controls the rate of decay. [Default: 0.5]

* LearningRateValueScheduler: The value scheduler object to be used by the learning rate.

* optimizerInternalParameterArray: The optimizer internal parameters that is used by the optimizer.

#### Returns:

* Optimizer: The generated optimizer object.

### LearningRateTimeDecay()

```

Optimizer.LearningRateTimeDecay{decayRate: number, LearningRateValueScheduler: LearningRateValueScheduler, optimizerInternalParameterArray: {}}: Optimizer

```

#### Parameters:

* decayRate: The value that controls the rate of decay. [Default: 0.5]

* LearningRateValueScheduler: The value scheduler object to be used by the learning rate.

* optimizerInternalParameterArray: The optimizer internal parameters that is used by the optimizer.

#### Returns:

* Optimizer: The generated optimizer object.

## Functions

### calculate()

```

Optimizer:calculate{learningRate: number, tensor: tensor}: tensor

```

#### Parameters:

* learningRate: The learning rate to be used by the optimizer.

* tensor: The tensor to be modified by the optimizer.

#### Returns:

* tensor: The modified tensor that is created as a result of calling this function.