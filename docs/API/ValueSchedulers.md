# [API Reference](../API.md) - ValueSchedulers

## Constructors

### new()

```

ValueScheduler.new{CalculateFunction: function, valueSchedulerInternalParameterArray: {}}: ValueScheduler

```

#### Parameters:

* CalculateFunction: The tensor that will be transformed.

* valueSchedulerInternalParameterArray: The value scheduler internal parameters that is used by the value scheduler.

#### Returns:

* ValueScheduler: The generated value scheduler object.

### StepDecay()

```

ValueScheduler.StepDecay{timeStepToDecay: number, decayRate: number, valueSchedulerInternalParameterArray: {}}: ValueScheduler

```

#### Parameters:

* timeStepToDecay: The number of time steps to decay the learning rate. [Default: 100]

* decayRate: The value that controls the rate of decay. [Default: 0.5]

* valueSchedulerInternalParameterArray: The value scheduler internal parameters that is used by the value scheduler.

#### Returns:

* ValueScheduler: The generated value scheduler object.

### TimeDecay()

```

ValueScheduler.TimeDecay{decayRate: number, valueSchedulerInternalParameterArray: {}}: ValueScheduler

```

#### Parameters:

* decayRate: The value that controls the rate of decay. [Default: 0.5]

* valueSchedulerInternalParameterArray: The value scheduler internal parameters that is used by the value scheduler.

#### Returns:

* ValueScheduler: The generated value scheduler object.

## Functions

### calculate()

```

ValueScheduler:calculate{value: number}: value

```

#### Parameters:

* value: The value to be modified by the value scheduler.

#### Returns:

* value: The modified value that is created as a result of calling this function.