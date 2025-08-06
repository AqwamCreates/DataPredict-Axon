# [API Reference](../API.md) - ValueSchedulers

## Constructors

### new()

```

Regularizer.new{CalculateFunction: function}: ValueScheduler

```

#### Parameters:

* CalculateFunction: The calculate function to be set.

#### Returns:

* Regularizer: The generated regularizer object.

### ElasticNet()

```

Regularizer.ElasticNet{lambda: number}: Regularizer

```

#### Parameters:

* lambda: The regularization factor. Recommended values are between 0 to 1. [Default: 0.1]

#### Returns:

* Regularizer: The generated regularizer object.

### Lasso()

```

Regularizer.Lasso{lambda: number}: Regularizer

```

#### Parameters:

* lambda: The regularization factor. Recommended values are between 0 to 1. [Default: 0.1]

#### Returns:

* Regularizer: The generated regularizer object.

### Ridge()

```

Regularizer.Ridge{lambda: number}: Regularizer

```

#### Parameters:

* lambda: The regularization factor. Recommended values are between 0 to 1. [Default: 0.1]

#### Returns:

* Regularizer: The generated regularizer object.
