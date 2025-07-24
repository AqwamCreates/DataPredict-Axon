# [API Reference](../API.md) - RecurrentModels

## Constructors

### RecurrentNeuralNetworkCell()

```

RecurrentModels.RecurrentNeuralNetworkCell{inputSize: number, hiddenSize: number, learningRate: number, activationFunction: string}: function, WeightContainer, function, function

```

#### Parameters:

* inputSize: The number of features it takes as inputs.

* hiddenSize: The number of features it will produce.

* learningRate: The speed at which the model learns. Recommended that the value is set between 0 to 1.

* activationFunction: The activation function to be used for weight activation. [Default: FastTanh]

#### Returns:

* Model: The model that is constructed using a set of parameters.

* WeightContainer: The generated WeightContainer object.

* reset: The function to reset the hidden state.

* setHiddenStateTensor: The function to set the hiddent state.

### GatedRecurrentUnit()

```

RecurrentModels.GatedRecurrentUnit{inputSize: number, hiddenSize: number, learningRate: number, activationFunction: string}: function, WeightContainer, function, function

```

#### Parameters:

* inputSize: The number of features it takes as inputs.

* hiddenSize: The number of features it will produce.

* learningRate: The speed at which the model learns. Recommended that the value is set between 0 to 1.

* activationFunction: The activation function to be used for weight activation. [Default: FastTanh]

#### Returns:

* Model: The model that is constructed using a set of parameters.

* WeightContainer: The generated WeightContainer object.

* reset: The function to reset the hidden state.

* setHiddenStateTensor: The function to set the hiddent state.
