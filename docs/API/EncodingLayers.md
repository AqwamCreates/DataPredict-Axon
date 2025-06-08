# [API Reference](../API.md) - EncodingLayers

## Constructors

### OneHotEncoding

```

EncodingLayers.OneHotEncoding{tensor: tensor, finalDimensionSize: number, oneHotEncodingMode: string, indexDictionary: {any}}: AutomaticDifferentiationTensor

```

#### Parameters:

* tensor: The tensor that will be padded.

* finalDimensionSize: The final dimension size for the transformed tensor. It is equivalent to the number of labels that are available in the data.

* oneHotEncodingMode: The encoding mode to be used by the one hot encoding block. Available options are:

	* Index (Default)

	* Key

* indexDictionary: The index dictionary to be used to convert keys stored in the tensor to one hot encoding tensor. Must be given if using the "Key" one hot encoding mode.

#### Returns:

* AutomaticDifferentiationTensor: The automatic differentiation tensor that is created as a result of calling this function.


### LabelEncoding

```

EncodingLayers.LabelEncoding{tensor: tensor, valueDictionary: {any}}: AutomaticDifferentiationTensor

```

#### Parameters:

* tensor: The tensor that will be padded.

* valueDictionary: The value dictionary to be used to convert keys stored in the tensor to values for label encoding tensor.

#### Returns:

* AutomaticDifferentiationTensor: The automatic differentiation tensor that is created as a result of calling this function.

### PositionalEncoding

```

EncodingLayers.PositionalEncoding{tensor: tensor, sequenceLength: number, nValue: number}: AutomaticDifferentiationTensor

```

#### Parameters:

* tensor: The tensor that will be padded.

* sequenceLength: The length of the sequence.

* nValue: A user defined value for tuning.

#### Returns:

* AutomaticDifferentiationTensor: The automatic differentiation tensor that is created as a result of calling this function.
