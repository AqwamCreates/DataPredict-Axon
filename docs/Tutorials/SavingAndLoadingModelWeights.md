# Saving And Loading Model Weights

DataPredict Axon provides the ability to save and load weights from trained models. There are two ways to access these weights.

* AutomaticDifferentiationTensor (ADTensor)

* WeightContainer

## Saving And Loading Weights From The AutomaticDifferentiationTensor Classes

In order to save the weights from AutomaticDifferentiationTensor classes, we first need to call the getTensor() function on one of our ADTensors.

```lua

local savedTensor = ADTensor:getTensor()

```

This should make a deep copy of the weights to savedTensor variable.

To load the weights, all you need to do is to call the setTensor() function.

```lua

ADTensor:setTensor(savedTensor)

```

## Saving And Loading Weights From The WeightContainer

In order to save the weights from WeightContainer, we first need to call the getTensorArray() function.

```lua

local savedTensorArray = WeightContainer:getTensorArray()

```

This should make a deep copy of the weights to savedTensorArray variable.

To load a tensorArray, all you need to do is to call the setTensorArray() function.

```lua

WeightContainer:setTensorArray(savedTensorArray)

```

## What To Do With The Weights?

You have two ways of saving the weights:

* Storing it to DataStores.

* Copy paste the text printed out by the TensorL library and place it in a text file or Roblox's ModuleScripts.

## Wrapping up

Saving and loading on DataPredict Neural has never been easier. All you need is to call few lines of codes and you're off!

That's all you need to do. Pretty simple, right?

Thank you very much for reading this tutorial!
