# Getting Started With Deep Reinforcement Learning

## Requirements

* Knowledge on how to build neural networks, which can be found [here](CreatingOurFirstNeuralNetwork.md).

## What Is Reinforcement Learning?

Reinforcement learning is a way for our models to learn on its own without the labels.

We can expect our models to perform poorly at the start of the training but they will gradually improve over time.

## The Basics

### Environment Feature Tensor

An environment feature tensor is a tensor containing all the information related to model's environment. It can contain as many information such as:

* Distance

* Health

* Speed

An example of environment feature tensor will look like this:

```lua
local environmentFeatureTensor = {

  {1, -32, 234, 12, -97} -- 1 is added at first column for bias, but it is optional.

}
```

### Reward Value

This is the value where we reward or punish the models. The properties of reward value is shown below:

* Positive value: Reward

* Negative Value: Punishment

* Large value: Large reward / punishment

* Small value: Small reward / punishment

It is recommended to set the reward that is within the range of:

```lua
-1 <= (total reward * learning rate) <= 1
```

### Action Labels

Action label is a label produced by the model. This label can be a part of decision-making classes or classification classes. For example:

* Decision-making classes: "Up", "Down", "Left", "Right", "Forward", "Backward"

* Classification classes: 1, 2, 3, 4, 5, 6

# Setting Up Our Reinforcement Learning Model

First, let's define a number of variables and setup the first part of the model.

```lua

local ClassesList = {1, 2}

local ADTensor = DataPredictAxon.AutomaticDifferentiationTensor

-- Setting up our weights.

local weightTensor = ADTensor.createRandomNormalTensor{{4, 2}}

local biasTensor = ADTensor.createRandomNormalTensor{{1, 2}}

-- Creating the WeightContainer

local WeightContainer = DataPredictAxon.WeightContainer.new{}

WeightContainer:setWeightTensorDataArray{

  {weightTensor, 0.001},

  {biasTensor, 0.001}

} 

-- Creating the neural network model.

local ActivationLayers = DataPredictAxon.ActivationLayers

local function Model(parameterDictionary) -- Make sure to only pass a table.

  local inputTensor = parameterDictionary[1]

  local weightedInputTensor = inputTensor:dotProduct(weightTensor)

  local weightedInputTensorWithBiasTensor = weightedInputTensor + biasTensor

  local outputTensor = ActivationLayers.LeakyRectifiedLinearUnit{weightedInputTensorWithBiasTensor}

  return outputTensor

end

-- Creating the deep reinforcement learning model.

local DeepSARSA = DataPredictAxon.ReinforcementLearningModels.DeepStateActionRewardStateAction{Model = Model, WeightContainer = WeightContainer}

```

## The Update Functions

All the reinforcement learning models have two important functions: 

* update()

  * categoricalUpdate() is for discrete action spaces

  * diagonalGaussianUpdate() is for continuous action spaces

* episodeUpdate()

Below, I will show a code sample using these functions.

```lua

while true do

  local previousEnvironmentFeatureTensor = {{0, 0, 0, 0, 0}} -- We must keep track our previous feature tensor.

  local action = 1

  for step = 1, 1000, 1 do

    local currentEnvironmentFeatureTensor = fetchEnvironmentFeatureTensor(previousEnvironmentFeatureTensor, action)

    local actionTensor = Model{currentEnvironmentFeatureTensor}

    local reward = getReward(currentEnvironmentFeatureTensor)

    DeepSARSA:categoricalUpdate{previousEnvironmentFeatureTensor, reward, action, currentEnvironmentFeatureTensor, 0} -- update() is called whenever a step is made. The value of zero indicates that the current environment feature tensor is not a terminal state.

    previousEnvironmentFeatureTensor = environmentTensor

    local hasGameEnded = checkIfGameHasEnded(environmentTensor)

    if hasGameEnded then break end

  end

  QLearningNeuralNetwork:episodeUpdate(1) -- episodeUpdate() is used whenever an episode ends. An episode is the total number of steps that determines when the model should stop training. The value of one indicates that the current environment feature tensor is a terminal state.

end

```

As you can see, there are a lot of things that we must track of, but it gives you total freedom on what you want to do with the reinforcement learning models.

That's all for today!
