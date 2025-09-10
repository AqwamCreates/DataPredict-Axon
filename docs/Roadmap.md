# Roadmap

## Core

The list of items shown below are likely to be implemented due to their mainstream use, ability to increase learning speed, or ability to reduce computational resources.

* Eligibility Traces

  * If actions are selected per game frame, eligibility traces are a must as it allows faster learning due to its ability to filter out noise from overlapping actions.

  * Currently trying to figure out on how to fit this into DataPredict Axon's API. The autodifferentiation nature of the tensors makes it hard to modify temporal difference values with eligibility traces.

## Nice-To-Have

The list of items shown below may not necessarily be implemented in the future. However, they could be prioritized with external demand, collaboration, or funding.

* Dilated Convolution Neural Network

  * Enables larger receptive field without more weight parameters.

  * Good in sparse-data settings.

  * Unknown use cases related to game environments.

* Generalized N-Dimensional Convolution Layer And Pooling Layer

  * Currently we have up to 3 dimensional kernels.

  * Useful for pushing the boundaries of convolutional neural networks.

  * 4 dimensional kernels are used in videos. Unknown use cases for game environments.
