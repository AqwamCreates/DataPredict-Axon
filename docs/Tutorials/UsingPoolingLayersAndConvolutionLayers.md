# Using Pooling Layers And Convolution Layers

Pooling layers and convolution layers allow use to capture spatial information from the data. These layers can be confusing to use at first, but in this tutorial, we will show you how those layers works.

## Requirements

* An understanding on these two tutorials:

  * [General Tensor Conventions](GeneralTensorConventions.md)

  * [Spatial Dimension, Kernel And Stride](SpatialDimensionKernelAndStride.md)

## The Pooling Layers

We will first create our input tensor as shown below for the purpose of this tutorial.

```lua

local inputTensor = TensorL.createRandomNormalTensor{{20, 3, 10, 10}} -- Creating a 4D tensor with the size of 20 x 3 x 10 x 10. 

```

In here, we can see that we have created a 4D tensor. This is because:

* The first dimension is used for the number of data.

* The second dimension is used for the number of channels.

The last two dimensions are used for the kernel dimensions, where the average pooling 2D layer requires these dimensions to get the average input value for the output value. Note that if you use average pooling 1D, you only need one kernel dimension.

Once the input tensor and the average pooling 2D layer is created, we can transform the input tensor into output tensor.

```lua

local outputTensor = DataPredict.PoolingLayers.AveragePooling2D{tensor = inputTensor, kernelDimensionSizeArray = {2, 2}, strideDimensionSizeArray = {2, 2}}

local outputTensorDimensionSizeArray = TensorL:getDimensionSizeArray(outputTensor)

print(outputTensorDimensionSizeArray) -- This would be a 4D tensor with the size of 20 x 3 x 5 x 5.

```

From here, we can observe that the first two dimension sizes remain the same. This is because the pooling operation generally affects the dimensions after the second one.

## The Convolution Layers

The convolution layers generally behaves the same as the pooling layers. However, the difference is that the convolution layers will change the number of channels. 

Below, we will on demonstrate how the way we set up the convolution layers affects the number of channels. Additionally, We will also use the same input tensor that we have used for the pooling layer.

For this, we need our weight tensor first.

```

local weightTensor = TensorL.createRandomNormalTensor{{7, 3, 2, 2}}

```

In here, the first dimension is the number of kernels. This means that this will the output tensor channel size once the input tensor passes through the convolutional layer.

Now, in order for the convolution layer to work properly, the weight tensor dimension size in the second dimension must match to the input tensor dimension size in the second dimension.

After that, the dimensions after the second dimension are the kernel dimension size, which determines the dimension sizes of each kernels at the first dimension of the output tensor.

Below, we will demonstrate on how the input tensor would be changed when it is passed through the convolution layer.

```lua

local outputTensor = Convolution2D{tensor = inputTensor, weightTensor = weightTensor, strideDimensionSizeArray = {2, 2}}

local outputTensorDimensionSizeArray = TensorL:getDimensionSizeArray(outputTensor)

print(outputTensorDimensionSizeArray) -- This would be a 4D tensor with the size of 20 x 7 x 5 x 5.

```

As you can see from the above, the number of channels changes from 3 to 7. The reasoning behind this change is that the convolution layers will attempt to extract 7 different filters from the input channel.

# Conclusion

The pooling layers and convolution layers are important parts for the convolutional neural networks. These layers allow you to extract useful features and could be used to reduce the size of the input tensor, potentially leading to faster training times.

That's all for today!
