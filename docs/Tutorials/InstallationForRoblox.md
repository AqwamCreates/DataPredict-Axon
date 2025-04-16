# Getting Started

To start, we must first link our deep learning library with our tensor library. However, you must use "Aqwam's Tensor Library" as every calculations made by our models are based on that tensor library.

## Library Download Links 

### Deep Learning Library

| Name                                        | Beta Version                                                                                            | Release Version                                                                                          |
|---------------------------------------------|---------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------|
| Deep Learning Library (DataPredict Axon)    | [0.0.0](https://github.com/AqwamCreates/DataPredict-Axon/blob/main/module_scripts/DataPredictAxon.rbxm) |[1.0.0](https://github.com/AqwamCreates/DataPredict-Axon/blob/main/module_scripts/DataPredict%20Axon%20-%20Release%201.0.rbxm)   |

### Tensor Library

| Name                                                                          | Version
|-------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------|
| Tensor Library (TensorL Nested)                                               | [0.9.0](https://github.com/AqwamCreates/TensorL/blob/main/TensorL_Table_Nested.lua)                            |
| Tensor Library - Efficient (TensorL Nested Efficient)                         | [0.9.0](https://github.com/AqwamCreates/TensorL/blob/main/TensorL_Table_Nested_Efficient.lua)                  |
| Tensor Library - Efficient (TensorL Nested Efficient Version 2)               | [0.9.0](https://github.com/AqwamCreates/TensorL/blob/main/TensorL_Table_Nested_Efficient_Version_2.lua)        |

Note: Tensor Library - Efficient Version 2 has the most consistent high performance for all tensor operations. So choose the last one if you prefer speed over code readability.

You can read the Terms And Conditions for the TensorL Library [here](https://github.com/AqwamCreates/TensorL/blob/main/docs/TermsAndConditions.md).

## Installing The Files Into Roblox Studio

To download the files from GitHub, you must click on the download button highlighted in the red box.

![Github File Download Screenshot](https://github.com/AqwamCreates/DataPredict/assets/67371914/b921d568-81b9-4f47-8a96-e0ab0316a4fe)

Then drag the files into Roblox Studio from a file explorer of your choice.

Once you put those two libraries into your game, make sure you link the Deep Learning Library with the Tensor Library. This can be done via setting the “AqwamTensorLibraryLinker” value (under the Deep Learning library) to the Tensor Library.

![Screenshot 2025-04-13 035519](https://github.com/user-attachments/assets/a9e48d14-608f-42bd-9eed-d2e6ea1d8b33)

Next, we will use require() function to our deep learning library:

```lua

local DataPredictAxon = require(DataPredictAxon) 

```
