--[[

	--------------------------------------------------------------------

	Aqwam's Deep Learning Library (DataPredict Neural)

	Author: Aqwam Harish Aiman
	
	Email: aqwam.harish.aiman@gmail.com
	
	YouTube: https://www.youtube.com/channel/UCUrwoxv5dufEmbGsxyEUPZw
	
	LinkedIn: https://www.linkedin.com/in/aqwam-harish-aiman/
	
	--------------------------------------------------------------------
		
	By using this library, you agree to comply with our Terms and Conditions in the link below:
	
	https://github.com/AqwamCreates/DataPredict-Neural/blob/main/docs/TermsAndConditions.md
	
	--------------------------------------------------------------------
	
	DO NOT REMOVE THIS TEXT!
	
	--------------------------------------------------------------------

--]]

local AqwamTensorLibrary = require(script.Parent.AqwamTensorLibraryLinker.Value)

local AutomaticDifferentiationTensor = require(script.Parent.AutomaticDifferentiationTensor)

local PoolingLayers = {}

local defaultKernelDimensionSize = 2

local default2DKernelDimensionSizeArray = {2, 2}

local default3DKernelDimensionSizeArray = {2, 2, 2}

local defaultStrideDimensionSize = 1

local default2DStrideDimensionSizeArray = {1, 1}

local default3DStrideDimensionSizeArray = {1, 1, 1}

local defaultUnpoolingMethod = "NearestNeighbour"

local unpooling1DMethodFunctionList = {

	["BedOfNails"] = function(tensor, value, a, b, startC, endC)

		tensor[a][b][startC] = value 

	end,

	["NearestNeighbour"] = function(tensor, value, a, b, startC, endC)

		for c = startC, endC, 1 do

			tensor[a][b][c] = value 

		end

	end,

}

local unpooling1DMethodInverseFunctionList = {

	["BedOfNails"] = function(tensor, otherTensor, a, b, c, startC, endC)

		tensor[a][b][c] = otherTensor[startC]

	end,

	["NearestNeighbour"] = function(tensor, otherTensor, a, b, c, startC, endC)

		for x = startC, endC, 1 do

			tensor[a][b][c] = tensor[a][b][c] + otherTensor[a][b][x]

		end

	end,

}

local unpooling2DMethodFunctionList = {

	["BedOfNails"] = function(tensor, value, a, b, startC, startD, endC, endD)

		tensor[a][b][startC][startD] = value 

	end,

	["NearestNeighbour"] = function(tensor, value, a, b, startC, startD, endC, endD)

		for c = startC, endC, 1 do

			for d = startD, endD, 1 do

				tensor[a][b][c][d] = value 

			end

		end

	end,

}

local unpooling2DMethodInverseFunctionList = {

	["BedOfNails"] = function(tensor, otherTensor, a, b, c, d, startC, startD, endC, endD)

		tensor[a][b][c][d] = otherTensor[startC][startD]

	end,

	["NearestNeighbour"] = function(tensor, otherTensor, a, b, c, d, startC, startD, endC, endD)

		for x = startC, endC, 1 do

			for y = startD, endD, 1 do

				tensor[a][b][c][d] = tensor[a][b][c][d] + otherTensor[a][b][x][y]

			end

		end

	end,

}

local unpooling3DMethodFunctionList = {

	["BedOfNails"] = function(tensor, value, a, b, startC, startD, startE, endC, endD, endE)

		tensor[a][b][startC][startD][startE] = value 

	end,

	["NearestNeighbour"] = function(tensor, value, a, b, startC, startD, startE, endC, endD, endE)

		for c = startC, endC, 1 do

			for d = startD, endD, 1 do

				for e = startE, endE, 1 do

					tensor[a][b][c][d][e] = value 

				end

			end

		end

	end,

}

local unpooling3DMethodInverseFunctionList = {

	["BedOfNails"] = function(tensor, otherTensor, a, b, c, d, e, startC, startD, startE, endC, endD, endE)

		tensor[a][b][c][d][e] = otherTensor[startC][startD][startE]

	end,

	["NearestNeighbour"] = function(tensor, otherTensor, a, b, c, d, e, startC, startD, startE, endC, endD, endE)

		for x = startC, endC, 1 do

			for y = startD, endD, 1 do

				for z = startE, endE, 1 do

					tensor[a][b][c][d][e] = tensor[a][b][c][d][e] + otherTensor[a][b][x][y][z]

				end

			end

		end

	end,

}
function PoolingLayers.MaximumUnpooling1D(parameterDictionary)
	
	local tensor = parameterDictionary.tensor or parameterDictionary[1]

	local kernelDimensionSize = parameterDictionary.kernelDimensionSize or parameterDictionary[2] or defaultKernelDimensionSize

	local strideDimensionSize = parameterDictionary.strideDimensionSize or parameterDictionary[3] or defaultStrideDimensionSize

	local unpoolingMethod = parameterDictionary.unpoolingMethod or parameterDictionary[4] or defaultUnpoolingMethod

	local tensorDimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(tensor)

	local numberOfDimensions = #tensorDimensionSizeArray

	if (numberOfDimensions ~= 3) then error("Unable to pass the input tensor to the 1D spatial maximum unpooling function block. The number of dimensions of the input tensor does not equal to 3. The input tensor have " .. numberOfDimensions .. " dimensions.") end

	local unpoolingMethodFunction = unpooling1DMethodFunctionList[unpoolingMethod]

	if (not unpoolingMethodFunction) then error("Invalid unpooling method.") end

	local resultTensorDimensionSizeArray = table.clone(tensorDimensionSizeArray)

	local inputDimensionSize = tensorDimensionSizeArray[3]

	local outputDimensionSize = (inputDimensionSize - 1) * strideDimensionSize + kernelDimensionSize

	resultTensorDimensionSizeArray[3] = outputDimensionSize

	local tensorDimension1Size = tensorDimensionSizeArray[1]

	local tensorDimension2Size = tensorDimensionSizeArray[2]

	local tensorDimension3Size = tensorDimensionSizeArray[3]

	local resultTensor = AqwamTensorLibrary:createTensor(resultTensorDimensionSizeArray, 0)

	for a = 1, tensorDimension1Size, 1 do

		for b = 1, tensorDimension2Size, 1 do

			for c = 1, tensorDimension3Size, 1 do

				local value = tensor[a][b][c]

				local startC = (c - 1) * strideDimensionSize + 1

				local endC = startC + kernelDimensionSize - 1

				unpoolingMethodFunction(resultTensor, value, a, b, c, startC, endC)

			end

		end

	end

	local PartialDerivativeFunction = function(derivativeTensor)

		local unpoolingMethodInverseFunction = unpooling1DMethodInverseFunctionList[unpoolingMethod]

		if (not unpoolingMethodInverseFunction) then error("Invalid unpooling method.") end

		local derivativeTensorSizeArray = AqwamTensorLibrary:getDimensionSizeArray(derivativeTensor)

		local chainRuleFirstDerivativeTensor = AqwamTensorLibrary:createTensor(tensorDimensionSizeArray)

		for a = 1, tensorDimension1Size, 1 do

			for b = 1, tensorDimension2Size, 1 do

				for c = 1, tensorDimension3Size, 1 do

					local startC = (c - 1) * strideDimensionSize + 1

					local endC = startC + kernelDimensionSize - 1

					unpoolingMethodInverseFunction(chainRuleFirstDerivativeTensor, derivativeTensor, a, b, c, startC, endC)

				end

			end

		end

		tensor:differentiate(chainRuleFirstDerivativeTensor)

	end

	return AutomaticDifferentiationTensor.new({resultTensor, PartialDerivativeFunction, {tensor}})

end

function PoolingLayers.MaximumUnpooling2D(parameterDictionary)
	
	local tensor = parameterDictionary.tensor or parameterDictionary[1]

	local kernelDimensionSizeArray = parameterDictionary.kernelDimensionSizeArray or parameterDictionary[2] or default2DKernelDimensionSizeArray

	local strideDimensionSizeArray = parameterDictionary.strideDimensionSizeArray or parameterDictionary[3] or default2DStrideDimensionSizeArray

	local unpoolingMethod = parameterDictionary.unpoolingMethod or parameterDictionary[4] or defaultUnpoolingMethod

	local tensorDimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(tensor)

	local numberOfDimensions = #tensorDimensionSizeArray

	if (numberOfDimensions ~= 4) then error("Unable to pass the input tensor to the 2D spatial maximum unpooling function block. The number of dimensions of the input tensor does not equal to 4. The input tensor have " .. numberOfDimensions .. " dimensions.") end

	local unpoolingMethodFunction = unpooling2DMethodFunctionList[unpoolingMethod]

	if (not unpoolingMethodFunction) then error("Invalid unpooling method.") end

	local resultTensorDimensionSizeArray = table.clone(tensorDimensionSizeArray)

	for dimension = 1, 2, 1 do

		local inputDimensionSize = tensorDimensionSizeArray[dimension + 2]

		local outputDimensionSize = (inputDimensionSize - 1) * strideDimensionSizeArray[dimension] + kernelDimensionSizeArray[dimension]

		resultTensorDimensionSizeArray[dimension + 2] = outputDimensionSize

	end

	local tensorDimension1Size = tensorDimensionSizeArray[1]

	local tensorDimension2Size = tensorDimensionSizeArray[2]

	local tensorDimension3Size = tensorDimensionSizeArray[3]

	local tensorDimension4Size = tensorDimensionSizeArray[4]

	local kernelDimension1Size = kernelDimensionSizeArray[1]

	local kernelDimension2Size = kernelDimensionSizeArray[2]

	local strideDimension1Size = strideDimensionSizeArray[1]

	local strideDimension2Size = strideDimensionSizeArray[2]

	local resultTensor = AqwamTensorLibrary:createTensor(resultTensorDimensionSizeArray, 0)

	for a = 1, tensorDimension1Size, 1 do

		for b = 1, tensorDimension2Size, 1 do

			for c = 1, tensorDimension3Size, 1 do

				for d = 1, tensorDimension4Size, 1 do

					local value = tensor[a][b][c][d]

					local startC = (c - 1) * strideDimension1Size + 1
					local startD = (d - 1) * strideDimension2Size + 1

					local endC = startC + kernelDimension1Size - 1
					local endD = startD + kernelDimension2Size - 1

					unpoolingMethodFunction(resultTensor, value, a, b, c, d, startC, startD, endC, endD)

				end

			end

		end

	end

	local PartialDerivativeFunction = function(derivativeTensor)

		local unpoolingMethodInverseFunction = unpooling2DMethodInverseFunctionList[unpoolingMethod]

		if (not unpoolingMethodInverseFunction) then error("Invalid unpooling method.") end

		local chainRuleFirstDerivativeTensor = AqwamTensorLibrary:createTensor(tensorDimensionSizeArray)

		for a = 1, tensorDimension1Size, 1 do

			for b = 1, tensorDimension2Size, 1 do

				for c = 1, tensorDimension3Size, 1 do

					for d = 1, tensorDimension4Size, 1 do

						local startC = (c - 1) * strideDimension1Size + 1
						local startD = (d - 1) * strideDimension2Size + 1

						local endC = startC + kernelDimension1Size - 1
						local endD = startD + kernelDimension2Size - 1

						unpoolingMethodInverseFunction(chainRuleFirstDerivativeTensor, derivativeTensor, a, b, c, d, startC, startD, endC, endD)

					end

				end

			end

		end

		tensor:differentiate(chainRuleFirstDerivativeTensor)

	end

	return AutomaticDifferentiationTensor.new({resultTensor, PartialDerivativeFunction, {tensor}})

end

function PoolingLayers.MaximumUnpooling3D(parameterDictionary)

	local tensor = parameterDictionary.tensor or parameterDictionary[1]

	local kernelDimensionSizeArray = parameterDictionary.kernelDimensionSizeArray or parameterDictionary[2] or default3DKernelDimensionSizeArray

	local strideDimensionSizeArray = parameterDictionary.strideDimensionSizeArray or parameterDictionary[3] or default3DStrideDimensionSizeArray

	local unpoolingMethod = parameterDictionary.unpoolingMethod  or parameterDictionary[4] or defaultUnpoolingMethod

	local tensorDimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(tensor)

	local numberOfDimensions = #tensorDimensionSizeArray

	if (numberOfDimensions ~= 5) then error("Unable to pass the input tensor to the 3D spatial maximum unpooling function block. The number of dimensions of the input tensor does not equal to 5. The input tensor have " .. numberOfDimensions .. " dimensions.") end

	local unpoolingMethodFunction = unpooling3DMethodFunctionList[unpoolingMethod]

	if (not unpoolingMethodFunction) then error("Invalid unpooling method.") end

	local resultTensorDimensionSizeArray = table.clone(tensorDimensionSizeArray)

	for dimension = 1, 3, 1 do

		local inputDimensionSize = tensorDimensionSizeArray[dimension + 2]

		local outputDimensionSize = (inputDimensionSize - 1) * strideDimensionSizeArray[dimension] + kernelDimensionSizeArray[dimension]

		resultTensorDimensionSizeArray[dimension + 2] = outputDimensionSize

	end

	local tensorDimension1Size = tensorDimensionSizeArray[1]

	local tensorDimension2Size = tensorDimensionSizeArray[2]

	local tensorDimension3Size = tensorDimensionSizeArray[3]

	local tensorDimension4Size = tensorDimensionSizeArray[4]

	local tensorDimension5Size = tensorDimensionSizeArray[5]

	local kernelDimension1Size = kernelDimensionSizeArray[1]

	local kernelDimension2Size = kernelDimensionSizeArray[2]

	local kernelDimension3Size = kernelDimensionSizeArray[3]

	local strideDimension1Size = strideDimensionSizeArray[1]

	local strideDimension2Size = strideDimensionSizeArray[2]

	local strideDimension3Size = strideDimensionSizeArray[3]

	local resultTensor = AqwamTensorLibrary:createTensor(resultTensorDimensionSizeArray, 0)

	for a = 1, tensorDimension1Size, 1 do

		for b = 1, tensorDimension2Size, 1 do

			for c = 1, tensorDimension3Size, 1 do

				for d = 1, tensorDimension4Size, 1 do

					for e = 1, tensorDimension5Size, 1 do

						local value = tensor[a][b][c][d][e]

						local startC = (c - 1) * strideDimension1Size + 1
						local startD = (d - 1) * strideDimension2Size + 1
						local startE = (e - 1) * strideDimension3Size + 1

						local endC = startC + kernelDimension1Size - 1
						local endD = startD + kernelDimension2Size - 1
						local endE = startE + kernelDimension3Size - 1

						unpoolingMethodFunction(resultTensor, value, a, b, c, d, e, startC, startD, startE, endC, endD, endE)

					end

				end

			end

		end

	end

	local PartialDerivativeFunction = function(derivativeTensor)

		local unpoolingMethodInverseFunction = unpooling3DMethodInverseFunctionList[unpoolingMethod]

		if (not unpoolingMethodInverseFunction) then error("Invalid unpooling method.") end

		local chainRuleFirstDerivativeTensor = AqwamTensorLibrary:createTensor(tensorDimensionSizeArray)

		for a = 1, tensorDimension1Size, 1 do

			for b = 1, tensorDimension2Size, 1 do

				for c = 1, tensorDimension3Size, 1 do

					for d = 1, tensorDimension4Size, 1 do

						for e = 1, tensorDimension5Size, 1 do

							local startC = (c - 1) * strideDimension1Size + 1
							local startD = (d - 1) * strideDimension2Size + 1
							local startE = (e - 1) * strideDimension3Size + 1

							local endC = startC + kernelDimension1Size - 1
							local endD = startD + kernelDimension2Size - 1
							local endE = startE + kernelDimension3Size - 1

							unpoolingMethodInverseFunction(chainRuleFirstDerivativeTensor, derivativeTensor, a, b, c, d, e, startC, startD, startE, endC, endD, endE)

						end

					end

				end

			end

		end

		tensor:differentiate(chainRuleFirstDerivativeTensor)

	end

	return AutomaticDifferentiationTensor.new({resultTensor, PartialDerivativeFunction, {tensor}})

end

function PoolingLayers.AveragePooling1D(tensor, kernelDimensionSize, strideDimensionSize)

	kernelDimensionSize = kernelDimensionSize or defaultKernelDimensionSize

	strideDimensionSize = strideDimensionSize or defaultStrideDimensionSize

	local tensorDimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(tensor)

	local numberOfDimensions = #tensorDimensionSizeArray

	if (numberOfDimensions ~= 3) then error("Unable to pass the input tensor to the 1D spatial average pooling function block. The number of dimensions of the input tensor does not equal to 3. The input tensor have " .. numberOfDimensions .. " dimensions.") end

	local resultTensorDimensionSizeArray = table.clone(tensorDimensionSizeArray)

	local inputDimensionSize = tensorDimensionSizeArray[3]

	local outputDimensionSize = ((inputDimensionSize - kernelDimensionSize) / strideDimensionSize) + 1

	resultTensorDimensionSizeArray[3] = math.floor(outputDimensionSize)

	local resultTensorDimension1Size = resultTensorDimensionSizeArray[1]

	local resultTensorDimension2Size = resultTensorDimensionSizeArray[2]

	local resultTensorDimension3Size = resultTensorDimensionSizeArray[3]

	local resultTensor = AqwamTensorLibrary:createTensor(resultTensorDimensionSizeArray)

	for a = 1, resultTensorDimension1Size, 1 do

		for b = 1, resultTensorDimension2Size, 1 do

			for c = 1, resultTensorDimension3Size, 1 do

				local subTensor = tensor[a][b]

				local originDimensionIndexArray = {(c - 1) * strideDimensionSize + 1}

				local targetDimensionIndexArray = {(c - 1) * strideDimensionSize + kernelDimensionSize}

				local extractedSubTensor = AqwamTensorLibrary:extract(subTensor, originDimensionIndexArray, targetDimensionIndexArray)

				local averageValue = AqwamTensorLibrary:mean(extractedSubTensor)

				resultTensor[a][b][c] = averageValue

			end

		end

	end

	local PartialDerivativeFunction = function(derivativeTensor)

		if (not AutomaticDifferentiationTensor:checkIfIsAutomaticDifferentiationTensor(tensor)) then return end 

		local chainRuleFirstDerivativeTensor = AqwamTensorLibrary:createTensor(tensorDimensionSizeArray)

		for a = 1, resultTensorDimension1Size, 1 do

			for b = 1, resultTensorDimension2Size, 1 do

				for c = 1, resultTensorDimension3Size, 1 do

					local derivativeValue = derivativeTensor[a][b][c]

					local originDimensionIndexArray = {(c - 1) * strideDimensionSize + 1}

					local targetDimensionIndexArray = {(c - 1) * strideDimensionSize + kernelDimensionSize}

					for x = originDimensionIndexArray[1], targetDimensionIndexArray[1], 1 do

						chainRuleFirstDerivativeTensor[a][b][x] = chainRuleFirstDerivativeTensor[a][b][x] + derivativeValue

					end

				end

			end

		end

		chainRuleFirstDerivativeTensor = AqwamTensorLibrary:divide(chainRuleFirstDerivativeTensor, inputDimensionSize)

		tensor:differentiate(chainRuleFirstDerivativeTensor)

	end

	return AutomaticDifferentiationTensor.new(resultTensor, PartialDerivativeFunction, {tensor})

end

function PoolingLayers.AveragePooling2D(tensor, kernelDimensionSizeArray, strideDimensionSizeArray)

	kernelDimensionSizeArray = kernelDimensionSizeArray or default2DKernelDimensionSizeArray

	strideDimensionSizeArray = strideDimensionSizeArray or default2DStrideDimensionSizeArray

	local tensorDimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(tensor)

	local numberOfDimensions = #tensorDimensionSizeArray

	if (numberOfDimensions ~= 4) then error("Unable to pass the input tensor to the 2D spatial average pooling function block. The number of dimensions of the input tensor does not equal to 4. The input tensor have " .. numberOfDimensions .. " dimensions.") end

	local resultTensorDimensionSizeArray = table.clone(tensorDimensionSizeArray)

	for dimension = 1, 2, 1 do

		local inputDimensionSize = tensorDimensionSizeArray[dimension + 2]

		local outputDimensionSize = ((inputDimensionSize - kernelDimensionSizeArray[dimension]) / strideDimensionSizeArray[dimension]) + 1

		resultTensorDimensionSizeArray[dimension + 2] = math.floor(outputDimensionSize)

	end

	local resultTensorDimension1Size = resultTensorDimensionSizeArray[1]

	local resultTensorDimension2Size = resultTensorDimensionSizeArray[2]

	local resultTensorDimension3Size = resultTensorDimensionSizeArray[3]

	local resultTensorDimension4Size = resultTensorDimensionSizeArray[4]

	local kernelDimension1Size = kernelDimensionSizeArray[1]

	local kernelDimension2Size = kernelDimensionSizeArray[2]

	local strideDimension1Size = strideDimensionSizeArray[1]

	local strideDimension2Size = strideDimensionSizeArray[2]

	local resultTensor = AqwamTensorLibrary:createTensor(resultTensorDimensionSizeArray)

	for a = 1, resultTensorDimension1Size, 1 do

		for b = 1, resultTensorDimension2Size, 1 do

			for c = 1, resultTensorDimension3Size, 1 do

				for d = 1, resultTensorDimension4Size, 1 do

					local subTensor = tensor[a][b]

					local originDimensionIndexArray = {(c - 1) * strideDimension1Size + 1, (d - 1) * strideDimension2Size + 1}

					local targetDimensionIndexArray = {(c - 1) * strideDimension1Size + kernelDimension1Size, (d - 1) * strideDimension2Size + kernelDimension2Size}

					local extractedSubTensor = AqwamTensorLibrary:extract(subTensor, originDimensionIndexArray, targetDimensionIndexArray)

					local averageValue = AqwamTensorLibrary:mean(extractedSubTensor)

					resultTensor[a][b][c][d] = averageValue

				end

			end

		end

	end

	local PartialDerivativeFunction = function(derivativeTensor)

		if (not AutomaticDifferentiationTensor:checkIfIsAutomaticDifferentiationTensor(tensor)) then return end 

		local derivativeTensorSizeArray = AqwamTensorLibrary:getDimensionSizeArray(derivativeTensor)

		local chainRuleFirstDerivativeTensor = AqwamTensorLibrary:createTensor(tensorDimensionSizeArray)

		local kernelArea = kernelDimension1Size * kernelDimension2Size

		for a = 1, resultTensorDimension1Size, 1 do

			for b = 1, resultTensorDimension2Size, 1 do

				for c = 1, resultTensorDimension3Size, 1 do

					for d = 1, resultTensorDimension4Size, 1 do

						local derivativeValue = derivativeTensor[a][b][c][d]

						local originDimensionIndexArray = {(c - 1) * strideDimension1Size + 1, (d - 1) * strideDimension2Size + 1}

						local targetDimensionIndexArray = {(c - 1) * strideDimension1Size + kernelDimension1Size, (d - 1) * strideDimension2Size + kernelDimension2Size}

						for x = originDimensionIndexArray[1], targetDimensionIndexArray[1], 1 do

							for y = originDimensionIndexArray[2], targetDimensionIndexArray[2], 1 do

								chainRuleFirstDerivativeTensor[a][b][x][y] = chainRuleFirstDerivativeTensor[a][b][x][y] + derivativeValue

							end

						end

					end

				end

			end

		end

		chainRuleFirstDerivativeTensor = AqwamTensorLibrary:divide(chainRuleFirstDerivativeTensor, kernelArea)

		tensor:differentiate(chainRuleFirstDerivativeTensor)

	end

	return AutomaticDifferentiationTensor.new(resultTensor, PartialDerivativeFunction, {tensor})

end

function PoolingLayers.AveragePooling3D(tensor, kernelDimensionSizeArray, strideDimensionSizeArray)

	kernelDimensionSizeArray = kernelDimensionSizeArray or default3DKernelDimensionSizeArray

	strideDimensionSizeArray = strideDimensionSizeArray or default3DStrideDimensionSizeArray

	local tensorDimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(tensor)

	local numberOfDimensions = #tensorDimensionSizeArray

	if (numberOfDimensions ~= 5) then error("Unable to pass the input tensor to the 3D spatial average pooling function block. The number of dimensions of the input tensor does not equal to 5. The input tensor have " .. numberOfDimensions .. " dimensions.") end

	local resultTensorDimensionSizeArray = table.clone(tensorDimensionSizeArray)

	for dimension = 1, 3, 1 do

		local inputDimensionSize = tensorDimensionSizeArray[dimension + 2]

		local outputDimensionSize = ((inputDimensionSize - kernelDimensionSizeArray[dimension]) / strideDimensionSizeArray[dimension]) + 1

		resultTensorDimensionSizeArray[dimension + 2] = math.floor(outputDimensionSize)

	end

	local resultTensorDimension1Size = resultTensorDimensionSizeArray[1]

	local resultTensorDimension2Size = resultTensorDimensionSizeArray[2]

	local resultTensorDimension3Size = resultTensorDimensionSizeArray[3]

	local resultTensorDimension4Size = resultTensorDimensionSizeArray[4]

	local resultTensorDimension5Size = resultTensorDimensionSizeArray[5]

	local kernelDimension1Size = kernelDimensionSizeArray[1]

	local kernelDimension2Size = kernelDimensionSizeArray[2]

	local kernelDimension3Size = kernelDimensionSizeArray[3]

	local strideDimension1Size = strideDimensionSizeArray[1]

	local strideDimension2Size = strideDimensionSizeArray[2]

	local strideDimension3Size = strideDimensionSizeArray[3]

	local resultTensor = AqwamTensorLibrary:createTensor(resultTensorDimensionSizeArray)

	for a = 1, resultTensorDimension1Size, 1 do

		for b = 1, resultTensorDimension2Size, 1 do

			for c = 1, resultTensorDimension3Size, 1 do

				for d = 1, resultTensorDimension4Size, 1 do

					for e = 1, resultTensorDimension5Size, 1 do

						local subTensor = tensor[a][b]

						local originDimensionIndexArray = {(c - 1) * strideDimension1Size + 1, (d - 1) * strideDimension2Size + 1, (e - 1) * strideDimension3Size + 1}

						local targetDimensionIndexArray = {(c - 1) * strideDimension1Size + kernelDimension1Size, (d - 1) * strideDimension2Size + kernelDimension2Size, (e - 1) * strideDimension3Size + kernelDimension3Size}

						local extractedSubTensor = AqwamTensorLibrary:extract(subTensor, originDimensionIndexArray, targetDimensionIndexArray)

						local averageValue = AqwamTensorLibrary:mean(extractedSubTensor)

						resultTensor[a][b][c][d][e] = averageValue

					end

				end

			end

		end

	end

	local PartialDerivativeFunction = function(derivativeTensor)

		if (not AutomaticDifferentiationTensor:checkIfIsAutomaticDifferentiationTensor(tensor)) then return end 

		local chainRuleFirstDerivativeTensor = AqwamTensorLibrary:createTensor(tensorDimensionSizeArray)

		local kernelVolume = kernelDimension1Size * kernelDimension2Size * kernelDimension3Size

		for a = 1, resultTensorDimension1Size, 1 do

			for b = 1, resultTensorDimension2Size, 1 do

				for c = 1, resultTensorDimension3Size, 1 do

					for d = 1, resultTensorDimension4Size, 1 do

						for e = 1, resultTensorDimension5Size, 1 do

							local derivativeValue = derivativeTensor[a][b][c][d][e]

							local originDimensionIndexArray = {(c - 1) * strideDimension1Size + 1, (d - 1) * strideDimension2Size + 1, (e - 1) * strideDimension3Size + 1}

							local targetDimensionIndexArray = {(c - 1) * strideDimension1Size + kernelDimension1Size, (d - 1) * strideDimension2Size + kernelDimension2Size, (e - 1) * strideDimension3Size + kernelDimension3Size}

							for x = originDimensionIndexArray[1], targetDimensionIndexArray[1], 1 do

								for y = originDimensionIndexArray[2], targetDimensionIndexArray[2], 1 do

									for z = originDimensionIndexArray[3], targetDimensionIndexArray[3], 1 do

										chainRuleFirstDerivativeTensor[a][b][x][y][z] = chainRuleFirstDerivativeTensor[a][b][x][y][z] + derivativeValue

									end

								end

							end

						end

					end

				end

			end

		end

		chainRuleFirstDerivativeTensor = AqwamTensorLibrary:divide(chainRuleFirstDerivativeTensor, kernelVolume)

		tensor:differentiate(chainRuleFirstDerivativeTensor)

	end

	return AutomaticDifferentiationTensor.new(resultTensor, PartialDerivativeFunction, {tensor})

end

function PoolingLayers.MinimumPooling1D(tensor, kernelDimensionSize, strideDimensionSize)

	kernelDimensionSize = kernelDimensionSize or defaultKernelDimensionSize

	strideDimensionSize = strideDimensionSize or defaultStrideDimensionSize

	local tensorDimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(tensor)

	local numberOfDimensions = #tensorDimensionSizeArray

	if (numberOfDimensions ~= 3) then error("Unable to pass the input tensor to the 1D spatial minimum pooling function block. The number of dimensions of the input tensor does not equal to 3. The input tensor have " .. numberOfDimensions .. " dimensions.") end

	local resultTensorDimensionSizeArray = table.clone(tensorDimensionSizeArray)

	local inputDimensionSize = tensorDimensionSizeArray[3]

	local outputDimensionSize = ((inputDimensionSize - kernelDimensionSize) / strideDimensionSize) + 1

	resultTensorDimensionSizeArray[3] = math.floor(outputDimensionSize)

	local resultTensorDimension1Size = resultTensorDimensionSizeArray[1]

	local resultTensorDimension2Size = resultTensorDimensionSizeArray[2]

	local resultTensorDimension3Size = resultTensorDimensionSizeArray[3]

	local resultTensor = AqwamTensorLibrary:createTensor(resultTensorDimensionSizeArray)

	for a = 1, resultTensorDimension1Size, 1 do

		for b = 1, resultTensorDimension2Size, 1 do

			for c = 1, resultTensorDimension3Size, 1 do

				local subTensor = tensor[a][b]

				local originDimensionIndexArray = {(c - 1) * strideDimensionSize + 1}

				local targetDimensionIndexArray = {(c - 1) * strideDimensionSize + kernelDimensionSize}

				local extractedSubTensor = AqwamTensorLibrary:extract(subTensor, originDimensionIndexArray, targetDimensionIndexArray)

				local minimumValue = AqwamTensorLibrary:findMinimumValue(extractedSubTensor)

				resultTensor[a][b][c] = minimumValue

			end

		end

	end

	local PartialDerivativeFunction = function(derivativeTensor)

		if (not AutomaticDifferentiationTensor:checkIfIsAutomaticDifferentiationTensor(tensor)) then return end

		local chainRuleFirstDerivativeTensor = AqwamTensorLibrary:createTensor(tensorDimensionSizeArray)

		for a = 1, resultTensorDimension1Size, 1 do

			for b = 1, resultTensorDimension2Size, 1 do

				for c = 1, resultTensorDimension3Size, 1 do

					local derivativeValue = derivativeTensor[a][b][c]

					local originDimensionIndexArray = {(c - 1) * strideDimensionSize + 1}

					local targetDimensionIndexArray = {(c - 1) * strideDimensionSize + kernelDimensionSize}

					for x = originDimensionIndexArray[1], targetDimensionIndexArray[1], 1 do

						if (resultTensor[a][b][c] == tensor[a][b][x]) then chainRuleFirstDerivativeTensor[a][b][x] = chainRuleFirstDerivativeTensor[a][b][x] + derivativeValue end

					end

				end

			end

		end

		tensor:differentiate(chainRuleFirstDerivativeTensor)

	end

	return AutomaticDifferentiationTensor.new(resultTensor, PartialDerivativeFunction, {tensor})

end

function PoolingLayers.MinimumPooling2D(tensor, kernelDimensionSizeArray, strideDimensionSizeArray)

	kernelDimensionSizeArray = kernelDimensionSizeArray or default2DKernelDimensionSizeArray

	strideDimensionSizeArray = strideDimensionSizeArray or default2DStrideDimensionSizeArray

	local tensorDimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(tensor)

	local numberOfDimensions = #tensorDimensionSizeArray

	if (numberOfDimensions ~= 4) then error("Unable to pass the input tensor to the 2D spatial minimum pooling function block. The number of dimensions of the input tensor does not equal to 4. The input tensor have " .. numberOfDimensions .. " dimensions.") end

	local resultTensorDimensionSizeArray = table.clone(tensorDimensionSizeArray)

	for dimension = 1, 2, 1 do

		local inputDimensionSize = tensorDimensionSizeArray[dimension + 2]

		local outputDimensionSize = ((inputDimensionSize - kernelDimensionSizeArray[dimension]) / strideDimensionSizeArray[dimension]) + 1

		resultTensorDimensionSizeArray[dimension + 2] = math.floor(outputDimensionSize)

	end

	local resultTensorDimension1Size = resultTensorDimensionSizeArray[1]

	local resultTensorDimension2Size = resultTensorDimensionSizeArray[2]

	local resultTensorDimension3Size = resultTensorDimensionSizeArray[3]

	local resultTensorDimension4Size = resultTensorDimensionSizeArray[4]

	local kernelDimension1Size = kernelDimensionSizeArray[1]

	local kernelDimension2Size = kernelDimensionSizeArray[2]

	local strideDimension1Size = strideDimensionSizeArray[1]

	local strideDimension2Size = strideDimensionSizeArray[2]

	local resultTensor = AqwamTensorLibrary:createTensor(resultTensorDimensionSizeArray)

	for a = 1, resultTensorDimension1Size, 1 do

		for b = 1, resultTensorDimension2Size, 1 do

			for c = 1, resultTensorDimension3Size, 1 do

				for d = 1, resultTensorDimension4Size, 1 do

					local subTensor = tensor[a][b]

					local originDimensionIndexArray = {(c - 1) * strideDimension1Size + 1, (d - 1) * strideDimension2Size + 1}

					local targetDimensionIndexArray = {(c - 1) * strideDimension1Size + kernelDimension1Size, (d - 1) * strideDimension2Size + kernelDimension2Size}

					local extractedSubTensor = AqwamTensorLibrary:extract(subTensor, originDimensionIndexArray, targetDimensionIndexArray)

					local minimumValue = AqwamTensorLibrary:findMinimumValue(extractedSubTensor)

					resultTensor[a][b][c][d] = minimumValue

				end

			end

		end

	end

	local PartialDerivativeFunction = function(derivativeTensor)

		if (not AutomaticDifferentiationTensor:checkIfIsAutomaticDifferentiationTensor(tensor)) then return end 

		local derivativeTensorSizeArray = AqwamTensorLibrary:getDimensionSizeArray(derivativeTensor)

		local chainRuleFirstDerivativeTensor = AqwamTensorLibrary:createTensor(tensorDimensionSizeArray)

		for a = 1, resultTensorDimension1Size, 1 do

			for b = 1, resultTensorDimension2Size, 1 do

				for c = 1, resultTensorDimension3Size, 1 do

					for d = 1, resultTensorDimension4Size, 1 do

						local derivativeValue = derivativeTensor[a][b][c][d]

						local originDimensionIndexArray = {(c - 1) * strideDimension1Size + 1, (d - 1) * strideDimension2Size + 1}

						local targetDimensionIndexArray = {(c - 1) * strideDimension1Size + kernelDimension1Size, (d - 1) * strideDimension2Size + kernelDimension2Size}

						for x = originDimensionIndexArray[1], targetDimensionIndexArray[1], 1 do

							for y = originDimensionIndexArray[2], targetDimensionIndexArray[2], 1 do

								if (resultTensor[a][b][c][d] == tensor[a][b][x][y]) then chainRuleFirstDerivativeTensor[a][b][x][y] = chainRuleFirstDerivativeTensor[a][b][x][y] + derivativeValue end

							end

						end

					end

				end

			end

		end

		tensor:differentiate(chainRuleFirstDerivativeTensor)

	end

	return AutomaticDifferentiationTensor.new(resultTensor, PartialDerivativeFunction, {tensor})

end

function PoolingLayers.MinimumPooling3D(tensor, kernelDimensionSizeArray, strideDimensionSizeArray)

	kernelDimensionSizeArray = kernelDimensionSizeArray or default3DKernelDimensionSizeArray

	strideDimensionSizeArray = strideDimensionSizeArray or default3DStrideDimensionSizeArray

	local tensorDimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(tensor)

	local numberOfDimensions = #tensorDimensionSizeArray

	if (numberOfDimensions ~= 5) then error("Unable to pass the input tensor to the 3D spatial minimum pooling function block. The number of dimensions of the input tensor does not equal to 5. The input tensor have " .. numberOfDimensions .. " dimensions.") end

	local resultTensorDimensionSizeArray = table.clone(tensorDimensionSizeArray)

	for dimension = 1, 3, 1 do

		local inputDimensionSize = tensorDimensionSizeArray[dimension + 2]

		local outputDimensionSize = ((inputDimensionSize - kernelDimensionSizeArray[dimension]) / strideDimensionSizeArray[dimension]) + 1

		resultTensorDimensionSizeArray[dimension + 2] = math.floor(outputDimensionSize)

	end

	local resultTensorDimension1Size = resultTensorDimensionSizeArray[1]

	local resultTensorDimension2Size = resultTensorDimensionSizeArray[2]

	local resultTensorDimension3Size = resultTensorDimensionSizeArray[3]

	local resultTensorDimension4Size = resultTensorDimensionSizeArray[4]

	local resultTensorDimension5Size = resultTensorDimensionSizeArray[5]

	local kernelDimension1Size = kernelDimensionSizeArray[1]

	local kernelDimension2Size = kernelDimensionSizeArray[2]

	local kernelDimension3Size = kernelDimensionSizeArray[3]

	local strideDimension1Size = strideDimensionSizeArray[1]

	local strideDimension2Size = strideDimensionSizeArray[2]

	local strideDimension3Size = strideDimensionSizeArray[3]

	local resultTensor = AqwamTensorLibrary:createTensor(resultTensorDimensionSizeArray)

	for a = 1, resultTensorDimension1Size, 1 do

		for b = 1, resultTensorDimension2Size, 1 do

			for c = 1, resultTensorDimension3Size, 1 do

				for d = 1, resultTensorDimension4Size, 1 do

					for e = 1, resultTensorDimension5Size, 1 do

						local subTensor = tensor[a][b]

						local originDimensionIndexArray = {(c - 1) * strideDimension1Size + 1, (d - 1) * strideDimension2Size + 1, (e - 1) * strideDimension3Size + 1}

						local targetDimensionIndexArray = {(c - 1) * strideDimension1Size + kernelDimension1Size, (d - 1) * strideDimension2Size + kernelDimension2Size, (e - 1) * strideDimension3Size + kernelDimension3Size}

						local extractedSubTensor = AqwamTensorLibrary:extract(subTensor, originDimensionIndexArray, targetDimensionIndexArray)

						local minimumValue = AqwamTensorLibrary:findMinimumValue(extractedSubTensor)

						resultTensor[a][b][c][d][e] = minimumValue

					end

				end

			end

		end

	end

	local PartialDerivativeFunction = function(derivativeTensor)

		if (not AutomaticDifferentiationTensor:checkIfIsAutomaticDifferentiationTensor(tensor)) then return end 

		local chainRuleFirstDerivativeTensor = AqwamTensorLibrary:createTensor(tensorDimensionSizeArray)

		for a = 1, resultTensorDimension1Size, 1 do

			for b = 1, resultTensorDimension2Size, 1 do

				for c = 1, resultTensorDimension3Size, 1 do

					for d = 1, resultTensorDimension4Size, 1 do

						for e = 1, resultTensorDimension5Size, 1 do

							local derivativeValue = derivativeTensor[a][b][c][d][e]

							local originDimensionIndexArray = {(c - 1) * strideDimension1Size + 1, (d - 1) * strideDimension2Size + 1, (e - 1) * strideDimension3Size + 1}

							local targetDimensionIndexArray = {(c - 1) * strideDimension1Size + kernelDimension1Size, (d - 1) * strideDimension2Size + kernelDimension2Size, (e - 1) * strideDimension3Size + kernelDimension3Size}

							for x = originDimensionIndexArray[1], targetDimensionIndexArray[1], 1 do

								for y = originDimensionIndexArray[2], targetDimensionIndexArray[2], 1 do

									for z = originDimensionIndexArray[3], targetDimensionIndexArray[3], 1 do

										if (resultTensor[a][b][c][d][e] == tensor[a][b][x][y][z]) then chainRuleFirstDerivativeTensor[a][b][x][y][z] = chainRuleFirstDerivativeTensor[a][b][x][y][z] + derivativeValue end

									end

								end

							end

						end

					end

				end

			end

		end

		tensor:differentiate(chainRuleFirstDerivativeTensor)

	end

	return AutomaticDifferentiationTensor.new(resultTensor, PartialDerivativeFunction, {tensor})

end

function PoolingLayers.MaximumPooling1D(tensor, kernelDimensionSize, strideDimensionSize)

	kernelDimensionSize = kernelDimensionSize or defaultKernelDimensionSize

	strideDimensionSize = strideDimensionSize or defaultStrideDimensionSize

	local tensorDimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(tensor)

	local numberOfDimensions = #tensorDimensionSizeArray

	if (numberOfDimensions ~= 3) then error("Unable to pass the input tensor to the 1D spatial maximum pooling function block. The number of dimensions of the input tensor does not equal to 3. The input tensor have " .. numberOfDimensions .. " dimensions.") end

	local resultTensorDimensionSizeArray = table.clone(tensorDimensionSizeArray)

	local inputDimensionSize = tensorDimensionSizeArray[3]

	local outputDimensionSize = ((inputDimensionSize - kernelDimensionSize) / strideDimensionSize) + 1

	resultTensorDimensionSizeArray[3] = math.floor(outputDimensionSize)

	local resultTensorDimension1Size = resultTensorDimensionSizeArray[1]

	local resultTensorDimension2Size = resultTensorDimensionSizeArray[2]

	local resultTensorDimension3Size = resultTensorDimensionSizeArray[3]

	local resultTensor = AqwamTensorLibrary:createTensor(resultTensorDimensionSizeArray)

	for a = 1, resultTensorDimension1Size, 1 do

		for b = 1, resultTensorDimension2Size, 1 do

			for c = 1, resultTensorDimension3Size, 1 do

				local subTensor = tensor[a][b]

				local originDimensionIndexArray = {(c - 1) * strideDimensionSize + 1}

				local targetDimensionIndexArray = {(c - 1) * strideDimensionSize + kernelDimensionSize}

				local extractedSubTensor = AqwamTensorLibrary:extract(subTensor, originDimensionIndexArray, targetDimensionIndexArray)

				local maximumValue = AqwamTensorLibrary:findMaximumValue(extractedSubTensor)

				resultTensor[a][b][c] = maximumValue

			end

		end

	end

	local PartialDerivativeFunction = function(derivativeTensor)

		if (not AutomaticDifferentiationTensor:checkIfIsAutomaticDifferentiationTensor(tensor)) then return end

		local chainRuleFirstDerivativeTensor = AqwamTensorLibrary:createTensor(tensorDimensionSizeArray)

		for a = 1, resultTensorDimension1Size, 1 do

			for b = 1, resultTensorDimension2Size, 1 do

				for c = 1, resultTensorDimension3Size, 1 do

					local derivativeValue = derivativeTensor[a][b][c]

					local originDimensionIndexArray = {(c - 1) * strideDimensionSize + 1}

					local targetDimensionIndexArray = {(c - 1) * strideDimensionSize + kernelDimensionSize}

					for x = originDimensionIndexArray[1], targetDimensionIndexArray[1], 1 do

						if (resultTensor[a][b][c] == tensor[a][b][x]) then chainRuleFirstDerivativeTensor[a][b][x] = chainRuleFirstDerivativeTensor[a][b][x] + derivativeValue end

					end

				end

			end

		end

		tensor:differentiate(chainRuleFirstDerivativeTensor)

	end

	return AutomaticDifferentiationTensor.new(resultTensor, PartialDerivativeFunction, {tensor})

end

function PoolingLayers.MaximumPooling2D(tensor, kernelDimensionSizeArray, strideDimensionSizeArray)

	kernelDimensionSizeArray = kernelDimensionSizeArray or default2DKernelDimensionSizeArray

	strideDimensionSizeArray = strideDimensionSizeArray or default2DStrideDimensionSizeArray

	local tensorDimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(tensor)

	local numberOfDimensions = #tensorDimensionSizeArray

	if (numberOfDimensions ~= 4) then error("Unable to pass the input tensor to the 2D spatial maximum pooling function block. The number of dimensions of the input tensor does not equal to 4. The input tensor have " .. numberOfDimensions .. " dimensions.") end

	local resultTensorDimensionSizeArray = table.clone(tensorDimensionSizeArray)

	for dimension = 1, 2, 1 do

		local inputDimensionSize = tensorDimensionSizeArray[dimension + 2]

		local outputDimensionSize = ((inputDimensionSize - kernelDimensionSizeArray[dimension]) / strideDimensionSizeArray[dimension]) + 1

		resultTensorDimensionSizeArray[dimension + 2] = math.floor(outputDimensionSize)

	end

	local resultTensorDimension1Size = resultTensorDimensionSizeArray[1]

	local resultTensorDimension2Size = resultTensorDimensionSizeArray[2]

	local resultTensorDimension3Size = resultTensorDimensionSizeArray[3]

	local resultTensorDimension4Size = resultTensorDimensionSizeArray[4]

	local kernelDimension1Size = kernelDimensionSizeArray[1]

	local kernelDimension2Size = kernelDimensionSizeArray[2]

	local strideDimension1Size = strideDimensionSizeArray[1]

	local strideDimension2Size = strideDimensionSizeArray[2]

	local resultTensor = AqwamTensorLibrary:createTensor(resultTensorDimensionSizeArray)

	for a = 1, resultTensorDimension1Size, 1 do

		for b = 1, resultTensorDimension2Size, 1 do

			for c = 1, resultTensorDimension3Size, 1 do

				for d = 1, resultTensorDimension4Size, 1 do

					local subTensor = tensor[a][b]

					local originDimensionIndexArray = {(c - 1) * strideDimension1Size + 1, (d - 1) * strideDimension2Size + 1}

					local targetDimensionIndexArray = {(c - 1) * strideDimension1Size + kernelDimension1Size, (d - 1) * strideDimension2Size + kernelDimension2Size}

					local extractedSubTensor = AqwamTensorLibrary:extract(subTensor, originDimensionIndexArray, targetDimensionIndexArray)

					local maximumValue = AqwamTensorLibrary:findMaximumValue(extractedSubTensor)

					resultTensor[a][b][c][d] = maximumValue

				end

			end

		end

	end

	local PartialDerivativeFunction = function(derivativeTensor)

		if (not AutomaticDifferentiationTensor:checkIfIsAutomaticDifferentiationTensor(tensor)) then return end 

		local derivativeTensorSizeArray = AqwamTensorLibrary:getDimensionSizeArray(derivativeTensor)

		local chainRuleFirstDerivativeTensor = AqwamTensorLibrary:createTensor(tensorDimensionSizeArray)

		for a = 1, resultTensorDimension1Size, 1 do

			for b = 1, resultTensorDimension2Size, 1 do

				for c = 1, resultTensorDimension3Size, 1 do

					for d = 1, resultTensorDimension4Size, 1 do

						local derivativeValue = derivativeTensor[a][b][c][d]

						local originDimensionIndexArray = {(c - 1) * strideDimension1Size + 1, (d - 1) * strideDimension2Size + 1}

						local targetDimensionIndexArray = {(c - 1) * strideDimension1Size + kernelDimension1Size, (d - 1) * strideDimension2Size + kernelDimension2Size}

						for x = originDimensionIndexArray[1], targetDimensionIndexArray[1], 1 do

							for y = originDimensionIndexArray[2], targetDimensionIndexArray[2], 1 do

								if (resultTensor[a][b][c][d] == tensor[a][b][x][y]) then chainRuleFirstDerivativeTensor[a][b][x][y] = chainRuleFirstDerivativeTensor[a][b][x][y] + derivativeValue end

							end

						end

					end

				end

			end

		end

		tensor:differentiate(chainRuleFirstDerivativeTensor)

	end

	return AutomaticDifferentiationTensor.new(resultTensor, PartialDerivativeFunction, {tensor})

end

function PoolingLayers.MaximumPooling3D(tensor, kernelDimensionSizeArray, strideDimensionSizeArray)

	kernelDimensionSizeArray = kernelDimensionSizeArray or default3DKernelDimensionSizeArray

	strideDimensionSizeArray = strideDimensionSizeArray or default3DStrideDimensionSizeArray

	local tensorDimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(tensor)

	local numberOfDimensions = #tensorDimensionSizeArray

	if (numberOfDimensions ~= 5) then error("Unable to pass the input tensor to the 3D spatial maximum pooling function block. The number of dimensions of the input tensor does not equal to 5. The input tensor have " .. numberOfDimensions .. " dimensions.") end

	local resultTensorDimensionSizeArray = table.clone(tensorDimensionSizeArray)

	for dimension = 1, 3, 1 do

		local inputDimensionSize = tensorDimensionSizeArray[dimension + 2]

		local outputDimensionSize = ((inputDimensionSize - kernelDimensionSizeArray[dimension]) / strideDimensionSizeArray[dimension]) + 1

		resultTensorDimensionSizeArray[dimension + 2] = math.floor(outputDimensionSize)

	end

	local resultTensorDimension1Size = resultTensorDimensionSizeArray[1]

	local resultTensorDimension2Size = resultTensorDimensionSizeArray[2]

	local resultTensorDimension3Size = resultTensorDimensionSizeArray[3]

	local resultTensorDimension4Size = resultTensorDimensionSizeArray[4]

	local resultTensorDimension5Size = resultTensorDimensionSizeArray[5]

	local kernelDimension1Size = kernelDimensionSizeArray[1]

	local kernelDimension2Size = kernelDimensionSizeArray[2]

	local kernelDimension3Size = kernelDimensionSizeArray[3]

	local strideDimension1Size = strideDimensionSizeArray[1]

	local strideDimension2Size = strideDimensionSizeArray[2]

	local strideDimension3Size = strideDimensionSizeArray[3]

	local resultTensor = AqwamTensorLibrary:createTensor(resultTensorDimensionSizeArray)

	for a = 1, resultTensorDimension1Size, 1 do

		for b = 1, resultTensorDimension2Size, 1 do

			for c = 1, resultTensorDimension3Size, 1 do

				for d = 1, resultTensorDimension4Size, 1 do

					for e = 1, resultTensorDimension5Size, 1 do

						local subTensor = tensor[a][b]

						local originDimensionIndexArray = {(c - 1) * strideDimension1Size + 1, (d - 1) * strideDimension2Size + 1, (e - 1) * strideDimension3Size + 1}

						local targetDimensionIndexArray = {(c - 1) * strideDimension1Size + kernelDimension1Size, (d - 1) * strideDimension2Size + kernelDimension2Size, (e - 1) * strideDimension3Size + kernelDimension3Size}

						local extractedSubTensor = AqwamTensorLibrary:extract(subTensor, originDimensionIndexArray, targetDimensionIndexArray)

						local maximumValue = AqwamTensorLibrary:findMaximumValue(extractedSubTensor)

						resultTensor[a][b][c][d][e] = maximumValue

					end

				end

			end

		end

	end

	local PartialDerivativeFunction = function(derivativeTensor)

		if (not AutomaticDifferentiationTensor:checkIfIsAutomaticDifferentiationTensor(tensor)) then return end 

		local chainRuleFirstDerivativeTensor = AqwamTensorLibrary:createTensor(tensorDimensionSizeArray)

		for a = 1, resultTensorDimension1Size, 1 do

			for b = 1, resultTensorDimension2Size, 1 do

				for c = 1, resultTensorDimension3Size, 1 do

					for d = 1, resultTensorDimension4Size, 1 do

						for e = 1, resultTensorDimension5Size, 1 do

							local derivativeValue = derivativeTensor[a][b][c][d][e]

							local originDimensionIndexArray = {(c - 1) * strideDimension1Size + 1, (d - 1) * strideDimension2Size + 1, (e - 1) * strideDimension3Size + 1}

							local targetDimensionIndexArray = {(c - 1) * strideDimension1Size + kernelDimension1Size, (d - 1) * strideDimension2Size + kernelDimension2Size, (e - 1) * strideDimension3Size + kernelDimension3Size}

							for x = originDimensionIndexArray[1], targetDimensionIndexArray[1], 1 do

								for y = originDimensionIndexArray[2], targetDimensionIndexArray[2], 1 do

									for z = originDimensionIndexArray[3], targetDimensionIndexArray[3], 1 do

										if (resultTensor[a][b][c][d][e] == tensor[a][b][x][y][z]) then chainRuleFirstDerivativeTensor[a][b][x][y][z] = chainRuleFirstDerivativeTensor[a][b][x][y][z] + derivativeValue end

									end

								end

							end

						end

					end

				end

			end

		end

		tensor:differentiate(chainRuleFirstDerivativeTensor)

	end

	return AutomaticDifferentiationTensor.new(resultTensor, PartialDerivativeFunction, {tensor})

end

return PoolingLayers
