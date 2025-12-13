--[[

	--------------------------------------------------------------------

	Aqwam's Deep Learning Library (DataPredict Axon)

	Author: Aqwam Harish Aiman
	
	Email: aqwam.harish.aiman@gmail.com
	
	YouTube: https://www.youtube.com/channel/UCUrwoxv5dufEmbGsxyEUPZw
	
	LinkedIn: https://www.linkedin.com/in/aqwam-harish-aiman/
	
	--------------------------------------------------------------------
		
	By using this library, you agree to comply with our Terms and Conditions in the link below:
	
	https://github.com/AqwamCreates/DataPredict-Axon/blob/main/docs/TermsAndConditions.md
	
	--------------------------------------------------------------------
	
	DO NOT REMOVE THIS TEXT!
	
	--------------------------------------------------------------------

--]]

local AqwamTensorLibrary = require(script.Parent.AqwamTensorLibraryLinker.Value)

local AutomaticDifferentiationTensor = require(script.Parent.AutomaticDifferentiationTensor)

local ConvolutionLayers = {}

local defaultStrideDimensionSize = 1

local default2DStrideDimensionSizeArray = {1, 1}

local default3DStrideDimensionSizeArray = {1, 1, 1}

local function waitUntilAllCoroutinesFinished(coroutineArray)

	while true do

		local allFinished = true

		for _, coroutineInstance in ipairs(coroutineArray) do

			if coroutine.status(coroutineInstance) ~= "dead" then

				allFinished = false

				break

			end

		end

		if allFinished then break end

		task.wait()

	end

end

local function createCoroutineToArray(coroutineArray, functionToRun)

	local oneCoroutine = coroutine.create(functionToRun)

	table.insert(coroutineArray, oneCoroutine)

end

local function runCoroutinesUntilFinished(coroutineArray)

	for _, oneCoroutine in coroutineArray do coroutine.resume(oneCoroutine) end

	waitUntilAllCoroutinesFinished(coroutineArray)

end

function ConvolutionLayers.FastConvolution1D(parameterDictionary)
	
	parameterDictionary = parameterDictionary or {}
	
	local tensor = parameterDictionary.tensor or parameterDictionary[1]
	
	local weightTensor = parameterDictionary.weightTensor or parameterDictionary[2] 

	local strideDimensionSize = parameterDictionary.strideDimensionSize or parameterDictionary[3] or defaultStrideDimensionSize 
	
	local inputTensorArray = {tensor, weightTensor}
	
	local pureTensor = AutomaticDifferentiationTensor:fetchValue{tensor}
	
	local pureWeightTensor = AutomaticDifferentiationTensor:fetchValue{weightTensor}
	
	local tensorDimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(pureTensor)
	
	local weightTensorDimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(pureWeightTensor)
	
	local tensorNumberOfDimensions = #tensorDimensionSizeArray
	
	local weightNumberOfDimensions = #weightTensorDimensionSizeArray
	
	if (tensorNumberOfDimensions ~= 3) then error("Unable to pass the input tensor to the 1D spatial convolution function. The number of dimensions of the input tensor does not equal to 3. The input tensor have " .. tensorNumberOfDimensions .. " dimensions.") end
	
	if (weightNumberOfDimensions ~= 3) then error("Unable to pass the weight tensor to the 1D spatial convolution function. The number of dimensions of the input tensor does not equal to 3. The weight tensor have " .. weightNumberOfDimensions .. " dimensions.") end
	
	local numberOfKernels = weightTensorDimensionSizeArray[1]
	
	local channelSize = weightTensorDimensionSizeArray[2]
	
	if (tensorDimensionSizeArray[2] ~= channelSize) then error("The dimension size of input tensor is not equal to the dimension size of the weight tensor at dimension 2.") end
	
	local kernelDimensionSize = weightTensorDimensionSizeArray[3]
	
	local transformedTensorDimensionSizeArray = {tensorDimensionSizeArray[1], numberOfKernels}

	local inputDimensionSize = tensorDimensionSizeArray[3]

	local outputDimensionSize = ((inputDimensionSize - weightTensorDimensionSizeArray[3]) / strideDimensionSize) + 1

	transformedTensorDimensionSizeArray[3] = math.floor(outputDimensionSize)
	
	local resultTensor = {}
	
	local coroutineArray = {}
	
	for a = 1, transformedTensorDimensionSizeArray[1], 1 do

		local subTensor = pureTensor[a]

		resultTensor[a] = {}
		
		for w = 1, weightTensorDimensionSizeArray[1], 1 do
			
			local weight2DTensor = pureWeightTensor[w]
			
			resultTensor[a][w] = {}

			createCoroutineToArray(coroutineArray, function() -- Too slow. I had to use coroutines to speed it up.

				for c = 1, transformedTensorDimensionSizeArray[3], 1 do

					local originDimensionIndexArray = {1, (c - 1) * strideDimensionSize + 1}

					local targetDimensionIndexArray = {weightTensorDimensionSizeArray[2], (c - 1) * strideDimensionSize + kernelDimensionSize}

					local extractedInputTensor = AqwamTensorLibrary:extract(subTensor, originDimensionIndexArray, targetDimensionIndexArray)

					local subZTensor = AqwamTensorLibrary:multiply(extractedInputTensor, weight2DTensor)

					resultTensor[a][w][c] = AqwamTensorLibrary:sum(subZTensor)

				end

			end)
			
		end

	end
	
	local partialFirstDerivativeFunction

	local isFirstDerivativeFunctionNotCreatedForTheNextTensor = AutomaticDifferentiationTensor.isFirstDerivativeFunctionNotCreatedForTheNextTensor

	if (AutomaticDifferentiationTensor.isFirstDerivativeFunctionCreatedGlobally) and (not isFirstDerivativeFunctionNotCreatedForTheNextTensor) then
		
		partialFirstDerivativeFunction = function(firstDerivativeTensor)

			local tensor = inputTensorArray[1]

			local weightTensor = inputTensorArray[2]

			local firstDerivativeTensorDimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(firstDerivativeTensor)

			if AutomaticDifferentiationTensor:checkIfIsAutomaticDifferentiationTensor{tensor} then
				
				if (tensor:getIsFirstDerivativeTensorRequired()) then
					
					local pureWeightTensor = AutomaticDifferentiationTensor:fetchValue(weightTensor)

					local chainRuleFirstDerivativeTensor = AqwamTensorLibrary:createTensor(tensorDimensionSizeArray)

					for kernelIndex = 1, weightTensorDimensionSizeArray[1], 1 do

						createCoroutineToArray(coroutineArray, function() -- The calculation here is so slow that I am forced to use coroutines here. What the fuck.

							local weight2DTensor = pureWeightTensor[kernelIndex]

							for a = 1, firstDerivativeTensorDimensionSizeArray[1], 1 do

								for c = 1, firstDerivativeTensorDimensionSizeArray[3], 1 do

									local firstDerivativeValue = firstDerivativeTensor[a][kernelIndex][c]

									local originDimensionIndex = ((c - 1) * strideDimensionSize)

							--[[
								
								Since the target dimension index array can be determined by adding the kernel dimension size array to the origin index array, we don't need to find the target dimension index array. 
								Also we don't need to add 1 since the the kernel dimension size array "for" loop iteration will start with 1 and add to the origin dimension index array.
								
							--]]

									local subChainRuleFirstDerivativeTensor = AqwamTensorLibrary:multiply(firstDerivativeValue, weight2DTensor)

									for channelIndex = 1, channelSize, 1 do

										for i = 1, kernelDimensionSize, 1 do

											chainRuleFirstDerivativeTensor[a][channelIndex][originDimensionIndex + i] = chainRuleFirstDerivativeTensor[a][channelIndex][originDimensionIndex + i] + subChainRuleFirstDerivativeTensor[channelIndex][i]

										end

									end

								end

							end

						end)

					end

					runCoroutinesUntilFinished(coroutineArray)

					tensor:differentiate{chainRuleFirstDerivativeTensor}
					
				end

			end

			if AutomaticDifferentiationTensor:checkIfIsAutomaticDifferentiationTensor{weightTensor} then
				
				if (weightTensor:getIsFirstDerivativeTensorRequired()) then
					
					local pureTensor = AutomaticDifferentiationTensor:fetchValue(tensor)

					local chainRuleFirstDerivativeTensor = AqwamTensorLibrary:createTensor(weightTensorDimensionSizeArray)

					local coroutineArray = {}

					for kernelIndex = 1, weightTensorDimensionSizeArray[1], 1 do

						for kernelChannelIndex = 1, weightTensorDimensionSizeArray[2], 1 do

							createCoroutineToArray(coroutineArray, function()

								for a = 1, firstDerivativeTensorDimensionSizeArray[1], 1 do

									local subTensor = pureTensor[a][kernelChannelIndex]

									local subFirstDerivativeTensor = firstDerivativeTensor[a][kernelChannelIndex]

									for c = 1, firstDerivativeTensorDimensionSizeArray[3], 1 do

										local originDimensionIndexArray = {(c - 1) * strideDimensionSize + 1}

										local targetDimensionIndexArray = {(c - 1) * strideDimensionSize + kernelDimensionSize}

										local extractedSubTensor = AqwamTensorLibrary:extract(subTensor, originDimensionIndexArray, targetDimensionIndexArray)

										local subChainRuleFirstDerivativeTensor = AqwamTensorLibrary:multiply(extractedSubTensor, subFirstDerivativeTensor[c])

										chainRuleFirstDerivativeTensor[kernelIndex][kernelChannelIndex] = AqwamTensorLibrary:add(chainRuleFirstDerivativeTensor[kernelIndex][kernelChannelIndex], subChainRuleFirstDerivativeTensor)

									end

								end

							end)

						end

					end

					runCoroutinesUntilFinished(coroutineArray)

					weightTensor:differentiate{chainRuleFirstDerivativeTensor}
					
				end

			end

		end
		
	end
	
	if (isFirstDerivativeFunctionNotCreatedForTheNextTensor) then AutomaticDifferentiationTensor.isFirstDerivativeFunctionNotCreatedForTheNextTensor = false end
	
	runCoroutinesUntilFinished(coroutineArray)
	
	return AutomaticDifferentiationTensor.new{resultTensor, partialFirstDerivativeFunction, inputTensorArray}
	
end

function ConvolutionLayers.FastConvolution2D(parameterDictionary)

	parameterDictionary = parameterDictionary or {}

	local tensor = parameterDictionary.tensor or parameterDictionary[1]

	local weightTensor = parameterDictionary.weightTensor or parameterDictionary[2] 

	local strideDimensionSizeArray = parameterDictionary.strideDimensionSizeArray or parameterDictionary[3] or default2DStrideDimensionSizeArray

	local inputTensorArray = {tensor, weightTensor}
	
	local pureTensor = AutomaticDifferentiationTensor:fetchValue{tensor}

	local pureWeightTensor = AutomaticDifferentiationTensor:fetchValue{weightTensor}

	local tensorDimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(pureTensor)

	local weightTensorDimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(pureWeightTensor)
	
	local tensorNumberOfDimensions = #tensorDimensionSizeArray

	local weightNumberOfDimensions = #weightTensorDimensionSizeArray

	if (tensorNumberOfDimensions ~= 4) then error("Unable to pass the input tensor to the 2D spatial convolution function. The number of dimensions of the input tensor does not equal to 4. The input tensor have " .. tensorNumberOfDimensions .. " dimensions.") end

	if (weightNumberOfDimensions ~= 4) then error("Unable to pass the weight tensor to the 2D spatial convolution function. The number of dimensions of the input tensor does not equal to 4. The weight tensor have " .. weightNumberOfDimensions .. " dimensions.") end

	local numberOfKernels = weightTensorDimensionSizeArray[1]

	local channelSize = weightTensorDimensionSizeArray[2]
	
	if (tensorDimensionSizeArray[2] ~= channelSize) then error("The dimension size of input tensor is not equal to the dimension size of the weight tensor at dimension 2.") end

	local kernelDimensionSize = weightTensorDimensionSizeArray[3]

	local resultTensorDimensionSizeArray = {tensorDimensionSizeArray[1], numberOfKernels}

	for dimension = 1, 2, 1 do

		local inputDimensionSize = tensorDimensionSizeArray[dimension + 2]

		local outputDimensionSize = ((inputDimensionSize - weightTensorDimensionSizeArray[dimension + 2]) / strideDimensionSizeArray[dimension]) + 1

		resultTensorDimensionSizeArray[dimension + 2] = math.floor(outputDimensionSize)

	end

	local resultTensor = {}

	local coroutineArray = {}

	for a = 1, resultTensorDimensionSizeArray[1], 1 do

		local subTensor = pureTensor[a]

		resultTensor[a] = {}

		for w = 1, weightTensorDimensionSizeArray[1], 1 do

			local weight3DTensor = pureWeightTensor[w]

			resultTensor[a][w] = {}

			createCoroutineToArray(coroutineArray, function() -- Too slow. I had to use coroutines to speed it up.

				for c = 1, resultTensorDimensionSizeArray[3], 1 do

					resultTensor[a][w][c] = {}

					for d = 1, resultTensorDimensionSizeArray[4], 1 do

						local originDimensionIndexArray = {1, (c - 1) * strideDimensionSizeArray[1] + 1, (d - 1) * strideDimensionSizeArray[2] + 1}

						local targetDimensionIndexArray = {weightTensorDimensionSizeArray[2], (c - 1) * strideDimensionSizeArray[1] + weightTensorDimensionSizeArray[3], (d - 1) * strideDimensionSizeArray[2] + weightTensorDimensionSizeArray[4]}

						local extractedInputTensor = AqwamTensorLibrary:extract(subTensor, originDimensionIndexArray, targetDimensionIndexArray)

						local subZTensor = AqwamTensorLibrary:multiply(extractedInputTensor, weight3DTensor)

						resultTensor[a][w][c][d] = AqwamTensorLibrary:sum(subZTensor)

					end

				end

			end)

		end

	end
	
	local partialFirstDerivativeFunction

	local isFirstDerivativeFunctionNotCreatedForTheNextTensor = AutomaticDifferentiationTensor.isFirstDerivativeFunctionNotCreatedForTheNextTensor

	if (AutomaticDifferentiationTensor.isFirstDerivativeFunctionCreatedGlobally) and (not isFirstDerivativeFunctionNotCreatedForTheNextTensor) then
		
		partialFirstDerivativeFunction = function(firstDerivativeTensor)

			local tensor = inputTensorArray[1]

			local weightTensor = inputTensorArray[2]

			local firstDerivativeTensorDimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(firstDerivativeTensor)

			if AutomaticDifferentiationTensor:checkIfIsAutomaticDifferentiationTensor{tensor} then
				
				if (tensor:getIsFirstDerivativeTensorRequired()) then
					
					local pureWeightTensor = AutomaticDifferentiationTensor:fetchValue(weightTensor)

					local chainRuleFirstDerivativeTensor = AqwamTensorLibrary:createTensor(tensorDimensionSizeArray)

					for kernelIndex = 1, weightTensorDimensionSizeArray[1], 1 do

						createCoroutineToArray(coroutineArray, function() -- The calculation here is so slow that I am forced to use coroutines here. What the fuck.

							local weight3DTensor = pureWeightTensor[kernelIndex]

							for a = 1, firstDerivativeTensorDimensionSizeArray[1], 1 do

								for c = 1, firstDerivativeTensorDimensionSizeArray[3], 1 do

									for d = 1, firstDerivativeTensorDimensionSizeArray[4], 1 do

										local firstDerivativeValue = firstDerivativeTensor[a][kernelIndex][c][d]

										local originDimensionIndexArray = {(c - 1) * strideDimensionSizeArray[1], (d - 1) * strideDimensionSizeArray[2]} 

								--[[
								
									Since the target dimension index array can be determined by adding the kernel dimension size array to the origin index array, we don't need to find the target dimension index array. 
									Also we don't need to add 1 since the the kernel dimension size array "for" loop iteration will start with 1 and add to the origin dimension index array.
								
								--]]

										local subChainRuleFirstDerivativeTensor = AqwamTensorLibrary:multiply(firstDerivativeValue, weight3DTensor)

										for channelIndex = 1, channelSize, 1 do

											for i = 1, weightTensorDimensionSizeArray[3], 1 do

												for j = 1, weightTensorDimensionSizeArray[4], 1 do

													chainRuleFirstDerivativeTensor[a][channelIndex][originDimensionIndexArray[1] + i][originDimensionIndexArray[2] + j] = chainRuleFirstDerivativeTensor[a][channelIndex][originDimensionIndexArray[1] + i][originDimensionIndexArray[2] + j] + subChainRuleFirstDerivativeTensor[channelIndex][i][j]

												end

											end

										end

									end

								end

							end

						end)

					end

					runCoroutinesUntilFinished(coroutineArray)

					tensor:differentiate{chainRuleFirstDerivativeTensor}
					
				end

			end

			if AutomaticDifferentiationTensor:checkIfIsAutomaticDifferentiationTensor{weightTensor} then
				
				if (weightTensor:getIsFirstDerivativeTensorRequired()) then
					
					local pureTensor = AutomaticDifferentiationTensor:fetchValue(tensor)

					local chainRuleFirstDerivativeTensor = AqwamTensorLibrary:createTensor(weightTensorDimensionSizeArray)

					local coroutineArray = {}

					for kernelIndex = 1, weightTensorDimensionSizeArray[1], 1 do

						for kernelChannelIndex = 1, weightTensorDimensionSizeArray[2], 1 do

							createCoroutineToArray(coroutineArray, function()

								for a = 1, firstDerivativeTensorDimensionSizeArray[1], 1 do

									local subTensor = pureTensor[a][kernelChannelIndex]

									local subFirstDerivativeTensor = firstDerivativeTensor[a][kernelChannelIndex]

									for c = 1, firstDerivativeTensorDimensionSizeArray[3], 1 do

										for d = 1, firstDerivativeTensorDimensionSizeArray[4], 1 do

											local originDimensionIndexArray = {(c - 1) * strideDimensionSizeArray[1] + 1, (d - 1) * strideDimensionSizeArray[2] + 1}

											local targetDimensionIndexArray = {(c - 1) * strideDimensionSizeArray[1] + weightTensorDimensionSizeArray[3], (d - 1) * strideDimensionSizeArray[2] + weightTensorDimensionSizeArray[4]}

											local extractedSubTensor = AqwamTensorLibrary:extract(subTensor, originDimensionIndexArray, targetDimensionIndexArray)

											local subChainRuleFirstDerivativeTensor = AqwamTensorLibrary:multiply(extractedSubTensor, subFirstDerivativeTensor[c][d])

											chainRuleFirstDerivativeTensor[kernelIndex][kernelChannelIndex] = AqwamTensorLibrary:add(chainRuleFirstDerivativeTensor[kernelIndex][kernelChannelIndex], subChainRuleFirstDerivativeTensor)

										end

									end

								end

							end)

						end

					end

					runCoroutinesUntilFinished(coroutineArray)

					weightTensor:differentiate{chainRuleFirstDerivativeTensor}
					
				end

			end

		end
		
	end
	
	if (isFirstDerivativeFunctionNotCreatedForTheNextTensor) then AutomaticDifferentiationTensor.isFirstDerivativeFunctionNotCreatedForTheNextTensor = false end

	runCoroutinesUntilFinished(coroutineArray)

	return AutomaticDifferentiationTensor.new{resultTensor, partialFirstDerivativeFunction, inputTensorArray}

end

function ConvolutionLayers.FastConvolution3D(parameterDictionary)

	parameterDictionary = parameterDictionary or {}

	local tensor = parameterDictionary.tensor or parameterDictionary[1]

	local weightTensor = parameterDictionary.weightTensor or parameterDictionary[2] 

	local strideDimensionSizeArray = parameterDictionary.strideDimensionSizeArray or parameterDictionary[3] or default3DStrideDimensionSizeArray 

	local inputTensorArray = {tensor, weightTensor}
	
	local pureTensor = AutomaticDifferentiationTensor:fetchValue{tensor}

	local pureWeightTensor = AutomaticDifferentiationTensor:fetchValue{weightTensor}

	local tensorDimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(pureTensor)

	local weightTensorDimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(pureWeightTensor)
	
	local tensorNumberOfDimensions = #tensorDimensionSizeArray

	local weightNumberOfDimensions = #weightTensorDimensionSizeArray

	if (tensorNumberOfDimensions ~= 5) then error("Unable to pass the input tensor to the 3D spatial convolution function. The number of dimensions of the input tensor does not equal to 5. The input tensor have " .. tensorNumberOfDimensions .. " dimensions.") end

	if (weightNumberOfDimensions ~= 5) then error("Unable to pass the weight tensor to the 3D spatial convolution function. The number of dimensions of the input tensor does not equal to 5. The weight tensor have " .. weightNumberOfDimensions .. " dimensions.") end

	local numberOfKernels = weightTensorDimensionSizeArray[1]

	local channelSize = weightTensorDimensionSizeArray[2]
	
	if (tensorDimensionSizeArray[2] ~= channelSize) then error("The dimension size of input tensor is not equal to the dimension size of the weight tensor at dimension 2.") end

	local kernelDimensionSize = weightTensorDimensionSizeArray[3]

	local resultTensorDimensionSizeArray = {tensorDimensionSizeArray[1], numberOfKernels}

	for dimension = 1, 3, 1 do

		local inputDimensionSize = tensorDimensionSizeArray[dimension + 2]

		local outputDimensionSize = ((inputDimensionSize - weightTensorDimensionSizeArray[dimension + 2]) / strideDimensionSizeArray[dimension]) + 1

		resultTensorDimensionSizeArray[dimension + 2] = math.floor(outputDimensionSize)

	end

	local resultTensor = {}

	local coroutineArray = {}

	for a = 1, resultTensorDimensionSizeArray[1], 1 do

		local subTensor = pureTensor[a]

		resultTensor[a] = {}

		for w = 1, weightTensorDimensionSizeArray[1], 1 do

			local weight3DTensor = pureWeightTensor[w]

			resultTensor[a][w] = {}

			createCoroutineToArray(coroutineArray, function() -- Too slow. I had to use coroutines to speed it up.

				for c = 1, resultTensorDimensionSizeArray[3], 1 do

					resultTensor[a][w][c] = {}

					for d = 1, resultTensorDimensionSizeArray[4], 1 do
						
						resultTensor[a][w][c][d] = {}
						
						for e = 1, resultTensorDimensionSizeArray[5], 1 do
							
							local originDimensionIndexArray = {1, (c - 1) * strideDimensionSizeArray[1] + 1, (d - 1) * strideDimensionSizeArray[2] + 1, (e - 1) * strideDimensionSizeArray[3] + 1}

							local targetDimensionIndexArray = {weightTensorDimensionSizeArray[2], (c - 1) * strideDimensionSizeArray[1] + weightTensorDimensionSizeArray[3], (d - 1) * strideDimensionSizeArray[2] + weightTensorDimensionSizeArray[4], (e - 1) * strideDimensionSizeArray[3] + weightTensorDimensionSizeArray[5]}

							local extractedInputTensor = AqwamTensorLibrary:extract(subTensor, originDimensionIndexArray, targetDimensionIndexArray)

							local subZTensor = AqwamTensorLibrary:multiply(extractedInputTensor, weight3DTensor)

							resultTensor[a][w][c][d][e] = AqwamTensorLibrary:sum(subZTensor)
							
						end

					end

				end

			end)

		end

	end
	
	local partialFirstDerivativeFunction

	local isFirstDerivativeFunctionNotCreatedForTheNextTensor = AutomaticDifferentiationTensor.isFirstDerivativeFunctionNotCreatedForTheNextTensor

	if (AutomaticDifferentiationTensor.isFirstDerivativeFunctionCreatedGlobally) and (not isFirstDerivativeFunctionNotCreatedForTheNextTensor) then
		
		partialFirstDerivativeFunction = function(firstDerivativeTensor)

			local tensor = inputTensorArray[1]

			local weightTensor = inputTensorArray[2]

			local firstDerivativeTensorDimensionSizeArray = AqwamTensorLibrary:getDimensionSizeArray(firstDerivativeTensor)

			if AutomaticDifferentiationTensor:checkIfIsAutomaticDifferentiationTensor{tensor} then
				
				if (tensor:getIsFirstDerivativeTensorRequired()) then
					
					local pureWeightTensor = AutomaticDifferentiationTensor:fetchValue(weightTensor)

					local chainRuleFirstDerivativeTensor = AqwamTensorLibrary:createTensor(tensorDimensionSizeArray)

					for kernelIndex = 1, weightTensorDimensionSizeArray[1], 1 do

						createCoroutineToArray(coroutineArray, function() -- The calculation here is so slow that I am forced to use coroutines here. What the fuck.

							local weight3DTensor = pureWeightTensor[kernelIndex]

							for a = 1, firstDerivativeTensorDimensionSizeArray[1], 1 do

								for c = 1, firstDerivativeTensorDimensionSizeArray[3], 1 do

									for d = 1, firstDerivativeTensorDimensionSizeArray[4], 1 do

										for e = 1, firstDerivativeTensorDimensionSizeArray[5] do

											local firstDerivativeValue = firstDerivativeTensor[a][kernelIndex][c][d][e]

											local originDimensionIndexArray = {(c - 1) * strideDimensionSizeArray[1], (d - 1) * strideDimensionSizeArray[2], (e - 1) * strideDimensionSizeArray[3]} 

									--[[
									
										Since the target dimension index array can be determined by adding the kernel dimension size array to the origin index array, we don't need to find the target dimension index array. 
										Also we don't need to add 1 since the the kernel dimension size array "for" loop iteration will start with 1 and add to the origin dimension index array.
									
									--]]

											local subChainRuleFirstDerivativeTensor = AqwamTensorLibrary:multiply(firstDerivativeValue, weight3DTensor)

											for channelIndex = 1, channelSize, 1 do

												for i = 1, weightTensorDimensionSizeArray[3], 1 do

													for j = 1, weightTensorDimensionSizeArray[4], 1 do

														for k = 1, weightTensorDimensionSizeArray[4], 1 do

															chainRuleFirstDerivativeTensor[a][channelIndex][originDimensionIndexArray[1] + i][originDimensionIndexArray[2] + j][originDimensionIndexArray[3] + k] = chainRuleFirstDerivativeTensor[a][channelIndex][originDimensionIndexArray[1] + i][originDimensionIndexArray[2] + j][originDimensionIndexArray[3] + k] + subChainRuleFirstDerivativeTensor[channelIndex][i][j][k]

														end

													end

												end

											end

										end

									end

								end

							end

						end)

					end

					runCoroutinesUntilFinished(coroutineArray)

					tensor:differentiate{chainRuleFirstDerivativeTensor}
					
				end

			end

			if AutomaticDifferentiationTensor:checkIfIsAutomaticDifferentiationTensor{weightTensor} then
				
				if (weightTensor:getIsFirstDerivativeTensorRequired()) then
					
					local pureTensor = AutomaticDifferentiationTensor:fetchValue(tensor)

					local chainRuleFirstDerivativeTensor = AqwamTensorLibrary:createTensor(weightTensorDimensionSizeArray)

					local coroutineArray = {}

					for kernelIndex = 1, weightTensorDimensionSizeArray[1], 1 do

						for kernelChannelIndex = 1, weightTensorDimensionSizeArray[2], 1 do

							createCoroutineToArray(coroutineArray, function()

								for a = 1, firstDerivativeTensorDimensionSizeArray[1], 1 do

									local subTensor = pureTensor[a][kernelChannelIndex]

									local subFirstDerivativeTensor = firstDerivativeTensor[a][kernelChannelIndex]

									for c = 1, firstDerivativeTensorDimensionSizeArray[3], 1 do

										for d = 1, firstDerivativeTensorDimensionSizeArray[4], 1 do

											for e = 1, firstDerivativeTensorDimensionSizeArray[5], 1 do

												local originDimensionIndexArray = {(c - 1) * strideDimensionSizeArray[1] + 1, (d - 1) * strideDimensionSizeArray[2] + 1, (e - 1) * strideDimensionSizeArray[3] + 1}

												local targetDimensionIndexArray = {(c - 1) * strideDimensionSizeArray[1] + weightTensorDimensionSizeArray[3], (d - 1) * strideDimensionSizeArray[2] + weightTensorDimensionSizeArray[4], (e - 1) * strideDimensionSizeArray[3] + weightTensorDimensionSizeArray[5]}

												local extractedSubTensor = AqwamTensorLibrary:extract(subTensor, originDimensionIndexArray, targetDimensionIndexArray)

												local subChainRuleFirstDerivativeTensor = AqwamTensorLibrary:multiply(extractedSubTensor, subFirstDerivativeTensor[c][d][e])

												chainRuleFirstDerivativeTensor[kernelIndex][kernelChannelIndex] = AqwamTensorLibrary:add(chainRuleFirstDerivativeTensor[kernelIndex][kernelChannelIndex], subChainRuleFirstDerivativeTensor)

											end

										end

									end

								end

							end)

						end

					end

					runCoroutinesUntilFinished(coroutineArray)

					weightTensor:differentiate{chainRuleFirstDerivativeTensor}
					
				end

			end

		end
		
	end
	
	if (isFirstDerivativeFunctionNotCreatedForTheNextTensor) then AutomaticDifferentiationTensor.isFirstDerivativeFunctionNotCreatedForTheNextTensor = false end

	runCoroutinesUntilFinished(coroutineArray)

	return AutomaticDifferentiationTensor.new{resultTensor, partialFirstDerivativeFunction, inputTensorArray}

end

function ConvolutionLayers.Convolution1D(parameterDictionary)

	parameterDictionary = parameterDictionary or {}

	local tensor = parameterDictionary.tensor or parameterDictionary[1]

	local weightTensor = parameterDictionary.weightTensor or parameterDictionary[2] 

	local strideDimensionSize = parameterDictionary.strideDimensionSize or parameterDictionary[3] or defaultStrideDimensionSize 

	local inputTensorArray = {tensor, weightTensor}

	tensor = AutomaticDifferentiationTensor.coerce{tensor}

	weightTensor = AutomaticDifferentiationTensor.coerce{weightTensor}

	local tensorDimensionSizeArray = tensor:getDimensionSizeArray()

	local weightTensorDimensionSizeArray = weightTensor:getDimensionSizeArray()

	local tensorNumberOfDimensions = #tensorDimensionSizeArray

	local weightNumberOfDimensions = #weightTensorDimensionSizeArray

	if (tensorNumberOfDimensions ~= 3) then error("Unable to pass the input tensor to the 1D spatial convolution function. The number of dimensions of the input tensor does not equal to 3. The input tensor have " .. tensorNumberOfDimensions .. " dimensions.") end

	if (weightNumberOfDimensions ~= 3) then error("Unable to pass the weight tensor to the 1D spatial convolution function. The number of dimensions of the input tensor does not equal to 3. The weight tensor have " .. weightNumberOfDimensions .. " dimensions.") end

	local numberOfKernels = weightTensorDimensionSizeArray[1]

	local channelSize = weightTensorDimensionSizeArray[2]

	if (tensorDimensionSizeArray[2] ~= channelSize) then error("The dimension size of input tensor is not equal to the dimension size of the weight tensor at dimension 2.") end

	local kernelDimensionSize = weightTensorDimensionSizeArray[3]

	local transformedTensorDimensionSizeArray = {tensorDimensionSizeArray[1], numberOfKernels}

	local inputDimensionSize = tensorDimensionSizeArray[3]

	local outputDimensionSize = ((inputDimensionSize - weightTensorDimensionSizeArray[3]) / strideDimensionSize) + 1

	transformedTensorDimensionSizeArray[3] = math.floor(outputDimensionSize)

	local aSubTensorArray = {}

	for a = 1, transformedTensorDimensionSizeArray[1], 1 do

		local subTensor = tensor[a]
		
		local wSubTensorArray = {}

		for w = 1, weightTensorDimensionSizeArray[1], 1 do

			local weight2DTensor = weightTensor[w]
			
			local cSubTensorArray = {}

			for c = 1, transformedTensorDimensionSizeArray[3], 1 do

				local originDimensionIndexArray = {1, (c - 1) * strideDimensionSize + 1}

				local targetDimensionIndexArray = {weightTensorDimensionSizeArray[2], (c - 1) * strideDimensionSize + kernelDimensionSize}

				local extractedInputTensor = subTensor:extract{originDimensionIndexArray, targetDimensionIndexArray}

				local subZTensor = extractedInputTensor * weight2DTensor
				
				local resultValue = subZTensor:sum()
				
				cSubTensorArray[c] = resultValue

			end
			
			wSubTensorArray[w] = AutomaticDifferentiationTensor.stack(cSubTensorArray)

		end
		
		aSubTensorArray[a] = AutomaticDifferentiationTensor.stack(wSubTensorArray)

	end
	
	local resultTensor = AutomaticDifferentiationTensor.stack(aSubTensorArray)

	return resultTensor

end

function ConvolutionLayers.Convolution2D(parameterDictionary)

	parameterDictionary = parameterDictionary or {}

	local tensor = parameterDictionary.tensor or parameterDictionary[1]

	local weightTensor = parameterDictionary.weightTensor or parameterDictionary[2] 

	local strideDimensionSizeArray = parameterDictionary.strideDimensionSizeArray or parameterDictionary[3] or default2DStrideDimensionSizeArray

	local inputTensorArray = {tensor, weightTensor}

	tensor = AutomaticDifferentiationTensor.coerce{tensor}

	weightTensor = AutomaticDifferentiationTensor.coerce{weightTensor}

	local tensorDimensionSizeArray = tensor:getDimensionSizeArray()

	local weightTensorDimensionSizeArray = weightTensor:getDimensionSizeArray()

	local tensorNumberOfDimensions = #tensorDimensionSizeArray

	local weightNumberOfDimensions = #weightTensorDimensionSizeArray

	if (tensorNumberOfDimensions ~= 4) then error("Unable to pass the input tensor to the 2D spatial convolution function. The number of dimensions of the input tensor does not equal to 4. The input tensor have " .. tensorNumberOfDimensions .. " dimensions.") end

	if (weightNumberOfDimensions ~= 4) then error("Unable to pass the weight tensor to the 2D spatial convolution function. The number of dimensions of the input tensor does not equal to 4. The weight tensor have " .. weightNumberOfDimensions .. " dimensions.") end

	local numberOfKernels = weightTensorDimensionSizeArray[1]

	local channelSize = weightTensorDimensionSizeArray[2]

	if (tensorDimensionSizeArray[2] ~= channelSize) then error("The dimension size of input tensor is not equal to the dimension size of the weight tensor at dimension 2.") end

	local kernelDimensionSize = weightTensorDimensionSizeArray[3]

	local resultTensorDimensionSizeArray = {tensorDimensionSizeArray[1], numberOfKernels}

	for dimension = 1, 2, 1 do

		local inputDimensionSize = tensorDimensionSizeArray[dimension + 2]

		local outputDimensionSize = ((inputDimensionSize - weightTensorDimensionSizeArray[dimension + 2]) / strideDimensionSizeArray[dimension]) + 1

		resultTensorDimensionSizeArray[dimension + 2] = math.floor(outputDimensionSize)

	end

	local aSubTensorArray = {}

	for a = 1, resultTensorDimensionSizeArray[1], 1 do

		local subTensor = tensor[a]

		local wSubTensorArray = {}

		for w = 1, weightTensorDimensionSizeArray[1], 1 do

			local weight3DTensor = weightTensor[w]

			local cSubTensorArray = {}

			for c = 1, resultTensorDimensionSizeArray[3], 1 do

				local dSubTensorArray = {}

				for d = 1, resultTensorDimensionSizeArray[4], 1 do

					local originDimensionIndexArray = {1, (c - 1) * strideDimensionSizeArray[1] + 1, (d - 1) * strideDimensionSizeArray[2] + 1}

					local targetDimensionIndexArray = {weightTensorDimensionSizeArray[2], (c - 1) * strideDimensionSizeArray[1] + weightTensorDimensionSizeArray[3], (d - 1) * strideDimensionSizeArray[2] + weightTensorDimensionSizeArray[4]}

					local extractedInputTensor = subTensor:extract{originDimensionIndexArray, targetDimensionIndexArray}

					local subZTensor = extractedInputTensor * weight3DTensor

					local resultValue = subZTensor:sum()
					
					dSubTensorArray[d] = resultValue

				end
				
				cSubTensorArray[c] = AutomaticDifferentiationTensor.stack(dSubTensorArray)

			end
			
			wSubTensorArray[w] = AutomaticDifferentiationTensor.stack(cSubTensorArray)

		end
		
		aSubTensorArray[a] = AutomaticDifferentiationTensor.stack(wSubTensorArray)

	end
	
	local resultTensor = AutomaticDifferentiationTensor.stack(aSubTensorArray)

	return resultTensor

end

function ConvolutionLayers.Convolution3D(parameterDictionary)

	parameterDictionary = parameterDictionary or {}

	local tensor = parameterDictionary.tensor or parameterDictionary[1]

	local weightTensor = parameterDictionary.weightTensor or parameterDictionary[2] 

	local strideDimensionSizeArray = parameterDictionary.strideDimensionSizeArray or parameterDictionary[3] or default3DStrideDimensionSizeArray 

	local inputTensorArray = {tensor, weightTensor}

	tensor = AutomaticDifferentiationTensor.coerce{tensor}

	weightTensor = AutomaticDifferentiationTensor.coerce{weightTensor}

	local tensorDimensionSizeArray = tensor:getDimensionSizeArray()

	local weightTensorDimensionSizeArray = weightTensor:getDimensionSizeArray()

	local tensorNumberOfDimensions = #tensorDimensionSizeArray

	local weightNumberOfDimensions = #weightTensorDimensionSizeArray

	if (tensorNumberOfDimensions ~= 5) then error("Unable to pass the input tensor to the 3D spatial convolution function. The number of dimensions of the input tensor does not equal to 5. The input tensor have " .. tensorNumberOfDimensions .. " dimensions.") end

	if (weightNumberOfDimensions ~= 5) then error("Unable to pass the weight tensor to the 3D spatial convolution function. The number of dimensions of the input tensor does not equal to 5. The weight tensor have " .. weightNumberOfDimensions .. " dimensions.") end

	local numberOfKernels = weightTensorDimensionSizeArray[1]

	local channelSize = weightTensorDimensionSizeArray[2]

	if (tensorDimensionSizeArray[2] ~= channelSize) then error("The dimension size of input tensor is not equal to the dimension size of the weight tensor at dimension 2.") end

	local kernelDimensionSize = weightTensorDimensionSizeArray[3]

	local resultTensorDimensionSizeArray = {tensorDimensionSizeArray[1], numberOfKernels}

	for dimension = 1, 3, 1 do

		local inputDimensionSize = tensorDimensionSizeArray[dimension + 2]

		local outputDimensionSize = ((inputDimensionSize - weightTensorDimensionSizeArray[dimension + 2]) / strideDimensionSizeArray[dimension]) + 1

		resultTensorDimensionSizeArray[dimension + 2] = math.floor(outputDimensionSize)

	end

	local aSubTensorArray = {}

	for a = 1, resultTensorDimensionSizeArray[1], 1 do

		local subTensor = tensor[a]

		local wSubTensorArray = {}

		for w = 1, weightTensorDimensionSizeArray[1], 1 do

			local weight3DTensor = weightTensor[w]

			local cSubTensorArray = {}

			for c = 1, resultTensorDimensionSizeArray[3], 1 do

				local dSubTensorArray = {}

				for d = 1, resultTensorDimensionSizeArray[4], 1 do

					local eSubTensorArray = {}

					for e = 1, resultTensorDimensionSizeArray[5], 1 do

						local originDimensionIndexArray = {1, (c - 1) * strideDimensionSizeArray[1] + 1, (d - 1) * strideDimensionSizeArray[2] + 1, (e - 1) * strideDimensionSizeArray[3] + 1}

						local targetDimensionIndexArray = {weightTensorDimensionSizeArray[2], (c - 1) * strideDimensionSizeArray[1] + weightTensorDimensionSizeArray[3], (d - 1) * strideDimensionSizeArray[2] + weightTensorDimensionSizeArray[4], (e - 1) * strideDimensionSizeArray[3] + weightTensorDimensionSizeArray[5]}

						local extractedInputTensor = subTensor:extract{originDimensionIndexArray, targetDimensionIndexArray}

						local subZTensor = extractedInputTensor * weight3DTensor

						local resultValue = subZTensor:sum()
						
						eSubTensorArray[e] = resultValue

					end
					
					dSubTensorArray[d] = AutomaticDifferentiationTensor.stack(eSubTensorArray)

				end
				
				cSubTensorArray[c] = AutomaticDifferentiationTensor.stack(dSubTensorArray)

			end
			
			wSubTensorArray[w] = AutomaticDifferentiationTensor.stack(cSubTensorArray)

		end
		
		aSubTensorArray[a] = AutomaticDifferentiationTensor.stack(wSubTensorArray)

	end
	
	local resultTensor = AutomaticDifferentiationTensor.stack(aSubTensorArray)

	return resultTensor

end

return ConvolutionLayers
