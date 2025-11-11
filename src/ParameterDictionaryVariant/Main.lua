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

local DataPredictAxon = {}

DataPredictAxon.AutomaticDifferentiationTensor = require(script.AutomaticDifferentiationTensor)

DataPredictAxon.ActivationLayers = require(script.ActivationLayers)

DataPredictAxon.CostFunctions = require(script.CostFunctions)

DataPredictAxon.ConvolutionLayers = require(script.ConvolutionLayers)

DataPredictAxon.PoolingLayers = require(script.PoolingLayers)

DataPredictAxon.PaddingLayers = require(script.PaddingLayers)

DataPredictAxon.DropoutLayers = require(script.DropoutLayers)

DataPredictAxon.EncodingLayers = require(script.EncodingLayers)

DataPredictAxon.WeightContainer = require(script.WeightContainer)

DataPredictAxon.Optimizers = require(script.Optimizers)

DataPredictAxon.ValueSchedulers = require(script.ValueSchedulers)

DataPredictAxon.GradientClippers = require(script.GradientClippers)

DataPredictAxon.Regularizers = require(script.Regularizers)

DataPredictAxon.ReinforcementLearningModels = require(script.ReinforcementLearningModels)

DataPredictAxon.EligibilityTraces = require(script.EligibilityTraces)

DataPredictAxon.ExperienceReplays = require(script.ExperienceReplays)

DataPredictAxon.RecurrentModels = require(script.RecurrentModels)

return DataPredictAxon
