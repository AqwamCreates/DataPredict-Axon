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

local AqwamDeepLearningLibrary = {}

AqwamDeepLearningLibrary.AutomaticDifferentiationTensor = require(script.AutomaticDifferentiationTensor)

AqwamDeepLearningLibrary.ActivationLayers = require(script.ActivationLayers)

AqwamDeepLearningLibrary.LinkLayers = require(script.LinkLayers)

AqwamDeepLearningLibrary.CostFunctions = require(script.CostFunctions)

AqwamDeepLearningLibrary.FusedCostFunctions = require(script.FusedCostFunctions)

AqwamDeepLearningLibrary.RegularizationFunctions = require(script.RegularizationFunctions)

AqwamDeepLearningLibrary.ConvolutionLayers = require(script.ConvolutionLayers)

AqwamDeepLearningLibrary.PoolingLayers = require(script.PoolingLayers)

AqwamDeepLearningLibrary.PaddingLayers = require(script.PaddingLayers)

AqwamDeepLearningLibrary.DropoutLayers = require(script.DropoutLayers)

AqwamDeepLearningLibrary.EncodingLayers = require(script.EncodingLayers)

AqwamDeepLearningLibrary.WeightContainer = require(script.WeightContainer)

AqwamDeepLearningLibrary.Optimizers = require(script.Optimizers)

AqwamDeepLearningLibrary.ValueSchedulers = require(script.ValueSchedulers)

AqwamDeepLearningLibrary.GradientClippers = require(script.GradientClippers)

AqwamDeepLearningLibrary.Regularizers = require(script.Regularizers)

AqwamDeepLearningLibrary.ReinforcementLearningModels = require(script.ReinforcementLearningModels)

AqwamDeepLearningLibrary.EligibilityTraces = require(script.EligibilityTraces)

AqwamDeepLearningLibrary.ExperienceReplays = require(script.ExperienceReplays)

AqwamDeepLearningLibrary.RecurrentModels = require(script.RecurrentModels)

AqwamDeepLearningLibrary.QuickSetups = require(script.QuickSetups)

return AqwamDeepLearningLibrary
