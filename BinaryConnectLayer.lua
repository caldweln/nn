
local BinaryConnectLayer, Parent = torch.class('nn.BinaryConnectLayer', 'nn.Linear')

function BinaryConnectLayer:__init(inputSize, outputSize, powerGlorot, opt)
   Parent.__init(self, inputSize, outputSize)
   -- binarized parameters for propagation
   self.rvWeight = torch.Tensor(outputSize, inputSize)
   self.binWeight = torch.Tensor(outputSize, inputSize)
   self.weightsDirty = 1 -- flag to update binWeights
   self.powerGlorot = powerGlorot or 0 -- Glorot disable => 0, adam optim => 1, sgd optim => 2
   self.binarization = opt.binarization or 'det' -- 'det' | 'stoch'
   -- pointer to real-valued weights by default
   self.weight = self.rvWeight
  -- initialize
   self:reset()
end

function BinaryConnectLayer:_binSigmoid(x)
  return torch.cmax(torch.cmin((x+1)/2,1),0)
end

function BinaryConnectLayer:_binarize(data, threshold) -- inclusive threshold for +1
  local result = data:clone() -- non-destructive
  if self.binarization == 'stoch' then
    threshold = threshold or 0.5
    local p = self:_binSigmoid(result)
    result[ p:ge(threshold) ] = 1
    result[ p:lt(threshold) ] = -1
  elseif self.binarization == 'det' then
    threshold = threshold or 0
    result[ result:ge(threshold) ] = 1
    result[ result:lt(threshold) ] = -1
  end
  return result
end

function BinaryConnectLayer:_clip(data, upper, lower)
  local upper = upper or 1
  local lower = lower or -1
  data[ data:gt(upper) ] = upper
  data[ data:lt(lower) ] = lower
  return data
end

function BinaryConnectLayer:updateOutput(input)

  if self.weightsDirty > 0 then
    self.binWeight = self:_binarize(self.rvWeight)
    self.weightsDirty = 0
  end

  -- switch to binary weights for duration of upgradeGradInput
  self.weight = self.binWeight
  self.output = Parent.updateOutput(self, input)
  self.weight = self.rvWeight

  return self.output
end

function BinaryConnectLayer:updateGradInput(input, gradOutput)
  -- switch to binary weights for duration of upgradeGradInput
  self.weight = self.binWeight
  self.gradInput = Parent.updateGradInput(self, input, gradOutput)
  self.weight = self.rvWeight

  return self.gradInput
end

function BinaryConnectLayer:updateParameters(learningRate)
  -- calculation of Glorot coefficients for use in scaling learningRate
  local coeffGlorot = 1 / math.sqrt( 1.5 / ( self.weight:size(2) + self.weight:size(1) ) )
  -- update real-valued parameters
  Parent.updateParameters(self, learningRate * math.pow(coeffGlorot,self.powerGlorot))
  -- clip weights
  self.weight = self:_clip(self.weight)
  -- flag binWeights for update
  self.weightsDirty = 1

end
