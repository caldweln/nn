
local BinaryConnectLayer, Parent = torch.class('nn.BinaryConnectLayer', 'nn.Module')

function BinaryConnectLayer:__init(inputSize, outputSize, powerGlorot, opt)
   Parent.__init(self)

   -- binarized parameters for propagation
   self.binWeight = torch.Tensor(outputSize, inputSize)
   self.weightsDirty = 1 -- flag to update binWeights
   self.powerGlorot = powerGlorot or 0 -- Glorot disable => 0, adam optim => 1, sgd optim => 2
   self.binarization = opt.binarization or 'det' -- 'det' | 'stoch'

   self.weight = torch.Tensor(outputSize, inputSize)
   self.bias = torch.Tensor(outputSize)
   self.gradWeight = torch.Tensor(outputSize, inputSize)
   self.gradBias = torch.Tensor(outputSize)

   self:reset()
end

function BinaryConnectLayer:_binSigmoid(x)
  return torch.cmax(torch.cmin((x+1)/2,1),0)
end

function BinaryConnectLayer:_binarize(data, threshold) -- inclusive threshold for +1
  local result = data:clone()
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

--TODO reset both rv weights & bin weights/bias
function BinaryConnectLayer:reset(stdv)
   if stdv then
      stdv = stdv * math.sqrt(3)
   else
      stdv = 1./math.sqrt(self.weight:size(2))
   end
   if nn.oldSeed then
      for i=1,self.weight:size(1) do
         self.weight:select(1, i):apply(function()
            return torch.uniform(-stdv, stdv)
         end)
         self.bias[i] = torch.uniform(-stdv, stdv)
      end
   else
      self.weight:uniform(-stdv, stdv)
      self.bias:uniform(-stdv, stdv)
   end

   return self
end

function BinaryConnectLayer:updateOutput(input)

  if self.weightsDirty > 0 then
    self.binWeight = self:_binarize(self.weight)
    self.weightsDirty = 0
  end

  if input:dim() == 1 then
    self.output:resize(self.bias:size(1))
    self.output:copy(self.bias)
    self.output:addmv(1, self.binWeight, input)
  elseif input:dim() == 2 then
    local nframe = input:size(1)
    local nElement = self.output:nElement()
    self.output:resize(nframe, self.bias:size(1))
    if self.output:nElement() ~= nElement then
       self.output:zero()
    end

    self.addBuffer = self.addBuffer or input.new()

    if self.addBuffer:nElement() ~= nframe then
       self.addBuffer:resize(nframe):fill(1)
    end

    self.output:addmm(0, self.output, 1, input, self.binWeight:t())
    self.output:addr(1, self.addBuffer, self.bias)
  else
    error('input must be vector or matrix')
  end

  return self.output
end

function BinaryConnectLayer:updateGradInput(input, gradOutput)
   if self.gradInput then

      local nElement = self.gradInput:nElement()
      self.gradInput:resizeAs(input)
      if self.gradInput:nElement() ~= nElement then
         self.gradInput:zero()
      end
      if input:dim() == 1 then
         self.gradInput:addmv(0, 1, self.binWeight:t(), gradOutput)
      elseif input:dim() == 2 then
         self.gradInput:addmm(0, 1, gradOutput, self.binWeight)
      end

      return self.gradInput
   end
end

function BinaryConnectLayer:accGradParameters(input, gradOutput, scale)
   scale = scale or 1
   if input:dim() == 1 then
      self.gradWeight:addr(scale, gradOutput, input)
      self.gradBias:add(scale, gradOutput)
   elseif input:dim() == 2 then
      self.gradWeight:addmm(scale, gradOutput:t(), input)
      self.gradBias:addmv(scale, gradOutput:t(), self.addBuffer)
   end
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


-- we do not need to accumulate parameters when sharing
BinaryConnectLayer.sharedAccUpdateGradParameters = BinaryConnectLayer.accUpdateGradParameters

function BinaryConnectLayer:__tostring__()
  return torch.type(self) ..
      string.format('(%d -> %d)', self.weight:size(2), self.weight:size(1))
end
