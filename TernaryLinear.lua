local TernaryLinear, parent = torch.class('nn.TernaryLinear', 'nn.Module')

function TernaryLinear:__init(inputSize, outputSize, outputThresholdHighs, outputThresholdLows)
   parent.__init(self)

   self.weight = torch.Tensor(outputSize, inputSize)
   self.bias = torch.Tensor(outputSize):zero() -- never used
   self.gradWeight = torch.Tensor(outputSize, inputSize):zero() -- never used
   self.gradBias = torch.Tensor(outputSize):zero() -- never used

   -- flag per neuron to indicate output negation required before thresholding
   self.inversion = torch.Tensor(outputSize):zero()

   self.outputThresholdHighs = outputThresholdHighs
   self.outputThresholdLows = outputThresholdLows
end

function TernaryLinear:updateOutput(input)

  -- sum(input * weights)
  if input:dim() == 1 then
    self.output:resize(self.weight:size(1)):zero()
    self.output:addmv(1, self.weight, input)
  elseif input:dim() == 2 then
    local nframe = input:size(1)
    local nElement = self.output:nElement()
    self.output:resize(nframe, self.weight:size(1))
    if self.output:nElement() ~= nElement then
       self.output:zero()
    end
    self.output:addmm(0, self.output, 1, input, self.weight:t())
  else
    error('input must be vector or matrix')
  end

  -- ternarize output
  for j=1,self.output:size(2) do
    self.output[{{},j}] = (self.inversion ~= nil and self.inversion[j] > 0) and torch.mul(self.output[{{},j}],-1) or self.output[{{},j}] -- inefficient
    self.output[{{},j}]:snapd(self.outputThresholdLows[j], -1, self.outputThresholdHighs[j], 1, 0)
  end
  
  return self.output
end

function TernaryLinear:updateGradInput(input, gradOutput)
   if self.gradInput then

      local nElement = self.gradInput:nElement()
      self.gradInput:resizeAs(input)
      if self.gradInput:nElement() ~= nElement then
         self.gradInput:zero()
      end

      return self.gradInput
   end
end


function TernaryLinear:updateParameters(learningRate)
end

function TernaryLinear:__tostring__()
  return torch.type(self) ..
      string.format('(%d -> %d)', self.weight:size(2), self.weight:size(1))
end
