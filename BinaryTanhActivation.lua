local BinaryTanhActivation, Parent = torch.class('nn.BinaryTanhActivation', 'nn.Module')

function BinaryTanhActivation:__init(binWeights)
   Parent.__init(self)
   self.binWeights = binWeights or 'stoch'
end

function BinaryTanhActivation:_binSigmoid(x)
  return torch.cmax(torch.cmin((x+1)/2,1),0)
end
function BinaryTanhActivation:_binTanh(x)
  return self:_binSigmoid(x):mul(2):add(-1)
end

function BinaryTanhActivation:updateOutput(input)
  self.output:resizeAs(input)
  self.output = input:clone() -- non-destructive on input
  if self.binWeights == 'stoch' then
    local p = self:_binSigmoid(self.output)
    self.output[ p:ge(0.5) ] = 1
    self.output[ p:lt(0.5) ] = -1
  elseif self.binWeights == 'det' then
    --  make all positive values  1
    self.output[ input:gt(0) ] = 1;
    -- make all other values -1
    self.output[ input:le(0) ] = -1;
  end
  return self.output
end


function BinaryTanhActivation:updateGradInput(input, gradOutput)

  if self.gradInput then
    local nElement = self.gradInput:nElement()
    self.gradInput:resizeAs(input)
    if self.gradInput:nElement() ~= nElement then
      self.gradInput:zero()
    end

    self.gradInput:copy(gradOutput)
    self.gradInput = self:_binTanh(self.gradInput)

    return self.gradInput
  end
end
