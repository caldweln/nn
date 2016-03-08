
local BinaryOutputLinear, Parent = torch.class('nn.BinaryOutputLinear', 'nn.Linear')

function BinaryOutputLinear:__init(inputSize, outputSize)
   Parent.__init(self, inputSize, outputSize)
   -- binarized parameters for propagation
   self.rvWeight = torch.Tensor(outputSize, inputSize)
   self.binWeight = torch.Tensor(outputSize, inputSize)
   self.weightsDirty = 1 -- flag to update binWeights
   -- pointer to real-valued weights by default
   self.weight = self.rvWeight
  -- initialize
   self:reset()
end

function BinaryOutputLinear:updateOutput(input)
  if not self.train then
    if self.weightsDirty > 0 then
      self.binWeight = torch.sign(self.rvWeight)
      self.weightsDirty == 0
    end
    self.weight = self.binWeight
  end

  self.output = Parent.updateOutput(input)

  if not self.train then
    self.weight = self.rvWeight
  end

  return self.output
end


function BinaryOutputLinear:updateParameters(learningRate)
  -- update real-valued parameters
  Parent.updateParameters(self, learningRate)
  -- flag binWeights for update
  self.weightsDirty = 1
end
