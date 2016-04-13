
local PausableLinear, Parent = torch.class('nn.PausableLinear', 'nn.Linear')

function PausableLinear:__init(inputSize, outputSize)
   Parent.__init(self, inputSize, outputSize)
   self.pause = 0 -- pos value to prevent accGradParameters/updateParameters
end

function PausableLinear:pause()
  self.pause = 1
end

function PausableLinear:resume()
  self.pause = 0
end

function PausableLinear:accGradParameters(input, gradOutput, scale)
  if self.pause > 0 then return end -- do nothing
  Parent.accGradParameters(input, gradOutput, scale)
end

function PausableLinear:updateParameters(learningRate)
  if self.pause > 0 then return end -- do nothing
  Parent.updateParameters(self, learningRate)
end
