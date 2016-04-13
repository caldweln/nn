
local PausableLinear, Parent = torch.class('nn.PausableLinear', 'nn.Linear')

function PausableLinear:__init(inputSize, outputSize)
   Parent.__init(self, inputSize, outputSize)
   self.pause = 0 -- pos value to prevent accGradParameters/updateParameters
end

function PausableLinear:accGradParameters(input, gradOutput, scale)
  if self.pause > 0 then return end -- do nothing
  Parent.accGradParameters(self, input, gradOutput, scale)
end

function PausableLinear:updateParameters(learningRate)
  if self.pause > 0 then return end -- do nothing
  Parent.updateParameters(self, learningRate)
end

function PausableLinear:__tostring__()
  local pauseStr = ''
  if self.pause > 0 then pauseStr = ' paused' end

  return torch.type(self) ..
      string.format('(%d -> %d)'..pauseStr, self.weight:size(2), self.weight:size(1))
end
