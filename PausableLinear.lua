
local PausableLinear, Parent = torch.class('nn.PausableLinear', 'nn.Linear')

function PausableLinear:__init(inputSize, outputSize, opt)
   Parent.__init(self, inputSize, outputSize)
   self.pause = 0 -- pos value to prevent accGradParameters/updateParameters
   self.isRecording = 0
   self.inputRecord = nil

   -- initialize
   if opt.arch.weightInit ~= nil and opt.arch.weightInit ~= '' then
     self:initWeights(opt.arch.weightInit)
   end
end

function PausableLinear:initWeights(formula)
  self.fanin = self.weight:size(2)
  self.fanout = self.weight:size(1)
  formula = string.gsub(formula,'fanin',self.fanin)
  formula = string.gsub(formula,'fanout',self.fanout)
  local u = tonumber(formula) or loadstring('return '..formula)
  if type(u) == 'function' then
    u = u()
  end
  self.weight:uniform(-u, u)
  self.bias:uniform(-u, u)
  return self
end

function PausableLinear:updateOutput(input)
  if self.isRecording > 0 and self.train == true then
    if self.inputRecord ~= nil then
      self.inputRecord = torch.cat(self.inputRecord, input, 1)
    else
      self.inputRecord = input:clone()
    end
  end
  return Parent.updateOutput(self, input)
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
