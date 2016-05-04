
local RecordableLinear, Parent = torch.class('nn.RecordableLinear', 'nn.Linear')

function RecordableLinear:__init(inputSize, outputSize, opt)
   Parent.__init(self, inputSize, outputSize)
   self.isRecording = 0
   self.inputRecord = nil
   self.outputRecord = nil

   -- initialize
   if opt.arch.weightInit ~= nil and opt.arch.weightInit ~= '' then
     self:initWeights(opt.arch.weightInit)
   end
end

function RecordableLinear:initWeights(formula)
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

function RecordableLinear:updateOutput(input)
  if self.isRecording > 0 then -- record input
    if self.inputRecord ~= nil then
      self.inputRecord = torch.cat(self.inputRecord, input, 1)
    else
      self.inputRecord = input:clone()
    end
  end

  local output = Parent.updateOutput(self, input)

  if self.isRecording > 0 then -- record output
    if self.outputRecord ~= nil then
      self.outputRecord = torch.cat(self.outputRecord, output, 1)
    else
      self.outputRecord = output:clone()
    end
  end

  return output
end


function RecordableLinear:__tostring__()
  local pauseStr = ''
  if self.isRecording > 0 then pauseStr = ' recording' end

  return torch.type(self) ..
      string.format('(%d -> %d)'..pauseStr, self.weight:size(2), self.weight:size(1))
end
