local Linear_CustomWeights, parent = torch.class('nn.Linear_CustomWeights', 'nn.Module')

function Linear_CustomWeights:__init(inputSize, outputSize, opt)
   parent.__init(self)

   self.weight = torch.Tensor(outputSize, inputSize)
   self.bias = torch.Tensor(outputSize)
   self.gradWeight = torch.Tensor(outputSize, inputSize)
   self.gradBias = torch.Tensor(outputSize)

   -- initialize
   if opt.arch.weightInit ~= nil and opt.arch.weightInit ~= '' then
     self:initWeights(opt.arch.weightInit)
   else
     self:reset()
   end
end

function Linear_CustomWeights:initWeights(formula)
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

function Linear_CustomWeights:reset(stdv)
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

function Linear_CustomWeights:updateOutput(input)
   if input:dim() == 1 then
      self.output:resize(self.bias:size(1))
      self.output:copy(self.bias)
      self.output:addmv(1, self.weight, input)
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
      self.output:addmm(0, self.output, 1, input, self.weight:t())
      self.output:addr(1, self.addBuffer, self.bias)
   else
      error('input must be vector or matrix')
   end

   return self.output
end

function Linear_CustomWeights:updateGradInput(input, gradOutput)
   if self.gradInput then

      local nElement = self.gradInput:nElement()
      self.gradInput:resizeAs(input)
      if self.gradInput:nElement() ~= nElement then
         self.gradInput:zero()
      end
      if input:dim() == 1 then
         self.gradInput:addmv(0, 1, self.weight:t(), gradOutput)
      elseif input:dim() == 2 then
         self.gradInput:addmm(0, 1, gradOutput, self.weight)
      end

      return self.gradInput
   end
end

function Linear_CustomWeights:accGradParameters(input, gradOutput, scale)
   scale = scale or 1
   if input:dim() == 1 then
      self.gradWeight:addr(scale, gradOutput, input)
      self.gradBias:add(scale, gradOutput)
   elseif input:dim() == 2 then
      self.gradWeight:addmm(scale, gradOutput:t(), input)
      self.gradBias:addmv(scale, gradOutput:t(), self.addBuffer)
   end
end

-- we do not need to accumulate parameters when sharing
Linear_CustomWeights.sharedAccUpdateGradParameters = Linear_CustomWeights.accUpdateGradParameters


function Linear_CustomWeights:__tostring__()
  return torch.type(self) ..
      string.format('(%d -> %d)', self.weight:size(2), self.weight:size(1))
end