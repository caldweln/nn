local NormalizedLinear, parent = torch.class('nn.NormalizedLinear', 'nn.Linear')

function NormalizedLinear:__init(inputSize, outputSize, mean, stdv)
  parent.__init(self,inputSize,outputSize)
  self:reset(mean,stdv)
end
  
  -- override the :reset method to use custom weight initialization.        
function NormalizedLinear:reset(mean,stdv)      
  if mean and stdv then
      self.weight:normal(mean,stdv)
      self.bias:normal(mean,stdv)
  else
    self.weight:normal(0,math.sqrt(6)/math.sqrt(self.weight:size(1)+self.weight:size(2)))
    self.bias:fill(0)
  end
end
--
--function NormalizedLinear:reset(stdv)
--   if stdv then
--      stdv = stdv * math.sqrt(3)
--   else
--      stdv = 1./math.sqrt(self.weight:size(2))
--   end
--   if nn.oldSeed then
--      for i=1,self.weight:size(1) do
--         self.weight:select(1, i):apply(function()
--            return torch.uniform(-stdv, stdv)
--         end)
--         self.bias[i] = torch.uniform(-stdv, stdv)
--      end
--   else
--      self.weight:uniform(-stdv, stdv)
--      self.bias:uniform(-stdv, stdv)
--   end
--
--   return self
--end