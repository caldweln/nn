
local BinaryConnectLayer, parent = torch.class('nn.BinaryConnectLayer', 'nn.Linear')

function BinaryConnectLayer:__init(inputSize, outputSize, binarize, mean, stdv)
  self.binarize = binarize or 0 -- binarize inputs and activations
  parent.__init(self,inputSize,outputSize)
  self:reset(mean,stdv)
end
  
  -- override the :reset method to use custom weight initialization.        
function BinaryConnectLayer:reset(mean,stdv)      
  if mean and stdv then
      self.weight:normal(mean,stdv)
      self.bias:normal(mean,stdv)
  else
    self.weight:normal(0,math.sqrt(6)/math.sqrt(self.weight:size(1)+self.weight:size(2)))
    self.bias:fill(0)
  end
end

function BinaryConnectLayer:updateOutput(input)
  
  -- binarize inputs
  if self.binarize > 0 then self:_binarize(input) end
  
  parent.updateOutput(self, input)
  
  -- binarize activations
  if self.binarize > 0 then self:_binarize(self.output) end
  
  return self.output
end


function BinaryConnectLayer:accGradParameters(input, gradOutput, scale)
  
  -- binarize inputs, if not already by updateOutput
  if self.binarize > 0 then self:_binarize(input) end
  
  parent.accGradParameters(self, input, gradOutput, scale)
  
end

function BinaryConnectLayer:_binarize(data)
  --  make all positive values  1 the others are zero
  data[ data:gt(0) ] = 1;
  data[ data:le(0) ] = 0;
  return data 
end