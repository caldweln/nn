local TernSpatialConvolution, parent = torch.class('nn.TernSpatialConvolution', 'nn.Module')

function TernSpatialConvolution:__init(nInputPlane, nOutputPlane, kW, kH, dW, dH, padW, padH)
   parent.__init(self)

   dW = dW or 1
   dH = dH or 1

   self.nInputPlane = nInputPlane
   self.nOutputPlane = nOutputPlane
   self.kW = kW
   self.kH = kH

   self.dW = dW
   self.dH = dH
   self.padW = padW or 0
   self.padH = padH or self.padW

   self.weight = torch.Tensor(nOutputPlane, nInputPlane, kH, kW):zero()
   self.biasHi = torch.Tensor(nOutputPlane):zero()
   self.biasLo = torch.Tensor(nOutputPlane):zero()
   self.bias  = torch.Tensor(nOutputPlane):zero()
   
   self.gradWeight = torch.Tensor(nOutputPlane, nInputPlane, kH, kW):zero()
   self.gradBias = torch.Tensor(nOutputPlane):zero()

   self:reset()
end

function TernSpatialConvolution:reset(stdv)
   self.weight:zero()
   self.biasHi:zero()
   self.biasLo:zero()
   self.bias:zero()
end

local function backCompatibility(self)
   self.finput = self.finput or self.weight.new()
   self.fgradInput = self.fgradInput or self.weight.new()
   if self.padding then
      self.padW = self.padding
      self.padH = self.padding
      self.padding = nil
   else
      self.padW = self.padW or 0
      self.padH = self.padH or 0
   end
   if self.weight:dim() == 2 then
      self.weight = self.weight:view(self.nOutputPlane, self.nInputPlane, self.kH, self.kW)
   end
   if self.gradWeight and self.gradWeight:dim() == 2 then
      self.gradWeight = self.gradWeight:view(self.nOutputPlane, self.nInputPlane, self.kH, self.kW)
   end
end

local function makeContiguous(self, input, gradOutput)
   if not input:isContiguous() then
      self._input = self._input or input.new()
      self._input:resizeAs(input):copy(input)
      input = self._input
   end
   if gradOutput then
      if not gradOutput:isContiguous() then
	 self._gradOutput = self._gradOutput or gradOutput.new()
	 self._gradOutput:resizeAs(gradOutput):copy(gradOutput)
	 gradOutput = self._gradOutput
      end
   end
   return input, gradOutput
end

-- function to re-view the weight layout in a way that would make the MM ops happy
local function viewWeight(self)
   self.weight = self.weight:view(self.nOutputPlane, self.nInputPlane * self.kH * self.kW)
   if self.gradWeight and self.gradWeight:dim() > 0 then 
      self.gradWeight = self.gradWeight:view(self.nOutputPlane, self.nInputPlane * self.kH * self.kW)
   end
end

local function unviewWeight(self)
   self.weight = self.weight:view(self.nOutputPlane, self.nInputPlane, self.kH, self.kW)
   if self.gradWeight and self.gradWeight:dim() > 0 then 
      self.gradWeight = self.gradWeight:view(self.nOutputPlane, self.nInputPlane, self.kH, self.kW)
   end
end

function TernSpatialConvolution:updateOutput(input)
   backCompatibility(self)
   viewWeight(self)
   input = makeContiguous(self, input)
   local out = input.nn.SpatialConvolutionMM_updateOutput(self, input)
   unviewWeight(self)

  for i=1,out:size(1) do
   for j=1,out:size(2) do
    local mask1 = out[i][j]:gt( self.biasHi[j]  )
    local mask2 = out[i][j]:lt( self.biasLo[j]  )
    out[i][j]:maskedFill( mask1,1)
    out[i][j]:maskedFill( mask2,-1)
    out[i][j]:apply(function(x) if (x ~= -1 and x ~= 1)  then return 0 else return x end end)
   end
  end
  
   return out
end

function TernSpatialConvolution:updateGradInput(input, gradOutput)
   if self.gradInput then
      backCompatibility(self)
      viewWeight(self)
      input, gradOutput = makeContiguous(self, input, gradOutput)
      local out = input.nn.SpatialConvolutionMM_updateGradInput(self, input, gradOutput)
      unviewWeight(self)
      return out
   end
end

function TernSpatialConvolution:accGradParameters(input, gradOutput, scale)
   backCompatibility(self)
   input, gradOutput = makeContiguous(self, input, gradOutput)
   viewWeight(self)
   local out = input.nn.SpatialConvolutionMM_accGradParameters(self, input, gradOutput, scale)
   unviewWeight(self)
   return out
end

function TernSpatialConvolution:type(type,tensorCache)
   self.finput = torch.Tensor()
   self.fgradInput = torch.Tensor()
   return parent.type(self,type,tensorCache)
end

function TernSpatialConvolution:__tostring__()
   local s = string.format('%s(%d -> %d, %dx%d', torch.type(self),
         self.nInputPlane, self.nOutputPlane, self.kW, self.kH)
   if self.dW ~= 1 or self.dH ~= 1 or self.padW ~= 0 or self.padH ~= 0 then
     s = s .. string.format(', %d,%d', self.dW, self.dH)
   end
   if (self.padW or self.padH) and (self.padW ~= 0 or self.padH ~= 0) then
     s = s .. ', ' .. self.padW .. ',' .. self.padH
   end
   return s .. ')'
end

