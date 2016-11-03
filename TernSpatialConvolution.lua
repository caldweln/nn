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

   -- flag per neuron to indicate output negation required before thresholding
   self.inversion = torch.Tensor(nOutputPlane):zero()

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
   input.THNN.SpatialConvolutionMM_updateOutput(
      input:cdata(),
      self.output:cdata(),
      self.weight:cdata(),
      self.bias:cdata(),
      self.finput:cdata(),
      self.fgradInput:cdata(),
      self.kW, self.kH,
      self.dW, self.dH,
      self.padW, self.padH
   )
   unviewWeight(self)

   if type(self.output.snapd) == 'function' then -- this is enough for our purposes
     for j=1,self.output:size(2) do
       self.output[{{},j}] = (self.inversion ~= nil and self.inversion[j] > 0) and torch.mul(self.output[{{},j}],-1) or self.output[{{},j}] -- inefficient
       self.output[{{},j}]:snapd(self.biasLo[j], -1, self.biasHi[j], 1, 0)
     end
   else

    for i=1,self.output:size(1) do
     for j=1,self.output:size(2) do
      local mask1 = self.output[i][j]:gt( self.biasHi[j]  )
      local mask2 = self.output[i][j]:lt( self.biasLo[j]  )
      local mask12 = torch.add(mask1,mask2)
      mask12[mask12:gt(0)] = 1
      local mask3 = torch.pow((mask12-1),2)
      self.output[i][j]:maskedFill( mask1,1)
      self.output[i][j]:maskedFill( mask2,-1)
      self.output[i][j]:maskedFill( mask3,0)
     end
    end
  end

   return self.output
end

function TernSpatialConvolution:updateGradInput(input, gradOutput)
   if self.gradInput then
      self.gradInput:resizeAs(input):zero()
      return self.gradInput
   end
end

function TernSpatialConvolution:updateParameters(learningRate)
end

function TernSpatialConvolution:type(type,tensorCache)
   self.finput = self.finput and torch.Tensor()
   self.fgradInput = self.fgradInput and torch.Tensor()
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

function TernSpatialConvolution:clearState()
   nn.utils.clear(self, 'finput', 'fgradInput', '_input', '_gradOutput')
   return parent.clearState(self)
end
