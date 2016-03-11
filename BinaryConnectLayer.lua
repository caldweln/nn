local BinaryConnectLayer, Parent = torch.class('nn.BinaryConnectLayer', 'nn.Linear')

function BinaryConnectLayer:__init(inputSize, outputSize, powerGlorot, opt)
  Parent.__init(self, inputSize, outputSize)
  self.verbose = opt.verbose
  -- binarized parameters for propagation
  self.rvWeight = torch.Tensor(outputSize, inputSize)
  self.binWeight = torch.Tensor(outputSize, inputSize)
  self.weightsDirty = 1 -- flag to update binWeights

  -- calculation of Glorot coefficients for use in scaling learningRate
  local coeffGlorot = 1 / math.sqrt( 1.5 / ( self.weight:size(2) + self.weight:size(1) ) )
  self.powerGlorot = powerGlorot or 0 -- Glorot disable => 0, adam optim => 1, sgd optim => 2
  self.scaleGlorot = math.pow(coeffGlorot,self.powerGlorot)

  self.binarization = opt.binarization or 'det' -- 'det' | 'stoch'
  -- pointer to real-valued weights by default
  self.weight = self.rvWeight
  -- initialize

  if opt.weightInitTbl then
    self:initWeights(opt.weightInitTbl.alpha,opt.weightInitTbl.beta,opt.weightInitTbl.gamma,opt.weightInitTbl.delta)
  else
    self:reset()
  end
end


function BinaryConnectLayer:initWeights(alpha, beta, gamma, delta)
  local u = alpha * math.pow(beta,gamma) + delta
  self.weight:uniform(-u, u)
  self.bias:uniform(-u, u)
  return self
end

function BinaryConnectLayer:_binSigmoid(x)
  return torch.cmax(torch.cmin((x+1)/2,1),0)
end

function BinaryConnectLayer:_binarize(data, threshold) -- inclusive threshold for +1
  local binTime = nil
  local result = data:clone() -- non-destructive
  if self.binarization == 'stoch' then
    binTime = sys.clock()
    threshold = threshold or 0.5
    local p = self:_binSigmoid(result)
    result[ p:ge(threshold) ] = 1
    result[ p:lt(threshold) ] = -1
  elseif self.binarization == 'det' then
    binTime = sys.clock()
    threshold = threshold or 0
    if type(result.snap) == 'function' then -- this is enough for our purposes
      result:snap(threshold, -1, threshold-1e-10, 1) --subtract v small number to approximate replace > with >= (carefull: replacing < with <= may have condition overlap)
      if self.verbose > 4 then print("<BinaryConnectLayer:_binarize> using 'Tensor.snap' to update tensor") end
    else
      result:apply(function(x) if x >= threshold then return 1 end return -1 end)
      if self.verbose > 4 then print("<BinaryConnectLayer:_binarize> using 'Tensor.apply' to update tensor") end
    end
  end
  if self.verbose > 4 then print("<BinaryConnectLayer:_binarize> time to binarize (" .. self.binarization .. "): " .. string.format("%.2f",(sys.clock() - binTime)) .. "s") end
  return result
end

function BinaryConnectLayer:updateOutput(input)
  if self.weightsDirty > 0 then
    local binTime = sys.clock()
    self.binWeight = self:_binarize(self.rvWeight)
    self.weightsDirty = 0
    if self.verbose > 4 then print("<BinaryConnectLayer:updateOutput> time to binarize: " .. string.format("%.2f",(sys.clock() - binTime)) .. "s") end

  end

  -- switch to binary weights for duration of upgradeGradInput
  self.weight = self.binWeight
  local updateTime = sys.clock()
  self.output = Parent.updateOutput(self, input)
  if self.verbose > 4 then print("<BinaryConnectLayer:updateOutput> time to update output: " .. string.format("%.2f",(sys.clock() - updateTime)) .. "s") end
  self.weight = self.rvWeight

  return self.output
end

function BinaryConnectLayer:updateGradInput(input, gradOutput)
  -- switch to binary weights for duration of upgradeGradInput
  self.weight = self.binWeight
  self.gradInput = Parent.updateGradInput(self, input, gradOutput)
  self.weight = self.rvWeight

  return self.gradInput
end

function BinaryConnectLayer:updateParameters(learningRate)
  -- update real-valued parameters
  Parent.updateParameters(self, learningRate * self.scaleGlorot)
  -- clip weights
  self.weight:clamp(-1,1)
  -- flag binWeights for update
  self.weightsDirty = 1

end
