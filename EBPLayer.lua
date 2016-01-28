--require 'nn'
require 'cephes'

local EBPLayer, Parent = torch.class('nn.EBPLayer', 'nn.Module')


function EBPLayer:__init(inputSize, outputSize, opt)
  Parent.__init(self)

  self.weight = torch.Tensor(outputSize, inputSize)
  self.bias = torch.Tensor(outputSize,1)
  self.isFirstLayer = 0
  self.isLastLayer = 0
  self.inputSize = inputSize
  if opt.sigma and opt.sigma > 0 then
    self:reset(1,opt.sigma)  -- original method
  elseif opt.alpha and opt.alpha > 0 then
    self:reset(opt.alpha) -- sigmoid or other methods
  else
    self:reset()  -- tanh method
  end
end


function EBPLayer:reset(alpha, stdv)

  local size = self.weight:size(1) + self.weight:size(2)
  if stdv then  -- original 
     size = self.weight:size(2) -- use only fan in
     stdv =  0.5 * math.sqrt( stdv * (12 /size) )
  else
    if not alpha then
      alpha = 1 -- corresponds to tanh method
    end
    stdv = alpha * math.sqrt(6/size)
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

-- forward 
function EBPLayer:updateOutput(input)
  collectgarbage()
  local T =  input:size(2)

  if self.train then  -- the network is in train mode
    local tanh_weight = torch.tanh(self.weight);

    if self.isFirstLayer == 1 then -- If first Layer we calculate differently
      self.mean_v = input

      -- self.mean_u = bsxfun(@plus, math.tanh(self.wieght) * self.mean_v, self.bias) / math.sqrt( self.inputSize + 1 )
      self.mean_u =  ( tanh_weight * self.mean_v + torch.expand(self.bias, self.bias:size(1) ,T) ) / math.sqrt( self.inputSize + 1 )

      -- self.var_u     = ( ( 1 -  math.tanh(self.wieght) .^ 2 ) * ( self.mean_v .^ 2 ) + 1 ) / ( self.inputSize + 1 );     
      self.var_u  = torch.add(-torch.cmul( tanh_weight, tanh_weight ),1) * torch.cmul( self.mean_v , self.mean_v )
      self.var_u:add(1):div( self.inputSize + 1 )

    else

      -- self.mean_v = 2 * input - 1
      self.mean_v = torch.mul(input,2) 
      self.mean_v:add(-1)

      -- self.var_v =  4 * (input - input .^ 2 )
      self.var_v = input - torch.cmul(input,input)
      self.var_v:mul(4)

      -- self.mean_u = bsxfun(@plus, math.tanh(wieght) * self.mean_v, self.bias) / math.sqrt( self.inputSize + 1 )
      self.mean_u =  ( tanh_weight * self.mean_v + torch.expand(self.bias,self.bias:size(1),T) ) / math.sqrt( self.inputSize + 1 )

      -- self.var_u  = bsxfun(@plus, sum(self.var_v, 1), (1 - math.tanh(wieght)  .^ 2) * (1 - self.var_v) + 1 ) / ( self.inputSize + 1 )      
      -- torch.sum(self.var_v, 1)  this is a row matrix we need to replicate it n many times where n is the first dimension of tmp
      local temp = torch.add(-torch.cmul( tanh_weight, tanh_weight ),1) * torch.add(-self.var_v,1)
      temp:add(1)

      self.var_u = ( temp + torch.repeatTensor( torch.sum(self.var_v, 1), temp:size(1) ,1) ) /  ( self.inputSize + 1 )
    end

    -- self.output    = normcdf(self.mean_u ./ sqrt(self.var_u), 0, 1)   
    self.output  = normcdf( torch.cdiv( self.mean_u, torch.sqrt(self.var_u)) , 0, 1)

  else -- the network is in evaluate mode
    self.output =  torch.sign(self.weight)  * input + torch.repeatTensor(self.bias,1,T) 
    if self.isLastLayer == 0 then 
      self.output = torch.sign(self.output) -- we take sign if it is not last layer
    end 
  end

  return self.output 
end

-- backward
function EBPLayer:updateGradInput(input, gradOutput)
  -- input is the output of the layer below
  collectgarbage()
  if self.gradInput then

    local npdf = normpdfm(0, self.mean_u , torch.sqrt(self.var_u))
    local Gi = 0

    -- gradOutput is the target variables for the last layer
    if self.isLastLayer == 1 then -- If last Layer we calculate differently
      -- Gi          = 2 * ( normpdf(0, mean_u, sqrt(var_u)) ./ normcdf(0, -Y .* mean_u, sqrt(var_u)) ) / sqrt( K(ll) + 1 );
      local ncdf = normcdfm(0, torch.cmul( -gradOutput, self.mean_u ) , torch.sqrt(self.var_u))
      Gi = torch.cdiv(npdf,ncdf)
      Gi:mul(2):div( math.sqrt(self.inputSize + 1) ) 

      -- check for not finite values 
      local mask  = torch.add( torch.add( Gi:ne(Gi) ,  Gi:eq(math.huge) ),  Gi:eq(-math.huge) )
      if torch.any(mask) then  -- if any of the elements is true
        local out = gradOutput:maskedSelect( mask ) 
        local mu = self.mean_u:maskedSelect( mask )
        local vu =  self.var_u:maskedSelect( mask )  

        out:cmul(mu)
        --  make all positive values  0
        out[ out:gt(0) ] = 0; 
        -- make all negative values  1
        out[ out:lt(0) ] = 1;       
        out:mul(-2):cmul( torch.cdiv(mu,vu) ):div( math.sqrt(self.inputSize + 1) )

        --  put back the values in out to original Gi
        Gi:maskedCopy(mask, out)
      end  
    else
      -- Gi  = 2 * normpdf(0, mean_u, sqrt(var_u)) / sqrt( K(ll) + 1 );
      Gi = torch.mul(npdf,2)
      Gi:div( math.sqrt(self.inputSize + 1) ) 
    end 

    local temp = torch.cmul( gradOutput , Gi )
    self.gradWeight = temp
    self.gradBias = torch.sum(temp,2)

    -- gradInput is delta 
    self.gradInput = torch.tanh(self.weight):t() * temp

    return self.gradInput
  end
end


function EBPLayer:accGradParameters(input, gradOutput, scale)
  --scale = scale or 1
  collectgarbage()
  local tmp = self.gradWeight * self.mean_v:t()
  self.gradWeight = torch.mul( tmp, 0.5 )
  self.gradBias = torch.mul( self.gradBias, 0.5 )
end


function EBPLayer:__tostring__()
  return torch.type(self) ..
    string.format('(%d -> %d)', self.weight:size(2), self.weight:size(1))
end

-- Never change the value of a parameter it changes globally
function normcdf(x, mu, sigma) 
  -- mu and sigma are numbers
  -- x is Tensor

  -- returns  0.5 * (1.0 + cephes.erf((x-mu)/math.sqrt(2*sigma*sigma)))
  local funcOutput  = cephes.erf((x-mu)/math.sqrt(2*sigma*sigma))
  funcOutput = funcOutput:type(torch.getdefaulttensortype()) -- convert it to the default tensor Type
  funcOutput:resizeAs(x):add(1):mul(0.5)
  return funcOutput
end

function normcdfm(x, mu, sigma)  
  -- mu and sigma are Tensors
  -- x is number

  -- replicate x to match the sizes of mu and sigma
  local x_in = torch.Tensor(mu:size()):fill(x):add(-mu) 
  local div = torch.sqrt( torch.mul( torch.cmul(sigma,sigma),2) )
  -- returns  0.5 * (1.0 + cephes.erf((x-mu)/math.sqrt(2*sigma*sigma)))
  local funcOutput  = cephes.erf( torch.cdiv(x_in,div) )
  funcOutput = funcOutput:type(torch.getdefaulttensortype()) -- convert it to the default tensor Type
  funcOutput:resizeAs(mu):add(1):mul(0.5)
  return funcOutput
end


function normpdfm(x, mu, sigma) 
  -- mu and sigma are Tensors
  -- x is number

  -- replicate x to match the sizes of mu and sigma
  local x_in = torch.Tensor(mu:size()):fill(x):add(-mu) 
  local sigsq = torch.cmul(sigma,sigma);
  local funcOutput = cephes.exp ( x_in:cmul(x_in):mul(-0.5):cdiv( sigsq ))
  funcOutput = funcOutput:type(torch.getdefaulttensortype()) -- convert it to the default tensor Type
  funcOutput:resizeAs(mu):cdiv( torch.sqrt( sigsq:mul(2):mul(math.pi) )  )
  return funcOutput
    -- return cephes.exp(-.5 * (x-mu)*(x-mu)/(sigma*sigma)) / math.sqrt(2.0*math.pi*sigma*sigma)
end

