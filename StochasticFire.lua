local StochasticFire, Parent = torch.class('nn.StochasticFire', 'nn.Module')

function StochasticFire:__init()
   Parent.__init(self)
   self.inplace = false
end

function StochasticFire:updateOutput(input)
   if self.inplace then
      self.output = input
   else
      self.output:resizeAs(input):copy(input)
   end

   if self.train == false then
     self.output:round()
   else
     if type(self.output.cbernoulli) == 'function' then
       local neg = - self.output:clone()
       neg:cbernoulli()
       self.output:cbernoulli()
       self.output:add(-neg)
     else
       self.output:apply(function(x) if x < 0 then return -torch.bernoulli(-x) else return torch.bernoulli(x) end end)
     end
   end
   return self.output
end

function StochasticFire:updateGradInput(input, gradOutput)
   if self.train then
      if self.inplace then
         self.gradInput = gradOutput
      else
         self.gradInput:resizeAs(gradOutput):copy(gradOutput)
      end
   else
      error('backprop only defined while training')
   end
   return self.gradInput
end