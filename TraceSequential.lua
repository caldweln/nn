local TraceSequential, _ = torch.class('nn.TraceSequential', 'nn.Container')

function TraceSequential:__len()
   return #self.modules
end

function TraceSequential:add(module)
   if #self.modules == 0 then
      self.gradInput = module.gradInput
   end
   table.insert(self.modules, module)
   self.output = module.output
   return self
end

function TraceSequential:insert(module, index)
   index = index or (#self.modules + 1)
   if index > (#self.modules + 1) or index < 1 then
      error"index should be contiguous to existing modules"
   end
   table.insert(self.modules, index, module)
   self.output = self.modules[#self.modules].output
   self.gradInput = self.modules[1].gradInput
end

function TraceSequential:remove(index)
   index = index or #self.modules
   if index > #self.modules or index < 1 then
      error"index out of range"
   end
   table.remove(self.modules, index)
   if #self.modules > 0 then
       self.output = self.modules[#self.modules].output
       self.gradInput = self.modules[1].gradInput
   else
       self.output = torch.Tensor()
       self.gradInput = torch.Tensor()
   end
end

function TraceSequential:updateOutput(input)
   local currentOutput = input
   local updateTime = sys.clock()
   for i=1,#self.modules do
     updateTime = sys.clock()
      currentOutput = self.modules[i]:updateOutput(currentOutput)
      Log.write("<TraceSequential:updateOutput> time to updateOutput: " .. string.format("%.2f",(sys.clock() - updateTime)) .. "s "..tostring(self.modules[i]))
   end
   self.output = currentOutput
   return currentOutput
end

function TraceSequential:updateGradInput(input, gradOutput)
   local currentGradOutput = gradOutput
   local currentModule = self.modules[#self.modules]
   for i=#self.modules-1,1,-1 do
      local previousModule = self.modules[i]
      currentGradOutput = currentModule:updateGradInput(previousModule.output, currentGradOutput)
      currentModule = previousModule
   end
   currentGradOutput = currentModule:updateGradInput(input, currentGradOutput)
   self.gradInput = currentGradOutput
   return currentGradOutput
end

function TraceSequential:accGradParameters(input, gradOutput, scale)
   scale = scale or 1

   local currentGradOutput = gradOutput
   local currentModule = self.modules[#self.modules]
   for i=#self.modules-1,1,-1 do
      local previousModule = self.modules[i]
      currentModule:accGradParameters(previousModule.output, currentGradOutput, scale)
      currentGradOutput = currentModule.gradInput
      currentModule = previousModule
   end

   currentModule:accGradParameters(input, currentGradOutput, scale)
end

function TraceSequential:backward(input, gradOutput, scale)
   scale = scale or 1
   local currentGradOutput = gradOutput
   local currentModule = self.modules[#self.modules]
   for i=#self.modules-1,1,-1 do
      local previousModule = self.modules[i]
      currentGradOutput = currentModule:backward(previousModule.output, currentGradOutput, scale)
      currentModule.gradInput = currentGradOutput
      currentModule = previousModule
   end
   currentGradOutput = currentModule:backward(input, currentGradOutput, scale)
   self.gradInput = currentGradOutput
   return currentGradOutput
end

function TraceSequential:accUpdateGradParameters(input, gradOutput, lr)
   local currentGradOutput = gradOutput
   local currentModule = self.modules[#self.modules]
   for i=#self.modules-1,1,-1 do
      local previousModule = self.modules[i]
      currentModule:accUpdateGradParameters(previousModule.output, currentGradOutput, lr)
      currentGradOutput = currentModule.gradInput
      currentModule = previousModule
   end

   currentModule:accUpdateGradParameters(input, currentGradOutput, lr)
end


function TraceSequential:__tostring__()
   local tab = '  '
   local line = '\n'
   local next = ' -> '
   local str = 'nn.TraceSequential'
   str = str .. ' {' .. line .. tab .. '[input'
   for i=1,#self.modules do
      str = str .. next .. '(' .. i .. ')'
   end
   str = str .. next .. 'output]'
   for i=1,#self.modules do
      str = str .. line .. tab .. '(' .. i .. '): ' .. tostring(self.modules[i]):gsub(line, line .. tab)
   end
   str = str .. line .. '}'
   return str
end