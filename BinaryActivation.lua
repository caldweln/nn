local BinaryActivation, parent = torch.class('nn.BinaryActivation', 'nn.Module')

function BinaryActivation:__init(inputThresholdLows, inputThresholdHighs)
   parent.__init(self)
   self.inputThresholdLows = inputThresholdLows
   self.inputThresholdHighs = inputThresholdHighs
end

function BinaryActivation:updateOutput(input)

  self.output:resizeAs(input)

  for i=1,self.output:size(1) do
    self.output[i]:map2(self.inputThresholdLows, self.inputThresholdHighs,
      function(inputVal, inThreshLow, inThreshHigh)
        if inputVal > inThreshHigh then
          return 1
        elseif inputVal < inThreshLow then
          return -1
        else
          return 0
        end
      end)

    return self.output
  end

end

function BinaryActivation:updateGradInput(input, gradOutput)
    self.gradInput:resizeAs(gradOutput)
    self.gradInput:copy(gradOutput)
   return self.gradInput
end
