
local EBPInputLayer, Parent = torch.class('nn.EBPInputLayer', 'nn.EBPLayer')

function EBPInputLayer:__init(inputSize, outputSize, opt)
  Parent.__init(self, inputSize, outputSize, opt)
  self.isFirstLayer = 1
  self.isLastLayer = 0
end