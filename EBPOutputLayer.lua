
local EBPOutputLayer, Parent = torch.class('nn.EBPOutputLayer', 'nn.EBPLayer')

function EBPOutputLayer:__init(inputSize, outputSize, opt)
  Parent.__init(self, inputSize, outputSize, opt)
  self.isFirstLayer = 0
  self.isLastLayer = 1
end
