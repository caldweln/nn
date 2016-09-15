local TanhScaled, parent = torch.class('nn.TanhScaled','nn.Module')

function TanhScaled:__init(_scale)
   parent.__init(self)
   self.scale = _scale or 1
end

function TanhScaled:updateOutput(input)
   input.THNN.TanhScaled_updateOutput(
      input:cdata(),
      self.scale,
      self.output:cdata()
   )
   return self.output
end

function TanhScaled:updateGradInput(input, gradOutput)
   input.THNN.TanhScaled_updateGradInput(
      input:cdata(),
      gradOutput:cdata(),
      self.scale,
      self.gradInput:cdata(),
      self.output:cdata()
   )
   return self.gradInput
end
