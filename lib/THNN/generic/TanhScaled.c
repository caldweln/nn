#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/TanhScaled.c"
#else

void THNN_(TanhScaled_updateOutput)(
          THNNState *state,
          THTensor *input,
          real scale,
          THTensor *output)
{
  THTensor_(resizeAs)(output, input);
  if (scale != 1)
  {
      THTensor_(mul)(input, input, scale);
  }
  THTensor_(tanh)(output, input);
}

void THNN_(TanhScaled_updateGradInput)(
          THNNState *state,
          THTensor *input,
          THTensor *gradOutput,
          real scale,
          THTensor *gradInput,
          THTensor *output)
{
  THTensor_(resizeAs)(gradInput, output);

  if (output->nDimension == 1 ||
      !THTensor_(isContiguous)(output) ||
      !THTensor_(isContiguous)(gradOutput) ||
      !THTensor_(isContiguous)(gradInput))
  {
    TH_TENSOR_APPLY3(real, gradInput, real, gradOutput, real, output,
      real z = *output_data;            \
      *gradInput_data = *gradOutput_data * (1. - z*z);
    );
  }
  else
  {
    real* ptr_gradOutput = THTensor_(data)(gradOutput);
    real* ptr_gradInput  = THTensor_(data)(gradInput);
    real* ptr_output     = THTensor_(data)(output);
    long i;

#pragma omp parallel for private(i)
    for (i = 0; i < THTensor_(nElement)(gradInput); i++)
    {
      real z = ptr_output[i];
      ptr_gradInput[i] = ptr_gradOutput[i] * (1. - z*z);
    }
  }

  if (scale != 1)
  {
      THTensor_(mul)(gradInput, gradInput, scale);
  }
}

#endif
