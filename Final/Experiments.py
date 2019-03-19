import Network as nn
param =
{
'Epochs'  : 10,
'ExNum'   : 2,
'T'       : 32,    # num training samples
'TT'      : 98,    # num test samples
'L'       : 2,     # num layers
'I'       : 28,    # index of conv kernel height
'J'       : 28,    # index of conv kernel width
'C'       : 28,    # index of Image height
'D'       : 28,    # index of Image width
'R'       : 2,     # outpus of Perceptron
'lr'      : 0.1, # learning rate
'ACT'     : tanhAct,
'LOSS'    : SquaredError,
'vis'     : False
}

Activations = [nn.tanhAct, nn.sigAct, reluAct]
Loss        = [nn.SquaredError, nn.CrossEntropy, nn.SoftMax]

