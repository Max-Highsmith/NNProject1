#exp 3
import Network as nn 
param ={
'Epochs': 5,
'ExNum' : 3,
'T'     : 1000,
'TT'    : 20,
'L'     : 16,
'I'     : 25,
'J'     : 25,
'C'     : 28,
'D'     : 28,
'R'     : 10,
'lrA'   : 0.01,
'lrW'   : 0.01,
'ACT'   : 'tanhAct',
'LOSS'  : 'SquaredError',
'vis'   :  False,
'ORDER' : 'Random',
'INIT'  : 'AltUniform',  #FreezeTop converges, FreezeBot does not
'TrialNum':1
}

param['ACT'] = "tanhAct"
nn.runNetwork(**param)


