import torch.nn as nn
import torchkit.networks as nns
LSTM_name = "lstm"
GRU_name = "gru"
NCDE_name = "ncde"
RNNs = {
    LSTM_name: nn.LSTM,
    GRU_name: nn.GRU,
    NCDE_name: nns.NeuralCDE,
}
