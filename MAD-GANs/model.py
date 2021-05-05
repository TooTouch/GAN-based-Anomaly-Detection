import torch.nn as nn 


class LSTMGenerator(nn.Module):
    """
    LSTM based generator. 
    """

    def __init__(self, nb_features):
        """
        Arguments
        ---------

        nb_features: the number of features
        """
        super().__init__()
        self.nb_features = nb_features
        self.hidden_size = 100 # MAD-GAN uses 100 hidden size
        self.lstm_depth = 3 # MAD-GAN uses 3 layers in LSTM

        self.lstm = nn.LSTM(input_size  = self.nb_features, 
                            hidden_size = self.hidden_size, 
                            num_layers  = self.lstm_depth, 
                            batch_first = True)
        
        self.linear = nn.Sequential(
            nn.Linear(in_features  = self.hidden_size, 
                      out_features = self.nb_features), 
            nn.Tanh()
        )

    def forward(self, data):
        batch_size, seq_len, _ = data.size()
        
        outputs, _ = self.lstm(data)
        
        outputs = self.linear(outputs.contiguous().view(batch_size*seq_len, self.hidden_size))
        outputs = outputs.view(batch_size, seq_len, -1)
        return outputs


class LSTMDiscriminator(nn.Module):
    """
    LSTM based discriminator
    """

    def __init__(self, nb_features):
        super().__init__()
        self.nb_features = nb_features
        self.hidden_size = 100 # MAD-GAN uses 100 hidden size
        
        self.lstm = nn.LSTM(input_size  = self.nb_features, 
                            hidden_size = self.hidden_size, 
                            num_layers  = 1, 
                            batch_first = True)

        self.linear = nn.Sequential(
            nn.Linear(in_features  = self.hidden_size, 
                      out_features = 1), 
            nn.Sigmoid()
        )

    def forward(self, data):
        batch_size, seq_len, _ = data.size()

        recurrent_features, _ = self.lstm(data)
        outputs = self.linear(recurrent_features.contiguous().view(batch_size*seq_len, self.hidden_size))
        outputs = outputs.view(batch_size, seq_len, -1)
        return outputs