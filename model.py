import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        
        super(DecoderRNN, self).__init__()
        
        self.hidden_size = hidden_size
        
        #self.embedding = nn.Embedding(output_size, hidden_size)
        self.embed = nn.Embedding(vocab_size, embed_size)
        
        # Define LSTM
        # The LSTM takes Embedded image as inputs, and outputs hidden states
        # nn.LSTM(input_size, hidden_size, num_layers)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers)
        
        #self.out = nn.Linear(hidden_size, output_size)
        self.out = nn.Linear(hidden_size, vocab_size)
        
    
    def forward(self, features, captions):
        
        captions = captions[:,:-1]       
        embeddings = self.embed(captions)              
        inputs = torch.cat((features.unsqueeze(1), embeddings), 1)
        hiddens, _ = self.lstm(inputs)
        outputs = self.out(hiddens)
        return outputs

    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        pass