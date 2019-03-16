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
        """
        Parameters
        ----------
        - embed_size  : Size of embedding.
        - hidden_size : Number of nodes in the hidden layer.
        - vocab_size  : The size of vocabulary or output size.
        - num_layers  : Number of layers.
        
        """
        
        super(DecoderRNN, self).__init__()
        
        self.hidden_size = hidden_size
        
        # embedding layer that turns words into a vector of a specified size
        # self.embedding = nn.Embedding(output_size, hidden_size)
        self.word_embeddings = nn.Embedding(vocab_size, embed_size)
        
        # The LSTM takes Embedded image as inputs
        # and outputs hidden states of hidden_size
        # self.lstm = nn.LSTM(input_size, n_hidden, n_layers, 
        #                    dropout=drop_prob, batch_first=True)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers)
        
        # the linear layer that maps the hidden state output dimension 
        self.out = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, features, captions):
        """
        Parameters
        ----------
        - features : Contains the embedded image features
        - captions : Tensor corresponding to the last batch of captions .
        
        """
        captions = captions[:,:-1] # Reshaping from "torch.Size([10, 17])" to "torch.Size([10, 16])"
        
        # Initialize the hidden state
        batch_size = features.shape[0] # Notebook 1
        self.hidden = self.init_hidden(batch_size) 
        
        embeds = self.word_embeddings(captions)  #words
        
        # Concatenating features to embedding
        # torch.cat 3D tensors
        # imputs = torch.cat((input, hidden), 1)
        inputs = torch.cat((features.unsqueeze(1), embeds), 1)
        
        hiddens, _ = self.lstm(inputs)
        outputs = self.out(hiddens)
        return outputs
    
    def init_hidden(self,batch_size):
        """Returns the initial decoder state,
        conditioned on the final encoder state."""
        
        # The axes dimensions are (n_layers, batch_size, hidden_size)
        return (torch.zeros(1, batch_size, self.hidden_size),
                torch.zeros(1, batch_size, self.hidden_size))

    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        pass