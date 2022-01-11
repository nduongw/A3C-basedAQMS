import torch.nn as nn
import torch
 
class Model(nn.Module):
    def __init__(self, Config):
        super().__init__()
        self.batch_size = Config.get('batch_size')
        self.Config = Config
        self.in_channels1 = Config.get('conv').get('in_channels1')
        self.in_channels2 = Config.get('conv').get('in_channels2')
        self.in_channels3 = Config.get('conv').get('in_channels3')

        self.out_channels1 = Config.get('conv').get('out_channels1')
        self.out_channels2 = Config.get('conv').get('out_channels2')
        self.out_channels3 = Config.get('conv').get('out_channels3')

        self.kernel_size1 = Config.get('conv').get('kernel_size1')
        self.kernel_size2 = Config.get('conv').get('kernel_size2')
        self.kernel_size3 = Config.get('conv').get('kernel_size3')

        self.input_size = Config.get('lstm').get('input_size')
        self.hidden_size = Config.get('lstm').get('hidden_size')
        self.num_layers = Config.get('lstm').get('num_layers')


        # 3 1D-Conv layer
        self.conv1 = nn.Conv2d(in_channels=self.in_channels1, out_channels=self.out_channels1, kernel_size=self.kernel_size1, padding='same')
        self.conv2 = nn.Conv2d(in_channels=self.in_channels2, out_channels=self.out_channels2, kernel_size=self.kernel_size2, padding='same')
        self.conv3 = nn.Conv2d(in_channels=self.in_channels3, out_channels=self.out_channels3, kernel_size=self.kernel_size3, padding='same')

        # batchnorm
        self.batchnorm1 = nn.BatchNorm1d(self.out_channels1)
        self.batchnorm2 = nn.BatchNorm1d(self.out_channels2)
        self.batchnorm3 = nn.BatchNorm1d(self.out_channels3)

        self.lstm = nn.LSTM(input_size= self.input_size, hidden_size=self.hidden_size, num_layers=self.num_layers, batch_first=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()

        self.fc_v = nn.Linear()
        self.fc_p = nn.Linear()
    
    def forward(self, x):

        hidden = self.init_hidden()
        out_conv1 = self.pool(self.relu(self.batchnorm1(self.conv1(x))))
        out_conv2 = self.pool(self.relu(self.batchnorm2(self.conv2(out_conv1))))
        out_conv3 = self.pool(self.relu(self.batchnorm3(self.conv3(out_conv2))))
        out_flatten = self.flatten(out_conv3)

        in_lstm = out_flatten.view(out_flatten.size()[0], 4, -1)
        out_lstm, hidden = self.lstm(in_lstm, hidden)

        out = out_lstm[:,]

        out_v = self.fc_v(out)
        out_p = self.fc_p(out).view(self.Config.get('roadLength'), self.Config.get('roadWidth'))

        return out_p, out_v

    

    def init_hidden(self) :
        h0 = torch.zeros((self.num_layers, self.batchsize,self.hidden_size))
        c0 = torch.zeros((self.num_layers, self.batchsize,self.hidden_size))
        hidden = (h0,c0)
        return hidden
