import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import os #für import export

class MyNet(nn.Module):

    #Layer Definieren
    def __init__(self):
        pass
        super(MyNet, self).__init__()
        self.lin1 = nn.Linear(10, 10)   #
        self.lin2 = nn.Linear(10, 10)

    # Aktivieren (also das NN)
    def forward(self, x):
        x = F.relu(self.lin1(x))
        x = self.lin2(x)
        return x


    def num_flat_features(self, x):
        size = x.size()[1:] #minibatch mit [1:]
        num = 1
        for i in size:
            num *= i
        return num

net = MyNet()
#net = net.cuda()  #für die GPU (Muss auch beim input gemacht werden)

if os.path.isfile('mynet.pt'):
    net = torch.load('mynet.pt')

for i in range (100):
    #input gen
    x = [1,0,0,1,0,0,0,1,1,0]
    input = Variable(torch.Tensor([x for _ in range(10)]))

    #output
    out = net(input)

    x = [0,1,1,0,1,1,1,0,0,1]
    target  = Variable(torch.Tensor([x for _ in range(10)]))
    criterion = nn.MSELoss()
    loss = criterion(out, target)
    print(loss)

    net.zero_grad() #lernt den fehler von letzten mal + den fehler des aktuellen
    loss.backward()
    optimizer = optim.SGD(net.parameters(), lr = 0.01) #SGD standart ADAM auch noch ok | lr = Lernrate
    optimizer.step()

# Speicher und Laden von sachen #

torch.save(net, 'mynet.pt')