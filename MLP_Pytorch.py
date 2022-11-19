import torch
import torch.nn.functional as F

import numpy as np
import matplotlib.pyplot as plt


class MLP_Pytorch():
    def __init__(self, S: list, dtype=torch.float32):
        #####assert len(bias)==len(self.S)-1, "Se necesita el mismo número de bias que matrices de pesos."
        self.dtype = dtype
        self.device = self.SelectDevice()
        self.S = S

        # Ver donde situar la desviación estandar, la tengo en 1.
        self.std = 1
        self.W = [torch.normal(0, self.std, (self.S[i], self.S[i+1]), device=self.device, requires_grad=True, dtype=self.dtype) for i in range(len(self.S)-1)]
        self.bias = [torch.zeros(self.S[i+1], device=self.device, requires_grad=True, dtype=self.dtype).reshape(1, self.S[i+1]) for i in range(len(self.S)-1)]

    def Forward(self, X):
        y_aux = X.reshape(1, len(X))
        for l in range(len(self.W)-1):
            y_aux = self.ReLU(y_aux.mm(self.W[l]) + self.bias[l])
        y_pred = self.Softmax(y_aux.mm(self.W[-1]) + self.bias[-1])
        
        return y_pred


    def Train(self, X, Y, lr, n_epochs):
        """ Esta función será usada para poder entrenar la red neuronal con los datos
        de entrada X y los datos de objetivo Y."""
        loss_epochs = []
        for e in range(1, n_epochs+1):
            # Hacemos el forward pass para encontrar el valor de la predicción del modelo.
            y_pred = self.Forward(X)

            # Calculamos la pérdida utilizando la función cross entropy.
            loss = self.cross_entropy(y_pred, Y)
            loss_epochs.append(loss.item())

            # Back-propagation (Calculamos todos los gradientes automaticamente)
            loss.backward()

            with torch.no_grad():
                for l in range(len(self.W)):
                    # Actualizamos los pesos y bias de nuestro modelo.
                    self.W[l] -= lr * self.W[l].grad
                    self.bias[l] -= lr * self.bias[l].grad
                for l in range(len(self.W)):
                    # Ponemos los gradientes en cero para que no se acumulen.
                    self.W[l].grad.zero_()
                    self.bias[l].grad.zero_()
            
            print(f"Epoch {e}/{n_epochs}. Loss {np.mean(loss_epochs):.5f}")
        plt.plot(list(range(n_epochs)), loss_epochs)
        plt.show()
            
    
    def cross_entropy(self, output, target):
        return F.cross_entropy(input=output, target=target)

    def ReLU(self, x):
        return x.clamp(min=0)
    
    def Softmax(self, x):
        #return torch.softmax(x, dim=-1)
        return torch.exp(x) / torch.exp(x).sum(axis=-1, keepdims=True)

    def SelectDevice(self):
        if torch.cuda.is_available():
            return torch.device('cuda')
        else:
            return torch.device('cpu')



from torch.utils.data import DataLoader # Para dividir nuestros datos
from torch.utils.data import sampler # Para muestrar datos
import torchvision.datasets as dataset # Para importar DataSets
import torchvision.transforms as T # Para aplicar transformaciones a nuestros datos
# No es importante, sólo si tienen Jupyter Themes. Nothing to do with Deep Learning
from jupyterthemes import jtplot
jtplot.style()


NUM_TRAIN = 55000
BATCH_SIZE = 512

# Get our training, validation and test data.
# data_path = '/media/josh/MyData/Databases/' #use your own data path, you may use an existing data path to avoid having to download the data again.
data_path = 'mnist'     # Elegimos un directorio donde se va a guardar los datos de MNIST en caso de no tenerlos descargados ya.
mnist_train = dataset.MNIST(data_path, train=True, download=True, transform=T.ToTensor())
loader_train = DataLoader(mnist_train, batch_size=BATCH_SIZE, sampler=sampler.SubsetRandomSampler(range(NUM_TRAIN)))

mnist_val = dataset.MNIST(data_path, train=True, download=True, transform=T.ToTensor())
loader_val = DataLoader(mnist_val, batch_size=BATCH_SIZE, sampler=sampler.SubsetRandomSampler(range(NUM_TRAIN, 60000)))

mnist_test = dataset.MNIST(data_path, train=False, download=True, transform=T.ToTensor())
loader_test = DataLoader(mnist_test, batch_size=BATCH_SIZE)
""" Training Set: Son los datos de entrenamiento y vamos a querer muchos de ellos.
    ###Evaluation Set: No me queda claro con la diferencia del Training Set.
    Test Set: Son datos que la red neuronal NUNCA vió antes, para estimar el error que comete."""


MLP = MLP_Pytorch([784, 1000, 1000, 10])
'''
for i, (x, y) in enumerate(loader_train):
    x = x.reshape(len(x), 784)
    for i in range(len(x)):
        target = y[0].reshape(1)
        #target = torch.zeros(10, device=MLP.device, dtype=torch.float)
        #target[y[i]] = 1
        MLP.Train(x[i], target, 0.3, 5)
'''


x_train = loader_train.dataset.data
y_train = loader_train.dataset.targets
print(x_train.shape)
print(y_train.shape)
print(y_train[0:5])
