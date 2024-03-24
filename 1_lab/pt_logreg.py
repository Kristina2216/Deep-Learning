import torch.nn as nn
import numpy as np
import torch
import data

class PTLogreg(nn.Module):
    def __init__(self, D, C):
        """Arguments:
           - D: dimensions of each datapoint
           - C: number of classes
        """

        # inicijalizirati parametre (koristite nn.Parameter):
        # imena mogu biti self.W, self.b
        # ...
        super(PTLogreg, self).__init__()
        w = torch.randn(D, C)
        b = torch.zeros(C)
        self.w = nn.Parameter(w, requires_grad=True)
        self.b = nn.Parameter(b, requires_grad=True)




    def forward(self, X):

        # unaprijedni prolaz modela: izračunati vjerojatnosti
        #   koristiti: torch.mm, torch.softmax
        # ...
        mul = X.mm(self.w) + self.b
        s = nn.Softmax(dim=1)
        return s(mul)

    def get_loss(self, pred, Yoh_):
        # formulacija gubitka
        #   koristiti: torch.log, torch.mean, torch.sum
        # ...
        mul = Yoh_ * torch.log(pred)
        sum = torch.sum(Yoh_ * torch.log(X), dim=1)
        return -torch.mean(sum)

def train(model, X, Yoh_, param_niter, param_delta, param_lambda=1):
    """Arguments:
       - X: model inputs [NxD], type: torch.Tensor
       - Yoh_: ground truth [NxC], type: torch.Tensor
       - param_niter: number of training iterations
       - param_delta: learning rate
    """

    # inicijalizacija optimizatora
    optimizer = torch.optim.SGD(params=model.parameters(), lr=param_delta)

    # petlja učenja
    # ispisujte gubitak tijekom učenja
    for ep in range(param_niter):
        probs = model.forward(X)
        loss = model.get_loss(probs, Yoh_)
        loss += param_lambda * torch.norm(model.w) #regularization
        optimizer.step()
        optimizer.zero_grad()

        if ep % 100 == 0:
            print("Epoch num: " + str(ep) + " loss: " + str(loss))



def eval(model, X):
    """Arguments:
       - model: type: PTLogreg
       - X: actual datapoints [NxD], type: np.array
       Returns: predicted class probabilites [NxC], type: np.array
    """
    # ulaz je potrebno pretvoriti u torch.Tensor
    # izlaze je potrebno pretvoriti u numpy.array
    # koristite torch.Tensor.detach() i torch.Tensor.numpy()
    return model.forward(X).detach().numpy()



def oneHot(datapoints):
    claasN = np.max(datapoints) + 1
    data = list()

    for d in datapoints:
        t = np.zeros(classN)
        t[d] = 1
        data.append(t)

    return np.array(data)


np.random.seed(100)
X, Yoh_ = data.sample_gmm_2d(6, 2, 10)
X = torch.tensor(X, dtype=torch.float)
Yoh_ = torch.tensor(oneHot(Yoh_), dtype=torch.float)
logreg = PTLogreg(X.shape[1], Yoh_.shape[1])
train(logreg, X, Yoh_, 1000, 0.5, 10e-3)
probs = eval(logreg, X)