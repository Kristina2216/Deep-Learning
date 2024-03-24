import numpy as np
import data

class fcann2:

    def __init__(self, H=5):
        self.H = H

    def transferFunction(self, s):
        return  s * (s > 0)

    def softMax(self, s):
        expS = np.exp(s - np.max(s))
        return expS / expS.sum(axis=0)

    def fcann2_train( self, X, Y_):
        # eksponencirane klasifikacijske mjere
        # pri računanju softmaksa obratite pažnju
        # na odjeljak 4.1 udžbenika
        # (Deep Learning, Goodfellow et al)!

        C=max(Y_) + 1
        iterNum=int(1e5)
        delta=0.05
        paramLambda = 1e-3

        w1 = 0.01*np.random.randn(self.H, X.shape[1]) # dimenzija H x broj varijabli (prvi sloj)
        w2 = 0.01*np.random.randn(C, self.H)

        b1= np.zeros(self.H)
        b2= np.zeros(C)

        for i in range(iterNum):
            scores1= np.dot(X, w1.transpose()) + b1
            h1=self.transferFunction(scores1)

            scores2= np.dot(h1, w2.transpose()) + b2
            probs=self.softMax(scores2)
            loss = np.mean(np.sum(-np.log(probs[range(probs.shape[0]), Y_] + 1e-13)))
            #log=-np.log(np.array(corr_pred))
            #loss = np.sum(log)/ X.shape[0]

            # dijagnostički ispis
            if i % 10 == 0:
                print("iteration {}: loss {}".format(i, loss))

            # derivacije komponenata gubitka po mjerama
            dL_ds2 = probs  # N x C
            dL_ds2[X.shape[0]-1, Y_] -= 1
            dL_ds2=dL_ds2/X.shape[0]

            # gradijenti parametara
            dL_dw2 = np.dot(h1.transpose(), dL_ds2)  # C x D (ili D x C) grad w2
            #dL_dw2[scores1<=0] = 0
            print(dL_dw2.shape)
            print(scores1.shape)

            dL_ds1=np.dot(dL_ds2, w2)
            dL_ds1[h1 <= 0] = 0
            dL_dw1=np.dot(dL_ds1.transpose(), X)

            #print(dL_dw1)
           # print(dL_dw2)

            db2 = dL_ds2.sum(axis=0)
            db1 = dL_ds1.sum(axis=0)

            # poboljšani parametri
            w1 += -delta * dL_dw1
            w2 += -delta * dL_dw2
            b1 += -delta * db1
            b2 += -delta * db2

        return w1, w2, b1, b2

np.random.seed(100)
X,Y_ = data.sample_gmm_2d(6, 2, 10)
nn = fcann2(H=5)
nn.fcann2_train(X, Y_)