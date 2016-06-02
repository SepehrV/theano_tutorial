import numpy as np
import pdb
import theano
import theano.tensor as T

class network():
    def __init__ (self):
        self.params = {}
        lr = np.array(0.001, dtype="float32")

        self.x = T.vector('x')
        self.y = T.scalar('y')

        self.RNN_out = self.RNN(self.x)
        self.loss = self.euc_loss(self.RNN(self.x), self.y)


        self.grads = T.grad(self.loss, wrt = list(self.params.values()) )

        gshared = [theano.shared(p.get_value() * np.array(0.0, dtype="float32"), name='%s_grad' % k) for k, p in self.params.items()]

        gsup = [(gs, g) for gs, g in zip(gshared, self.grads)]
        pup = [(param, param - lr*g) for param, g in zip(self.params.values(), gshared)]
        self.f_forward = theano.function([self.x], outputs=[self.RNN_out])

        self.train = theano.function([self.x, self.y], outputs=[self.RNN_out, self.loss], updates=gsup + pup)


    def RNN(self, X):
        self.params['W'] = theano.shared(value=np.array(np.random.rand(), dtype="float32"), name= 'W')
        self.params['U'] = theano.shared(value=np.array(np.random.rand(), dtype="float32"), name= 'U')
        self.params['b'] = theano.shared(value=np.array(np.random.rand(), dtype="float32"), name= 'b')
        self.params['Wo'] = theano.shared(value=np.array(np.random.rand(), dtype="float32"), name= 'Wo')

        n_steps = X.shape[0]
        def step(x, _h):
            h = T.tanh(self.params['W']*_h + self.params['U']*x + self.params['b'])
            y = self.params['Wo'] * h

            return y, h


        results, update = theano.scan(step, sequences = X, outputs_info =[None,  np.array(0.0, dtype="float32")], n_steps = n_steps)
        return results[0][-1]


    def euc_loss(self, inp, label):
        return (inp-label)**2


def dummy(x):
    return x.sum()


def main():
    net = network()
    max_iter = 100000
    disp_freq = 2000

    loss = np.zeros(disp_freq)
    for i in range(max_iter):
        x = np.array(np.random.rand(2), dtype="float32")
        y = dummy(x)
        loss[i%disp_freq] = net.train(x,y)[1]
        if i%disp_freq == 0:
            print ("loss at iter%s = %s"%(i,loss.mean()))




if __name__ == "__main__":
    main()
