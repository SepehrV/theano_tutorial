import numpy as np
import theano.tensor as T
import theano
# 1- Create shared Variables for weights and initialize them to random + symbolic variables

#N: number of training samples
#D: number of features
#H: number of hidden neurons
#C: number of classes
N, D, H, C= 10, 1000, 100, 1

x= T.vector('x')
y= T.scalar('y', dtype= 'int64')

w1= theano.shared(np.random.randn(D, H),name='w1')
w2= theano.shared(np.random.randn(H, C),name='w2')
b1= theano.shared(np.zeros((H,)), name= 'b1')
b2= theano.shared(np.zeros((C,)), name= 'b2')

# 2- Forward Pass Computation Graph
y1= T.tanh(T.dot(x,w1)+b1)
y2= T.dot(y1, w2)+b2
p_1 = 1 / (1 + T.exp(-y2))   # Probability that target = 1
prediction = p_1 > 0.5                    # The prediction thresholded
cent = -y * T.log(p_1) - (1-y) * T.log(1-p_1) # Cross-entropy loss function
cost = cent.mean()

# 3- Backward Pass Compute Gradients
dw1, dw2, db1, db2= T.grad(cost, [w1, w2, b1, b2])

# 4- Define your train function with the proper updates using sgd
lr= 0.001
train= theano.function(inputs=[x,y],
                       outputs=[cost, prediction],
                       updates= [(w1, w1-lr*dw1), (w2, w2-lr*dw2), (b1, b1-lr*db1), (b2, b2-lr*db2)])

# 5- Evaluate your train function
xx= np.array(np.random.randn(N, D), dtype="float32")
yy= np.random.randint(size= N, low= 0.0, high= C+1)

iters=100
for i in range(iters):
    for j in range(N):
        cost, preds = train(xx[j, :], yy[j])
        if i == iters-1:
            print('Target Label is: ', yy[j], 'Current predictions is: ', preds[0])
