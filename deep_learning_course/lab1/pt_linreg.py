import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np


## Definicija računskog grafa
# podaci i parametri, inicijalizacija parametara
a = torch.randn(1, requires_grad=True)
b = torch.randn(1, requires_grad=True)

X = torch.tensor([1, 2, 3, 4])
Y = torch.tensor([3, 5, 7, 9])

# optimizacijski postupak: gradijentni spust
optimizer = optim.SGD([a, b], lr=0.01)

for i in range(500):
    # afin regresijski model
    Y_ = a*X + b

    diff = (Y - Y_)

    # kvadratni gubitak
    loss = torch.sum(diff**2).mean()

    # računanje gradijenata
    loss.backward()

    # korak optimizacije
    optimizer.step()

    grad_a = torch.sum(-2*diff*X).mean()
    grad_b = torch.sum(-2*diff).mean()

    print(f'step: {i}, loss:{loss}, Y_:{Y_.data}, a:{a.item()}, b:{b.item()}, grad_a:{a.grad.item()}, grad_b:{b.grad.item()},')
    print(f'\t\t   grad_a_calc:{grad_a.item()}, grad_b_calc:{grad_b.item()}')

    # Postavljanje gradijenata na nulu
    optimizer.zero_grad()


a, b = a.detach().numpy(), b.detach().numpy()
X, Y = X.numpy(), Y.numpy()

y = lambda x: a * x + b

plt.scatter(X, Y)
xs = np.arange(0, 5, 0.1)
plt.plot(xs, y(xs), color='red')
plt.show()
