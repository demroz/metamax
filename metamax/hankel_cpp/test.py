import torch
import hankel_cpp

x = torch.tensor([1.,2.,3.,4.], requires_grad = True)
y = torch.sum(hankel_cpp.dht(x,x))
y.backward()
print(y)
