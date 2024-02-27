# import necessary library
import torch

# create a 3x4 2D tensor
T = torch.Tensor([[2,4,7,-6], [7,33,-62,23],[2,-6,-77,54]])
print("T:\n", T)

# compute the mean and standard deviation
mean = torch.mean(T)
std = torch.std(T)
print("Mean:", mean)
print("Standard deviation:", std)

# Compute column-wise mean and std
mean = torch.mean(T, axis = 0)
std = torch.std(T, axis = 0)
print("Column-wise Mean:\n", mean)
print("Column-wise Standard deviation:\n", std)

# Compute row-wise mean and std
#mean = torch.mean(T, axis = 1)
#std = torch.std(T, axis = 1)
#print("Row-wise Mean:\n", mean)
#print("Row-wise Standard deviation:\n", std)
print((T-mean)/std)


