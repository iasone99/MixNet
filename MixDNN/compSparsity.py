import torch

def compSparsity(mask):
    nonzeros=[]
    for i in range(mask.size(0)):
        tmp=torch.count_nonzero(mask[i, :, :], dim=0)
        nonzeros.append(tmp.unsqueeze(0))
    nonzeros=torch.cat(nonzeros,dim=0)
    return nonzeros

if __name__ == '__main__':
    #for testing
    a = torch.rand(2, 40, 80)
    compSparsity(a)