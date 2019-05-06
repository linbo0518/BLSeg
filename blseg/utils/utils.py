import torch


def one_hot(tensor, num_classes):
    one_hot_tensor = torch.zeros(tensor.size(0), num_classes, tensor.size(-2),
                                 tensor.size(-1)).to(tensor.device)
    if tensor.dim() == 3:
        one_hot_tensor.scatter_(1, tensor.unsqueeze(1), 1)
    elif tensor.dim() == 4:
        one_hot_tensor.scatter_(1, tensor.long(), 1)
    else:
        raise NotImplementedError
    return one_hot_tensor
