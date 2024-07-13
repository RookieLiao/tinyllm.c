import torch

eps = 1e-5


class LayerNorm:
    @staticmethod
    def forward(x, w, b):
        # x is the input activations, of shape B,T,C
        # w are the weights, of shape C
        # b are the biases, of shape C
        B, T, C = x.size()
        # calculate the mean
        mean = x.sum(-1, keepdim=True) / C  # B,T,1
        # calculate the variance
        xshift = x - mean  # B,T,C
        var = (xshift**2).sum(-1, keepdim=True) / C  # B,T,1
        # calculate the inverse standard deviation: **0.5 is sqrt, **-0.5 is 1/sqrt
        rstd = (var + eps) ** -0.5  # B,T,1
        # normalize the input activations
        norm = xshift * rstd  # B,T,C
        # scale and shift the normalized activations at the end
        out = norm * w + b  # B,T,C

        # return the output and the cache, of variables needed later during the backward pass
        cache = (x, w, mean, rstd)
        return out, cache

    @staticmethod
    def backward(dout, cache):
        x, w, mean, rstd = cache
        dw = (dout * (x - mean) * rstd).sum(dim=(0, 1))
        db = dout.sum(dim=(0, 1))
        return dw, db, dx


B = 2  # some toy numbers here
T = 3
C = 4
x = torch.randn(B, T, C, requires_grad=True)
w = torch.randn(C, requires_grad=True)
b = torch.randn(C, requires_grad=True)
out, cache = LayerNorm.forward(x, w, b) # B,T,C

dout = torch.randn(B, T, C)
fakeloss = (out * dout).sum()
fakeloss.backward()

dw, db, dx = LayerNorm.backward(dout, cache)
print(((w.grad - dw)**2).mean())
print(((b.grad - db)**2).mean())
print(((x.grad - dx)**2).mean())
