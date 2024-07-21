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
        # recompute the norm (save memory at the cost of compute)
        norm = (x - mean) * rstd
        B, T, C = x.size()
        dw = (dout * norm).sum(dim=(0, 1))
        db = dout.sum(dim=(0, 1))
        dnorm = dout * w
        dx = (
            dnorm
            - dnorm.mean(dim=-1, keepdim=True)
            - norm * (dnorm * norm).mean(dim=-1, keepdim=True)
        )
        dx *= rstd
        return dw, db, dx


B = 2  # some toy numbers here
T = 3
C = 4
x = torch.randn(B, T, C, requires_grad=True)
w = torch.randn(C, requires_grad=True)
b = torch.randn(C, requires_grad=True)
out, cache = LayerNorm.forward(x, w, b)  # B,T,C

dout = torch.randn(B, T, C)
fakeloss = (out * dout).sum()
fakeloss.backward()

dw, db, dx = LayerNorm.backward(dout, cache)

print("dx error:", (x.grad - dx).abs().max().item())
print("dw error:", (w.grad - dw).abs().max().item())
print("db error:", (b.grad - db).abs().max().item())

# for reference checking in C also
x, w, mean, rstd = cache

def write(tensor, handle):
    handle.write(tensor.detach().numpy().astype("float32").tobytes())


# Write to file
with open("ln.bin", "wb") as file:
    write(x, file)  # (B, T, C)
    write(w, file)  # (C, )
    write(b, file)  # (C, )
    write(out, file)  # (B, T, C)
    write(mean, file)  # (B, T)
    write(rstd, file)  # (B, T)
    write(dout, file)  # (B, T, C)
    write(dx, file)  # (B, T, C)
    write(dw, file)  # (C, )
    write(db, file)  # (C, )
