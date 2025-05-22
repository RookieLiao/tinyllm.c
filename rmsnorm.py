import torch

eps = 1e-5

class RMSNorm:
    @staticmethod
    def forward(x, w):
        B, T, C = x.size()
        ms = torch.sum(x**2, dim=-1, keepdim=True) / C + eps
        rstd = 1 / torch.sqrt(ms)
        out = x * rstd * w
        return out


B = 2
T = 4
C = 8
x = torch.randn(B,T,C)
w = torch.randn(C)
out = RMSNorm.forward(x, w)


def write(tensor, handle): handle.write(tensor.detach().numpy().astype("float32").tobytes())

with open("rmsnorm.bin", "wb") as file:
    write(x, file)
    write(w, file)
    write(out, file)
