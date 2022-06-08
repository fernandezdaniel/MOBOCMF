import torch
from gpytorch.kernels import Kernel
from gpytorch.lazy import DiagLazyTensor, ZeroLazyTensor

# Code from: https://notebooks.githubusercontent.com/view/ipynb?browser=chrome&color_mode=auto&commit=48114e19c926827df95662afbb2d27050344fbba&device=unknown&enc_url=68747470733a2f2f7261772e67697468756275736572636f6e74656e742e636f6d2f626179657367726f75702f6465657062617965732d323031392f343831313465313963393236383237646639353636326166626232643237303530333434666262612f73656d696e6172732f646179342f67702f47502f67705f736f6c7574696f6e2e6970796e62&logged_in=false&nwo=bayesgroup%2Fdeepbayes-2019&path=seminars%2Fday4%2Fgp%2FGP%2Fgp_solution.ipynb&platform=android&repository_id=197838072&repository_type=Repository&version=98

class WhiteNoiseKernel(Kernel):
    def __init__(self, noise=1):
        super().__init__()
        self.noise = noise
    
    def forward(self, x1, x2, **params):
        if self.training and torch.equal(x1, x2):
            return DiagLazyTensor(torch.ones(x1.shape[0]).to(x1) * self.noise)
        elif x1.size(-2) == x2.size(-2) and torch.equal(x1, x2):
            return DiagLazyTensor(torch.ones(x1.shape[0]).to(x1) * self.noise)
        else:
            return torch.zeros(x1.shape[0], x2.shape[0]).to(x1)