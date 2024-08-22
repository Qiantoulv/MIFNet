import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat
from einops.layers.torch import Rearrange, Reduce

try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
except:
    pass

class Conv(nn.Module):
    def __init__(self, conv, activation=None, bn=None):
        nn.Module.__init__(self)
        self.conv = conv
        self.activation = activation
        if bn:
            self.conv.bias = None
        self.bn = bn

    def forward(self, x):
        x = self.conv(x)
        if self.bn:
            x = self.bn(x)
        if self.activation:
            x = self.activation(x)
        return x
        
class RMSNorm(nn.Module):
    def __init__(self, num_features, eps=1e-5):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(1, num_features))
        self.eps = eps

    def forward(self, x):
        mean_square = x.pow(2).mean(-1, keepdim=True)
        scale = self.scale / torch.sqrt(mean_square + self.eps)
        return scale * x

class SS1D(nn.Module):
    def __init__(self,
                 d_model,
                 d_state='auto',
                 d_conv=3,
                 expand=1,
                 dt_rank='auto',
                 dt_min=0.001,
                 dt_max=0.1,
                 dt_init='random',
                 dt_scale=1.0,
                 dt_init_floor=1e-4,
                 dropout=0.,
                 conv_bias=True,
                 bias=False,
                 device=None,
                 dtype=None,
                 **kwargs):
        factory_kwargs={'device':device,'dtype':dtype}
        super().__init__()
        self.d_model=d_model
        # self.d_state=d_state
        self.d_state=math.ceil(self.d_model/6) if d_state=='auto' else d_model
        self.d_conv=d_conv
        self.expand=expand
        self.d_inner=int(self.expand*self.d_model)
        self.dt_rank=math.ceil(self.d_model/16) if dt_rank=='auto' else dt_rank
        self.in_proj=nn.Linear(self.d_model,self.d_inner*2,bias=bias,**factory_kwargs)
        self.conv1d=nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv-1)//2, #same填充
            **factory_kwargs,
        )
        self.tconv=Conv(nn.Conv1d(64, 64, kernel_size=63,stride=1,groups=64, padding=63 //2,bias=False),bn=nn.BatchNorm1d(64), activation=None)
        self.activation = nn.SiLU()
        self.x_proj=(
            nn.Linear(self.d_inner,(self.dt_rank+self.d_state*2),bias=False,**factory_kwargs),
            )
        self.x_proj_weight=nn.Parameter(torch.stack([t.weight for t in self.x_proj],dim=0))
        del self.x_proj

        self.dt_projs=(
            self.dt_init(self.dt_rank,self.d_inner,dt_scale,dt_init,dt_min,dt_max,dt_init_floor,
                         **factory_kwargs),
        )
        self.dt_projs_weight=nn.Parameter(torch.stack([t.weight for t in self.dt_projs],dim=0)) #(k=1 inner rank )
        self.dt_projs_bias=nn.Parameter(torch.stack([t.bias for t in self.dt_projs],dim=0))  # (K=1, inner)
        del self.dt_projs

        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=1, merge=True)  # (K,D,N)
        self.Ds=self.D_init(self.d_inner,copies=1,merge=True) # (K=1 D N)
        self.forward_core=self.forward_corev0
        self.out_norm=nn.LayerNorm(self.d_inner)
        self.out_RMSnorm=RMSNorm(self.d_inner)
        self.out_proj=nn.Linear(self.d_inner,self.d_model,bias=bias,**factory_kwargs)
        self.dropout=nn.Dropout(dropout) if dropout>0. else None
    @staticmethod
    def dt_init(dt_rank,d_inner,dt_scale=1.0,dt_init='random',dt_min=0.001,dt_max=0.1,dt_init_floor=1e-4,
                **factory_kwargs):
        dt_proj=nn.Linear(dt_rank,d_inner,bias=True,**factory_kwargs)
        dt_init_std=dt_rank ** -0.5 * dt_scale
        if dt_init=='constant':
            nn.init.constant_(dt_proj.weight,dt_init_std)
        elif dt_init=='random':
            nn.init.uniform_(dt_proj.weight,-dt_init_std,dt_init_std)
        else:
            raise NotImplementedError

        dt=torch.exp(
            torch.rand(d_inner,**factory_kwargs)*(math.log(dt_max)-math.log(dt_min))
            +math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt=dt+torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        dt_proj.bias._no_reinit=True

        return dt_proj
    @staticmethod
    def A_log_init(d_state,d_inner,copies=1,device=None,merge=True):
        A=repeat(
            torch.arange(1,d_state+1,dtype=torch.float32,device=device),
            'n->d n',
            d=d_inner,
        ).contiguous()  # d_inner * d_state
        A_log=torch.log(A)
        if copies>1:
            A_log=repeat(A_log,'d n->r d n',r=copies)
            if merge:
                A_log=A_log.flatten(0,1)
        A_log=nn.Parameter(A_log)
        A_log._no_weight_decay=True
        return A_log
    @staticmethod
    def D_init(d_inner,copies=1,device=None,merge=True):
        D=torch.ones(d_inner,device=device)
        if copies>1:
            D=repeat(D,'n1->r n1',r=copies)
            if merge:
                D=D.flatten(0,1)
        D=nn.Parameter(D)
        D._no_weight_decay=True
        return D


    def forward_corev0(self,x:torch.Tensor):
        self.selective_scan=selective_scan_fn
        B,C,L=x.shape
        K=1
        x=x.unsqueeze(dim=1) #(b,k,d,l)

        x_dbl=torch.einsum('b k d l,k c d->b k c l',x.view(B,K,-1,L),self.x_proj_weight)

        dts,Bs,Cs=torch.split(x_dbl,[self.dt_rank,self.d_state,self.d_state],dim=2)

        dts=torch.einsum('b k r l,k d r->b k d l',dts.view(B,K,-1,L),self.dt_projs_weight)

        x=x.float().view(B,-1,L) #(B,K*D,L)
        dts=dts.contiguous().float().view(B,-1,L) #(B,1*D,L)
        Bs=Bs.float().view(B,K,-1,L) #(B,1,d_state,L)
        Cs=Cs.float().view(B,K,-1,L)#(B,1,d_state,L)

        Ds=self.Ds.float().view(-1) #(k*d)
        As=-torch.exp(self.A_logs.float()).view(-1,self.d_state)  #(1*d,d_state)
        dt_projs_bias=self.dt_projs_bias.float().view(-1) #(1*d)

        out_y=self.selective_scan(
            x,dts,
            As,Bs,Cs,Ds,z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B,-1,L)


        return out_y

    def forward(self,x:torch.Tensor,**kwargs):

        B,L,D=x.shape
        z=x 
        x=x.permute(0,2,1) # B D L
        x=F.silu(self.tconv(x))
        y=self.forward_core(x)

        y=torch.transpose(y,dim0=1,dim1=2).contiguous().view(B,L,-1)

        try:
            y=self.out_RMSnorm(y)
        except:
            y=self.out_norm.to(torch.float32)(y).half()

        y=y*F.silu(z)
        out=y
        if self.dropout is not None:
            out=self.dropout(out)
        return out
