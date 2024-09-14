'''
Copied and modified from 
https://github.com/hustvl/Vim/blob/main/mamba-1p1p1/mamba_ssm/modules/mamba_simple.py
'''

# Copyright (c) 2023, Tri Dao, Albert Gu.

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from einops import rearrange, repeat

try:
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
except ImportError:
    causal_conv1d_fn, causal_conv1d_update = None

# try:
#     from mamba.selective_scan_interface import selective_scan_fn, mamba_inner_fn, bimamba_inner_fn, mamba_inner_fn_no_out_proj
# except ImportError:
#     selective_scan_fn, mamba_inner_fn, bimamba_inner_fn, mamba_inner_fn_no_out_proj = None, None, None, None#, None
from models.mamba.selective_scan_interface import selective_scan_fn, mamba_inner_fn, bimamba_inner_fn, mamba_inner_fn_no_out_proj

try:
    from mamba_ssm.ops.triton.selective_state_update import selective_state_update
except ImportError:
    selective_state_update = None

try:
    from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None


class Mamba(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=16,
        d_conv=4,
        expand=2,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        conv_bias=True,
        bias=False,
        use_fast_path=True,  # Fused kernel options
        layer_idx=None,
        device=None,
        dtype=None,
        bimamba_type="none",
        if_devide_out=True, # False
        init_layer_scale=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.use_fast_path = use_fast_path
        self.layer_idx = layer_idx
        self.bimamba_type = bimamba_type
        self.if_devide_out = if_devide_out

        assert bimamba_type == 'v2'

        self.init_layer_scale = init_layer_scale
        if init_layer_scale is not None:
            self.a_gamma = nn.Parameter(init_layer_scale * torch.ones((d_model)), requires_grad=True)
            self.v_gammagamma = nn.Parameter(init_layer_scale * torch.ones((d_model)), requires_grad=True)

        self.a_in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.v_in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)


        self.a_conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            **factory_kwargs,
        )

        self.v_conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            **factory_kwargs,
        )

        self.activation = "silu"
        self.act = nn.SiLU()

        self.a_x_proj = nn.Linear(
            self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
        )
        self.a_dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)

        self.v_x_proj = nn.Linear(
            self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
        )
        self.v_dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)


        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = self.dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.a_dt_proj.weight, dt_init_std)
            nn.init.constant_(self.v_dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.a_dt_proj.weight, -dt_init_std, dt_init_std)
            nn.init.uniform_(self.v_dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        a_dt = torch.exp(
            torch.rand(self.d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        a_inv_dt = a_dt + torch.log(-torch.expm1(-a_dt))
        with torch.no_grad():
            self.a_dt_proj.bias.copy_(a_inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        self.a_dt_proj.bias._no_reinit = True

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        v_dt = torch.exp(
            torch.rand(self.d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        v_inv_dt = v_dt + torch.log(-torch.expm1(-v_dt))
        with torch.no_grad():
            self.v_dt_proj.bias.copy_(v_inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        self.v_dt_proj.bias._no_reinit = True

        # S4D real initialization
        # Shared A
        A = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=self.d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True

        # D "skip" parameter
        self.a_D = nn.Parameter(torch.ones(self.d_inner, device=device))  # Keep in fp32
        self.a_D._no_weight_decay = True
        self.v_D = nn.Parameter(torch.ones(self.d_inner, device=device))  # Keep in fp32
        self.v_D._no_weight_decay = True

        # bidirectional
        if bimamba_type == "v1":
            A_b = repeat(
                torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
                "n -> d n",
                d=self.d_inner,
            ).contiguous()
            A_b_log = torch.log(A_b)  # Keep A_b_log in fp32
            self.A_b_log = nn.Parameter(A_b_log)
            self.A_b_log._no_weight_decay = True
        elif bimamba_type == "v2":
            A_b = repeat(
                torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
                "n -> d n",
                d=self.d_inner,
            ).contiguous()
            A_b_log = torch.log(A_b)  # Keep A_b_log in fp32
            self.A_b_log = nn.Parameter(A_b_log)
            self.A_b_log._no_weight_decay = True 

            self.a_conv1d_b = nn.Conv1d(
                in_channels=self.d_inner,
                out_channels=self.d_inner,
                bias=conv_bias,
                kernel_size=d_conv,
                groups=self.d_inner,
                padding=d_conv - 1,
                **factory_kwargs,
            )

            self.v_conv1d_b = nn.Conv1d(
                in_channels=self.d_inner,
                out_channels=self.d_inner,
                bias=conv_bias,
                kernel_size=d_conv,
                groups=self.d_inner,
                padding=d_conv - 1,
                **factory_kwargs,
            )

            self.a_x_proj_b = nn.Linear(
                self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
            )
            self.a_dt_proj_b = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)

            self.v_x_proj_b = nn.Linear(
                self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
            )
            self.v_dt_proj_b = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)

            self.a_D_b = nn.Parameter(torch.ones(self.d_inner, device=device))  # Keep in fp32
            self.a_D_b._no_weight_decay = True
            self.v_D_b = nn.Parameter(torch.ones(self.d_inner, device=device))  # Keep in fp32
            self.v_D_b._no_weight_decay = True

        self.a_out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.v_out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)

    def forward(self, a_hidden_states, v_hidden_states, a_inference_params=None, v_inference_params=None):
        """
        hidden_states: (B, L, D)
        Returns: same shape as hidden_states
        """
        assert a_hidden_states.shape == v_hidden_states.shape
        batch, seqlen, dim = a_hidden_states.shape
        a_conv_state, a_ssm_state = None, None
        v_conv_state, v_ssm_state = None, None

        # TODO:
        if a_inference_params is not None and v_inference_params is not None:
            a_conv_state, a_ssm_state = self.a_get_states_from_cache(a_inference_params, batch)
            v_conv_state, v_ssm_state = self.v_get_states_from_cache(v_inference_params, batch)
            if a_inference_params.seqlen_offset > 0 and v_inference_params.seqlen_offset > 0:
                # The states are updated inplace
                a_out, _, _, v_out, _, _ = self.step(a_hidden_states, a_conv_state, a_ssm_state, v_hidden_states, v_conv_state, v_ssm_state)
                # v_out, _, _ = self.step(v_hidden_states, v_conv_state, v_ssm_state)
                return a_out, v_out

        # We do matmul and transpose BLH -> HBL at the same time
        ## Audio branch
        a_xz = rearrange(
            self.a_in_proj.weight @ rearrange(a_hidden_states, "b l d -> d (b l)"),
            "d (b l) -> b d l",
            l=seqlen,
        )
        if self.a_in_proj.bias is not None:
            a_xz = a_xz + rearrange(self.a_in_proj.bias.to(dtype=a_xz.dtype), "d -> d 1")
        ## Video branch
        v_xz = rearrange(
            self.v_in_proj.weight @ rearrange(v_hidden_states, "b l d -> d (b l)"),
            "d (b l) -> b d l",
            l=seqlen,
        )
        if self.v_in_proj.bias is not None:
            v_xz = v_xz + rearrange(self.v_in_proj.bias.to(dtype=v_xz.dtype), "d -> d 1")

        # Compute ∆ A B C D, the state space parameters.
        #     A, D 是独立于输入的 (see Mamba paper [1] Section 3.5.2 "Interpretation of A" for why A isn't selective)
        #     ∆, B, C 是依赖于输入的 (这是Mamba模型和 linear time invariant S4 的主要区别,这也是为什么Mamba被称为selective state spaces

        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)
        # In the backward pass we write dx and dz next to each other to avoid torch.cat
        if self.use_fast_path and a_inference_params is None and v_inference_params is None:  # Doesn't support outputting the states
            if self.bimamba_type == "v1":
                A_b = -torch.exp(self.A_b_log.float())
                a_out = bimamba_inner_fn(
                    a_xz,
                    self.a_conv1d.weight,
                    self.a_conv1d.bias,
                    self.a_x_proj.weight,
                    self.a_dt_proj.weight,
                    self.a_out_proj.weight,
                    self.a_out_proj.bias,
                    A,
                    A_b,
                    None,  # input-dependent B
                    None,  # input-dependent C
                    self.a_D.float(),
                    delta_bias=self.a_dt_proj.bias.float(),
                    delta_softplus=True,
                )    

                v_out = bimamba_inner_fn(
                    v_xz,
                    self.v_conv1d.weight,
                    self.v_conv1d.bias,
                    self.v_x_proj.weight,
                    self.v_dt_proj.weight,
                    self.v_out_proj.weight,
                    self.v_out_proj.bias,
                    A,
                    A_b,
                    None,  # input-dependent B
                    None,  # input-dependent C
                    self.v_D.float(),
                    delta_bias=self.v_dt_proj.bias.float(),
                    delta_softplus=True,
                )   
            elif self.bimamba_type == "v2":
                A_b = -torch.exp(self.A_b_log.float())
                a_out = mamba_inner_fn_no_out_proj(
                    a_xz,
                    self.a_conv1d.weight,
                    self.a_conv1d.bias,
                    self.a_x_proj.weight,
                    self.a_dt_proj.weight,
                    A,
                    None,  # input-dependent B
                    None,  # input-dependent C
                    self.a_D.float(),
                    delta_bias=self.a_dt_proj.bias.float(),
                    delta_softplus=True,
                )    

                a_out_b = mamba_inner_fn_no_out_proj(
                    a_xz.flip([-1]),
                    self.a_conv1d_b.weight,
                    self.a_conv1d_b.bias,
                    self.a_x_proj_b.weight,
                    self.a_dt_proj_b.weight,
                    A_b,
                    None,  # input-dependent B
                    None,  # input-dependent C
                    self.a_D_b.float(),
                    delta_bias=self.a_dt_proj_b.bias.float(),
                    delta_softplus=True,
                )    

                v_out = mamba_inner_fn_no_out_proj(
                    v_xz,
                    self.v_conv1d.weight,
                    self.v_conv1d.bias,
                    self.v_x_proj.weight,
                    self.v_dt_proj.weight,
                    A,
                    None,  # input-dependent B
                    None,  # input-dependent C
                    self.v_D.float(),
                    delta_bias=self.v_dt_proj.bias.float(),
                    delta_softplus=True,
                )    

                v_out_b = mamba_inner_fn_no_out_proj(
                    v_xz.flip([-1]),
                    self.v_conv1d_b.weight,
                    self.v_conv1d_b.bias,
                    self.v_x_proj_b.weight,
                    self.v_dt_proj_b.weight,
                    A_b,
                    None,  # input-dependent B
                    None,  # input-dependent C
                    self.v_D_b.float(),
                    delta_bias=self.v_dt_proj_b.bias.float(),
                    delta_softplus=True,
                )    

                if not self.if_devide_out:
                    a_out = F.linear(rearrange(a_out + a_out_b.flip([-1]), "b d l -> b l d"), self.a_out_proj.weight, self.a_out_proj.bias)
                    v_out = F.linear(rearrange(v_out + v_out_b.flip([-1]), "b d l -> b l d"), self.v_out_proj.weight, self.v_out_proj.bias)
                else:
                    a_out = F.linear(rearrange(0.5*a_out + 0.5*a_out_b.flip([-1]), "b d l -> b l d"), self.a_out_proj.weight, self.a_out_proj.bias)
                    v_out = F.linear(rearrange(0.5*v_out + 0.5*v_out_b.flip([-1]), "b d l -> b l d"), self.v_out_proj.weight, self.v_out_proj.bias)
            else:
                a_out = mamba_inner_fn(
                    a_xz,
                    self.a_conv1d.weight,
                    self.a_conv1d.bias,
                    self.a_x_proj.weight,
                    self.a_dt_proj.weight,
                    self.a_out_proj.weight,
                    self.a_out_proj.bias,
                    A,
                    None,  # input-dependent B
                    None,  # input-dependent C
                    self.a_D.float(),
                    delta_bias=self.a_dt_proj.bias.float(),
                    delta_softplus=True,
                )

                v_out = mamba_inner_fn(
                    v_xz,
                    self.v_conv1d.weight,
                    self.v_conv1d.bias,
                    self.v_x_proj.weight,
                    self.v_dt_proj.weight,
                    self.v_out_proj.weight,
                    self.v_out_proj.bias,
                    A,
                    None,  # input-dependent B
                    None,  # input-dependent C
                    self.v_D.float(),
                    delta_bias=self.v_dt_proj.bias.float(),
                    delta_softplus=True,
                )
        else:
            a_x, a_z = a_xz.chunk(2, dim=1)
            v_x, v_z = v_xz.chunk(2, dim=1)
            # Compute short convolution
            if a_conv_state is not None:
                # If we just take x[:, :, -self.d_conv :], it will error if seqlen < self.d_conv
                # Instead F.pad will pad with zeros if seqlen < self.d_conv, and truncate otherwise.
                a_conv_state.copy_(F.pad(a_x, (self.d_conv - a_x.shape[-1], 0)))  # Update state (B D W)
            if v_conv_state is not None:
                # If we just take x[:, :, -self.d_conv :], it will error if seqlen < self.d_conv
                # Instead F.pad will pad with zeros if seqlen < self.d_conv, and truncate otherwise.
                v_conv_state.copy_(F.pad(v_x, (self.d_conv - v_x.shape[-1], 0)))  # Update state (B D W)

            if causal_conv1d_fn is None:
                a_x = self.act(self.a_conv1d(a_x)[..., :seqlen])
                v_x = self.act(self.v_conv1d(v_x)[..., :seqlen])
            else:
                assert self.activation in ["silu", "swish"]
                a_x = causal_conv1d_fn(
                    x=a_x,
                    weight=rearrange(self.a_conv1d.weight, "d 1 w -> d w"),
                    bias=self.a_conv1d.bias,
                    activation=self.activation,
                )

                v_x = causal_conv1d_fn(
                    x=v_x,
                    weight=rearrange(self.v_conv1d.weight, "d 1 w -> d w"),
                    bias=self.v_conv1d.bias,
                    activation=self.activation,
                )

            # We're careful here about the layout, to avoid extra transposes.
            # We want dt to have d as the slowest moving dimension
            # and L as the fastest moving dimension, since those are what the ssm_scan kernel expects.
            a_x_dbl = self.a_x_proj(rearrange(a_x, "b d l -> (b l) d"))  # (bl d)
            a_dt, a_B, a_C = torch.split(a_x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
            a_dt = self.a_dt_proj.weight @ a_dt.t()
            a_dt = rearrange(a_dt, "d (b l) -> b d l", l=seqlen)
            a_B = rearrange(a_B, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
            a_C = rearrange(a_C, "(b l) dstate -> b dstate l", l=seqlen).contiguous()

            v_x_dbl = self.v_x_proj(rearrange(v_x, "b d l -> (b l) d"))  # (bl d)
            v_dt, v_B, v_C = torch.split(v_x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
            v_dt = self.v_dt_proj.weight @ v_dt.t()
            v_dt = rearrange(v_dt, "d (b l) -> b d l", l=seqlen)
            v_B = rearrange(v_B, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
            v_C = rearrange(v_C, "(b l) dstate -> b dstate l", l=seqlen).contiguous()


            assert self.activation in ["silu", "swish"]
            a_y = selective_scan_fn(
                a_x,
                a_dt,
                A,
                a_B,
                a_C,
                self.a_D.float(),
                z=a_z,
                delta_bias=self.a_dt_proj.bias.float(),
                delta_softplus=True,
                return_last_state=a_ssm_state is not None,
            )
            
            v_y = selective_scan_fn(
                v_x,
                v_dt,
                A,
                v_B,
                v_C,
                self.v_D.float(),
                z=v_z,
                delta_bias=self.v_dt_proj.bias.float(),
                delta_softplus=True,
                return_last_state=v_ssm_state is not None,
            )

            if a_ssm_state is not None:
                a_y, a_last_state = a_y
                a_ssm_state.copy_(a_last_state)
            a_y = rearrange(a_y, "b d l -> b l d")
            a_out = self.a_out_proj(a_y)

            if v_ssm_state is not None:
                v_y, v_last_state = v_y
                v_ssm_state.copy_(v_last_state)
            v_y = rearrange(v_y, "b d l -> b l d")
            v_out = self.v_out_proj(v_y)
        if self.init_layer_scale is not None:
            a_out = a_out * self.gamma    
            v_out = v_out * self.gamma    
        return a_out, v_out

    def step(self, a_hidden_states, a_conv_state, a_ssm_state, v_hidden_states, v_conv_state, v_ssm_state):
        dtype = a_hidden_states.dtype
        assert a_hidden_states.shape[1] == 1, "Only support decoding with 1 token at a time for now"
        a_xz = self.a_in_proj(a_hidden_states.squeeze(1))  # (B 2D)
        a_x, a_z = a_xz.chunk(2, dim=-1)  # (B D)

        v_xz = self.v_in_proj(v_hidden_states.squeeze(1))  # (B 2D)
        v_x, v_z = v_xz.chunk(2, dim=-1)  # (B D)

        # Conv step
        if causal_conv1d_update is None:
            a_conv_state.copy_(torch.roll(a_conv_state, shifts=-1, dims=-1))  # Update state (B D W)
            a_conv_state[:, :, -1] = a_x
            a_x = torch.sum(a_conv_state * rearrange(self.a_conv1d.weight, "d 1 w -> d w"), dim=-1)  # (B D)
            if self.a_conv1d.bias is not None:
                a_x = a_x + self.conv1d.bias
            a_x = self.act(a_x).to(dtype=dtype)

            v_conv_state.copy_(torch.roll(v_conv_state, shifts=-1, dims=-1))  # Update state (B D W)
            v_conv_state[:, :, -1] = v_x
            v_x = torch.sum(v_conv_state * rearrange(self.v_conv1d.weight, "d 1 w -> d w"), dim=-1)  # (B D)
            if self.v_conv1d.bias is not None:
                v_x = v_x + self.conv1d.bias
            v_x = self.act(v_x).to(dtype=dtype)
        else:
            a_x = causal_conv1d_update(
                a_x,
                a_conv_state,
                rearrange(self.a_conv1d.weight, "d 1 w -> d w"),
                self.a_conv1d.bias,
                self.activation,
            )

            v_x = causal_conv1d_update(
                v_x,
                v_conv_state,
                rearrange(self.v_conv1d.weight, "d 1 w -> d w"),
                self.v_conv1d.bias,
                self.activation,
            )

        a_x_db = self.a_x_proj(a_x)  # (B dt_rank+2*d_state)
        a_dt, a_B, a_C = torch.split(a_x_db, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        # Don't add dt_bias here
        a_dt = F.linear(a_dt, self.a_dt_proj.weight)  # (B d_inner)

        v_x_db = self.v_x_proj(v_x)  # (B dt_rank+2*d_state)
        v_dt, v_B, v_C = torch.split(v_x_db, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        # Don't add dt_bias here
        v_dt = F.linear(v_dt, self.v_dt_proj.weight)  # (B d_inner)

        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)

        # SSM step
        if selective_state_update is None:
            # Discretize A and B
            a_dt = F.softplus(a_dt + self.a_dt_proj.bias.to(dtype=a_dt.dtype))
            a_dA = torch.exp(torch.einsum("bd,dn->bdn", a_dt, A))
            a_dB = torch.einsum("bd,bn->bdn", a_dt, a_B)
            a_ssm_state.copy_(a_ssm_state * a_dA + rearrange(a_x, "b d -> b d 1") * a_dB)
            a_y = torch.einsum("bdn,bn->bd", a_ssm_state.to(dtype), a_C)
            a_y = a_y + self.a_D.to(dtype) * a_x
            a_y = a_y * self.act(a_z)  # (B D)

            v_dt = F.softplus(v_dt + self.v_dt_proj.bias.to(dtype=v_dt.dtype))
            v_dA = torch.exp(torch.einsum("bd,dn->bdn", v_dt, A))
            v_dB = torch.einsum("bd,bn->bdn", v_dt, v_B)
            v_ssm_state.copy_(v_ssm_state * v_dA + rearrange(v_x, "b d -> b d 1") * v_dB)
            v_y = torch.einsum("bdn,bn->bd", v_ssm_state.to(dtype), v_C)
            v_y = v_y + self.v_D.to(dtype) * v_x
            v_y = v_y * self.act(v_z)  # (B D)
        else:
            a_y = selective_state_update(
                a_ssm_state, a_x, a_dt, A, a_B, a_C, self.a_D, z=a_z, dt_bias=self.a_dt_proj.bias, dt_softplus=True
            )

            v_y = selective_state_update(
                v_ssm_state, v_x, v_dt, A, v_B, v_C, self.v_D, z=v_z, dt_bias=self.v_dt_proj.bias, dt_softplus=True
            )

        a_out = self.a_out_proj(a_y)
        v_out = self.v_out_proj(v_y)
        return a_out.unsqueeze(1), a_conv_state, a_ssm_state, v_out.unsqueeze(1), v_conv_state, v_ssm_state

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        device = self.a_out_proj.weight.device
        conv_dtype = self.a_conv1d.weight.dtype if dtype is None else dtype
        ssm_dtype = self.a_dt_proj.weight.dtype if dtype is None else dtype

        a_conv_state = torch.zeros(
            batch_size, self.d_model * self.expand, self.d_conv, device=device, dtype=conv_dtype
        )
        # ssm_dtype = torch.float32
        a_ssm_state = torch.zeros(
            batch_size, self.d_model * self.expand, self.d_state, device=device, dtype=ssm_dtype
        )
        return a_conv_state, a_ssm_state

    def a_get_states_from_cache(self, inference_params, batch_size, initialize_states=False):
        assert self.layer_idx is not None
        if self.layer_idx not in inference_params.key_value_memory_dict:
            batch_shape = (batch_size,)
            conv_state = torch.zeros(
                batch_size,
                self.d_model * self.expand,
                self.d_conv,
                device=self.a_conv1d.weight.device,
                dtype=self.a_conv1d.weight.dtype,
            )
            ssm_state = torch.zeros(
                batch_size,
                self.d_model * self.expand,
                self.d_state,
                device=self.a_dt_proj.weight.device,
                dtype=self.a_dt_proj.weight.dtype,
                # dtype=torch.float32,
            )
            inference_params.key_value_memory_dict[self.layer_idx] = (conv_state, ssm_state)
        else:
            conv_state, ssm_state = inference_params.key_value_memory_dict[self.layer_idx]
            # TODO: What if batch size changes between generation, and we reuse the same states?
            if initialize_states:
                conv_state.zero_()
                ssm_state.zero_()
        return conv_state, ssm_state

    def v_get_states_from_cache(self, inference_params, batch_size, initialize_states=False):
        assert self.layer_idx is not None
        if self.layer_idx not in inference_params.key_value_memory_dict:
            batch_shape = (batch_size,)
            conv_state = torch.zeros(
                batch_size,
                self.d_model * self.expand,
                self.d_conv,
                device=self.v_conv1d.weight.device,
                dtype=self.v_conv1d.weight.dtype,
            )
            ssm_state = torch.zeros(
                batch_size,
                self.d_model * self.expand,
                self.d_state,
                device=self.v_dt_proj.weight.device,
                dtype=self.v_dt_proj.weight.dtype,
                # dtype=torch.float32,
            )
            inference_params.key_value_memory_dict[self.layer_idx] = (conv_state, ssm_state)
        else:
            conv_state, ssm_state = inference_params.key_value_memory_dict[self.layer_idx]
            # TODO: What if batch size changes between generation, and we reuse the same states?
            if initialize_states:
                conv_state.zero_()
                ssm_state.zero_()
        return conv_state, ssm_state

class Block(nn.Module):
    def __init__(
        self, dim, mixer_cls, norm_cls=nn.LayerNorm, fused_add_norm=False, residual_in_fp32=False
    ):
        """
        Simple block wrapping a mixer class with LayerNorm/RMSNorm and residual connection"

        This Block has a slightly different structure compared to a regular
        prenorm Transformer block.
        The standard block is: LN -> MHA/MLP -> Add.
        [Ref: https://arxiv.org/abs/2002.04745]
        Here we have: Add -> LN -> Mixer, returning both
        the hidden_states (output of the mixer) and the residual.
        This is purely for performance reasons, as we can fuse add and LayerNorm.
        The residual needs to be provided (except for the very first block).
        """
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.mixer = mixer_cls(dim)
        self.norm = norm_cls(dim)
        if self.fused_add_norm:
            assert RMSNorm is not None, "RMSNorm import fails"
            assert isinstance(
                self.norm, (nn.LayerNorm, RMSNorm)
            ), "Only LayerNorm and RMSNorm are supported for fused_add_norm"

    def forward(
        self, hidden_states: Tensor, residual: Optional[Tensor] = None, inference_params=None
    ):
        r"""Pass the input through the encoder layer.

        Args:
            hidden_states: the sequence to the encoder layer (required).
            residual: hidden_states = Mixer(LN(residual))
        """
        if not self.fused_add_norm:
            residual = (hidden_states + residual) if residual is not None else hidden_states
            hidden_states = self.norm(residual.to(dtype=self.norm.weight.dtype))
            if self.residual_in_fp32:
                residual = residual.to(torch.float32)
        else:
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm, RMSNorm) else layer_norm_fn
            hidden_states, residual = fused_add_norm_fn(
                hidden_states,
                self.norm.weight,
                self.norm.bias,
                residual=residual,
                prenorm=True,
                residual_in_fp32=self.residual_in_fp32,
                eps=self.norm.eps,
            )
        hidden_states = self.mixer(hidden_states, inference_params=inference_params)
        return hidden_states, residual

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.mixer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)
