import torch
import torch.nn as nn
from .backbone import StructuralEncoder, SemanticEncoder
from .moe import MoEFusion
from .csrm import CSRM
from .uncertainty import UncertaintyHead, uncertainty_arbitration
from .decoder import Decoder

class CEIDR(nn.Module):

    def __init__(self, in_ch=None, base_ch=None, num_experts=None, use_region_router=True, use_source_router=True, use_spatial_gate=True, expert_mode='functional', use_experts=True, router_mode='hybrid', *, semantic_expand):
        super().__init__()
        self.struct_enc = StructuralEncoder(in_ch, base_ch)
        self.sem_enc = SemanticEncoder(in_ch, base_ch, expand=semantic_expand)
        moe_kwargs = {'use_region_router': use_region_router, 'use_source_router': use_source_router, 'use_spatial_gate': use_spatial_gate, 'expert_mode': expert_mode, 'use_experts': use_experts, 'router_mode': router_mode}
        self.moe1 = MoEFusion(base_ch, num_experts, **moe_kwargs)
        self.moe2 = MoEFusion(base_ch * 2, num_experts, **moe_kwargs)
        self.moe3 = MoEFusion(base_ch * 4, num_experts, **moe_kwargs)
        self.moe4 = MoEFusion(base_ch * 8, num_experts, **moe_kwargs)
        self.csrm1 = CSRM(base_ch, base_ch * 2)
        self.csrm2 = CSRM(base_ch * 2, base_ch * 4)
        self.csrm3 = CSRM(base_ch * 4, base_ch * 8)
        self.unc_head = UncertaintyHead(base_ch)
        self.decoder = Decoder(chs=[base_ch * 1, base_ch * 2, base_ch * 4, base_ch * 8])

    def forward(self, vis, ir):
        fsv = self.struct_enc(vis)
        fsi = self.struct_enc(ir)
        fsemv = self.sem_enc(vis)
        fsemi = self.sem_enc(ir)
        (f1, moe1) = self.moe1(fsv[0], fsi[0], fsemv[0], fsemi[0])
        (f2, moe2) = self.moe2(fsv[1], fsi[1], fsemv[1], fsemi[1])
        (f3, moe3) = self.moe3(fsv[2], fsi[2], fsemv[2], fsemi[2])
        (f4, moe4) = self.moe4(fsv[3], fsi[3], fsemv[3], fsemi[3])
        moe_aux = moe1 + moe2 + moe3 + moe4
        f3 = self.csrm3(f3, fsemv[3] + fsemi[3])
        f2 = self.csrm2(f2, fsemv[2] + fsemi[2])
        f1 = self.csrm1(f1, fsemv[1] + fsemi[1])
        (sigma_vis, sigma_ir) = self.unc_head(f1)
        f1 = uncertainty_arbitration(f1, f1, sigma_vis, sigma_ir)
        out = self.decoder(f4, f3, f2, f1)
        unc_aux = sigma_vis.mean() + sigma_ir.mean()
        return (out, moe_aux, unc_aux)
