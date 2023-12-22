import torch
import torch.nn as nn
import torch.nn.functional as F

def dprint(*args, **kwargs):
    import os
    if 'DEBUG' in os.environ:
        print(*args, **kwargs)

_dump_i = 0

class SRShadowForFlops(nn.Module):
    def __init__(self, in_dim, in_points, n_groups, query_dim=None,
                 out_dim=None, out_points=None, **kwargs):
        super(SRShadowForFlops, self).__init__()
        out_dim = out_dim if out_dim is not None else in_dim
        out_points = out_points if out_points is not None else in_points
        query_dim = query_dim if query_dim is not None else in_dim

        self.query_dim = query_dim
        self.in_dim = in_dim
        self.in_points = in_points
        self.n_groups = n_groups
        self.out_dim = out_dim
        self.out_points = out_points

    def forward(self, x, query):
        pass

    @staticmethod
    def __user_flops_handle__(module, input, output):
        B, num_query, num_group, num_point, num_channel = input[0].shape

        eff_in_dim = module.in_dim//num_group
        eff_out_dim = module.out_dim//num_group
        in_points = module.in_points
        out_points = module.out_points

        step1 = B*num_query*num_group*in_points*eff_in_dim*eff_out_dim
        step2 = B*num_query*num_group*eff_out_dim*in_points*out_points
        module.__flops__ += int(step1+step2)
        pass


class SpatialReasoning(nn.Module):
    def __init__(self, in_dim, in_points, n_groups, query_dim=None,
                 out_dim=None, out_points=None, sampling_rate=None):
        super(SpatialReasoning, self).__init__()
        out_dim = out_dim if out_dim is not None else in_dim
        out_points = out_points if out_points is not None else in_points
        query_dim = query_dim if query_dim is not None else in_dim
        sampling_rate = sampling_rate if sampling_rate is not None else 1

        self.query_dim = query_dim
        self.in_dim = in_dim
        self.in_points = in_points//sampling_rate
        self.n_groups = n_groups
        # out_dim = 512
        self.out_dim = out_dim
        # out_points = 64
        self.out_points = out_points

        self.eff_in_dim = in_dim//n_groups
        self.eff_out_dim = out_dim//n_groups

        self.pad_bias_dim = 0
        self.pad_bias_points = 0

        self.eff_in_dim = self.eff_in_dim + self.pad_bias_dim
        self.in_points = self.in_points + self.pad_bias_points

        self.REDUCTION = 1
        # self.linear_probe = nn.Linear(self.in_points*self.eff_in_dim*self.n_groups, self.query_dim)
        # self.linear_norm = nn.LayerNorm(self.query_dim)

        self.m_parameters = (
            self.eff_in_dim * self.eff_out_dim)
        self.s_parameters = (
            self.in_points * self.out_points)
        # self.r_parameters = (
        #     self.eff_in_dim * self.eff_out_dim)
            
        self.total_parameters = self.m_parameters + self.s_parameters

        self.parameter_generator = nn.Sequential(
            nn.Linear(self.query_dim, self.n_groups*self.total_parameters),
        )



        self.out_proj = nn.Linear(
            self.eff_out_dim*self.out_points*self.n_groups, self.query_dim, bias=True
        )

        # self.out1d_proj = nn.Linear(
        #     self.eff_in_dim*self.in_points*self.n_groups, self.query_dim,
        #     bias=True
        # )
        

        self.act = nn.ReLU(inplace=True)
        # self.act = nn.GELU()

        local_dict = locals()
        local_dict.pop('self')
        self.shadow = SRShadowForFlops(**local_dict)
        
        self.gamma = nn.Parameter(torch.zeros(1,))

        self._init_weights()

    @torch.no_grad()
    def _init_weights(self):
        print('~~~')
        nn.init.zeros_(self.parameter_generator[-1].weight)
        M, S = self.parameter_generator[-1].bias.view(self.n_groups, -1).split([self.m_parameters, self.s_parameters], dim=-1)
        # nn.init.uniform_(M, 0.0, (2.0/64)**0.5)
        # nn.init.uniform_(S, 0.0, (1.0/64)**0.5)
        # nn.init.trunc_normal_(self.parameter_generator[-1].bias, 0.0, 0.01)
            
            

    def forward(self, x, query):
        '''
        '''
        self.shadow(x, query)
        B, N, g, P, C = x.size()
        # batch, num_query, group, point, channel
        G = self.n_groups
        assert g == G
        # assert C*g == self.in_dim
        
        # query: B, N, C
        # x: B, N, G, Px, Cx

        global _dump_i

        # out = x.reshape(B, N, -1)
        # query = query + self.linear_probe(out)
        # query = self.linear_norm(query)
        

        params = self.parameter_generator(query)
        params = params.reshape(B*N, G, -1)

        # params = self.parameter_generator[-1].bias
        # params = params.view(1, G, -1).repeat(B*N, 1, 1)

        # print(params.shape, self.m_parameters, self.s_parameters)


        # params = self.parameter_prelude(query)
        # params = params.reshape(B*N, G, -1)
        # pg_w = self.parameter_generator.weight.reshape(G, params.size(2), -1)
        # pg_b = self.parameter_bias
        # params = torch.einsum('xyz,yzj->xyj', params, pg_w)
        # params = params.reshape(B*N, G, -1) + pg_b.reshape(1, G, -1)
        out = x.reshape(B*N, G, P, C)
        # torch.save(out, 'dump/dump_{}.pth'.format(_dump_i))
        # _dump_i += 1
        M, S = params.split(
            [self.m_parameters, self.s_parameters], 2)

        M = M.reshape(
            B*N, G, self.eff_in_dim, self.eff_in_dim)
        S = S.reshape(
            B*N, G, self.out_points, self.in_points)
        
        # Ra = self.r_linear.weight.reshape(
        #     1, self.n_groups, -1, self.out_points)
        
        # Ma = self.m_linear.weight.reshape(
        #     1, self.n_groups, self.eff_out_dim, -1)
        # out = F.layer_norm(out, [out.size(-2), out.size(-1)])
        # out = self.act(out)

        ''' M process'''
        # print(out.shape, M.shape)
        # M = M.abs().pow(2.0) * M.sign()
        # out = torch.matmul(S, out)
        out = torch.matmul(out, M)
        dprint(out.std())
        out = F.layer_norm(out, [out.size(-2), out.size(-1)])
        out = self.act(out)
        dprint(out.shape)
        
        # out = torch.matmul(out, M)
        out = torch.matmul(S, out)
        dprint(out.std())
        out = F.layer_norm(out, [out.size(-2), out.size(-1)])
        out = self.act(out)
        dprint(out.shape)

        # out = torch.matmul(out, R)
        # dprint(out.std())
        # out = F.layer_norm(out, [out.size(-2), out.size(-1)])
        # out = self.act(out)
        # dprint(out.shape)


        out = out.reshape(B, N, -1)
        out = self.out_proj(out)

        # out1d = x.reshape(B, N, -1)
        # out1d = self.out1d_proj(out1d)
        # out_w = self.out_proj.weight.reshape(G, -1, self.query_dim//G)
        # out = torch.einsum('xyz,yzj->xyj', out, out_w)
        # out = out.reshape(B, N, -1)
        # out = self.out_proj_mix(out)
        # dprint(out.std)

        out = query + out # + out1d

        # out_M = self.back_generator(out)
        # C = query.size(2)
        # out_M = out_M.reshape(B*N, G, C//G, C//G)
        # query_M = query.reshape(B*N, G, 1, C//G)
        # query_M = torch.matmul(query_M, out_M).view(B, N, -1)
        # out = query + self.back_linear(query_M)

        return out

    
def normalize(x, dim=0):
    # mean = x.mean(dim=dim, keepdim=True)
    mean = 0
    std = x.var(dim=dim, unbiased=False, keepdim=True).sqrt()+1e-7
    # print(mean)
    # print(std)
    # assert False
    return (x-mean)/std