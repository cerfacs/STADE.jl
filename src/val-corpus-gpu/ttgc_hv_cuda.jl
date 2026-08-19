import Pkg
haskey(Pkg.project().dependencies, "CUDA") || Pkg.add("CUDA")
using CUDA
using LinearAlgebra
CUDA.allowscalar(false)

function cuda_kernel_ttgc_hv_1!(aeresk_stack, aeresk_stack_d, aerex_stack, aerex_stack_d, aerey_stack, aerey_stack_d, aerez_stack, aerez_stack_d, beta, betad, c, cavgx_stack, cavgx_stack_d, cavgy_stack, cavgy_stack_d, cavgz_stack, cavgz_stack_d, cd, cell_vol, cell_vold, dt, dtd, factor_stack, factor_stack_d, gamma, gammad, i_cell_to_node, i_ncell, re_stack, re_stack_d, res, res2, res2d, resd, skx, skxd, sky, skyd, skz, skzd, u, ud, vere_stack, vere_stack_d)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(i_ncell - 1, 1) + 1
        return nothing
    end
    i_cell = 1 + (__tid - 1)
    cavgx_stack_d[(i_cell - 1) + 1] = cavgxd
    cavgx_stack[(i_cell - 1) + 1] = cavgx
    cavgxd = 0.0
    cavgx = 0.0
    cavgy_stack_d[(i_cell - 1) + 1] = cavgyd
    cavgy_stack[(i_cell - 1) + 1] = cavgy
    cavgyd = 0.0
    cavgy = 0.0
    cavgz_stack_d[(i_cell - 1) + 1] = cavgzd
    cavgz_stack[(i_cell - 1) + 1] = cavgz
    cavgzd = 0.0
    cavgz = 0.0
    for i_loc = 1:4
        i_node = i_cell_to_node[i_loc, i_cell]
        cavgx_stack_d[(div(i_ncell - 1, 1) + 1) + (((i_cell - 1) * (div(4 - 1, 1) + 1) + (i_loc - 1)) + 1)] = cavgxd
        cavgx_stack[(div(i_ncell - 1, 1) + 1) + (((i_cell - 1) * (div(4 - 1, 1) + 1) + (i_loc - 1)) + 1)] = cavgx
        cavgxd = cavgxd + 0.25 * cd[1, i_node]
        cavgx = cavgx + c[1, i_node] / 4
        cavgy_stack_d[(div(i_ncell - 1, 1) + 1) + (((i_cell - 1) * (div(4 - 1, 1) + 1) + (i_loc - 1)) + 1)] = cavgyd
        cavgy_stack[(div(i_ncell - 1, 1) + 1) + (((i_cell - 1) * (div(4 - 1, 1) + 1) + (i_loc - 1)) + 1)] = cavgy
        cavgyd = cavgyd + 0.25 * cd[2, i_node]
        cavgy = cavgy + c[2, i_node] / 4
        cavgz_stack_d[(div(i_ncell - 1, 1) + 1) + (((i_cell - 1) * (div(4 - 1, 1) + 1) + (i_loc - 1)) + 1)] = cavgzd
        cavgz_stack[(div(i_ncell - 1, 1) + 1) + (((i_cell - 1) * (div(4 - 1, 1) + 1) + (i_loc - 1)) + 1)] = cavgz
        cavgzd = cavgzd + 0.25 * cd[3, i_node]
        cavgz = cavgz + c[3, i_node] / 4
    end
    for i_loc = 1:4
        i_node = i_cell_to_node[i_loc, i_cell]
        vere_stack_d[((i_cell - 1) * (div(4 - 1, 1) + 1) + (i_loc - 1)) + 1] = vered
        vere_stack[((i_cell - 1) * (div(4 - 1, 1) + 1) + (i_loc - 1)) + 1] = vere
        vered = (-(1.0 / 3.0) * (c[1, i_node] * skx[i_loc, i_cell] + c[2, i_node] * sky[i_loc, i_cell] + c[3, i_node] * skz[i_loc, i_cell])) * ud[i_node] + (-(1.0 / 3.0) * u[i_node]) * (((skx[i_loc, i_cell] * cd[1, i_node] + c[1, i_node] * skxd[i_loc, i_cell]) + (sky[i_loc, i_cell] * cd[2, i_node] + c[2, i_node] * skyd[i_loc, i_cell])) + (skz[i_loc, i_cell] * cd[3, i_node] + c[3, i_node] * skzd[i_loc, i_cell]))
        vere = -(1.0 / 3.0) * u[i_node] * (c[1, i_node] * skx[i_loc, i_cell] + c[2, i_node] * sky[i_loc, i_cell] + c[3, i_node] * skz[i_loc, i_cell])
        re_stack_d[((i_cell - 1) * (div(4 - 1, 1) + 1) + (i_loc - 1)) + 1] = red
        re_stack[((i_cell - 1) * (div(4 - 1, 1) + 1) + (i_loc - 1)) + 1] = re
        red = (1.0 / cell_vol[i_cell]) * vered + -(vere / cell_vol[i_cell] ^ 2) * cell_vold[i_cell]
        re = vere / cell_vol[i_cell]
        aerex_stack_d[((i_cell - 1) * (div(4 - 1, 1) + 1) + (i_loc - 1)) + 1] = aerexd
        aerex_stack[((i_cell - 1) * (div(4 - 1, 1) + 1) + (i_loc - 1)) + 1] = aerex
        aerexd = re * cavgxd + cavgx * red
        aerex = cavgx * re
        aerey_stack_d[((i_cell - 1) * (div(4 - 1, 1) + 1) + (i_loc - 1)) + 1] = aereyd
        aerey_stack[((i_cell - 1) * (div(4 - 1, 1) + 1) + (i_loc - 1)) + 1] = aerey
        aereyd = re * cavgyd + cavgy * red
        aerey = cavgy * re
        aerez_stack_d[((i_cell - 1) * (div(4 - 1, 1) + 1) + (i_loc - 1)) + 1] = aerezd
        aerez_stack[((i_cell - 1) * (div(4 - 1, 1) + 1) + (i_loc - 1)) + 1] = aerez
        aerezd = re * cavgzd + cavgz * red
        aerez = cavgz * re
        resid = (((0.5 - gamma[i_cell]) * vere) * -(0.25dtd) + (-(dt / 4) * vere) * -(gammad[i_cell])) + (-(dt / 4) * (0.5 - gamma[i_cell])) * vered
        resi = -(dt / 4) * (0.5 - gamma[i_cell]) * vere
        for i_k = 1:4
            aeresk_stack_d[((i_cell - 1) * ((div(4 - 1, 1) + 1) * (div(4 - 1, 1) + 1)) + (i_loc - 1) * (div(4 - 1, 1) + 1) + (i_k - 1)) + 1] = aereskd
            aeresk_stack[((i_cell - 1) * ((div(4 - 1, 1) + 1) * (div(4 - 1, 1) + 1)) + (i_loc - 1) * (div(4 - 1, 1) + 1) + (i_k - 1)) + 1] = aeresk
            aereskd = ((skx[i_k, i_cell] * aerexd + aerex * skxd[i_k, i_cell]) + (sky[i_k, i_cell] * aereyd + aerey * skyd[i_k, i_cell])) + (skz[i_k, i_cell] * aerezd + aerez * skzd[i_k, i_cell])
            aeresk = aerex * skx[i_k, i_cell] + aerey * sky[i_k, i_cell] + aerez * skz[i_k, i_cell]
            factor_stack_d[((i_cell - 1) * ((div(4 - 1, 1) + 1) * (div(4 - 1, 1) + 1)) + (i_loc - 1) * (div(4 - 1, 1) + 1) + (i_k - 1)) + 1] = factord
            factor_stack[((i_cell - 1) * ((div(4 - 1, 1) + 1) * (div(4 - 1, 1) + 1)) + (i_loc - 1) * (div(4 - 1, 1) + 1) + (i_k - 1)) + 1] = factor
            factord = aeresk * (0.3333333333333333 * ((2dt) * dtd)) + (dt ^ 2 / 3) * aereskd
            factor = (dt ^ 2 / 3) * aeresk
            auxresd = resid + (beta[i_cell] * factord + factor * betad[i_cell])
            auxres = resi + factor * beta[i_cell]
            auxres2d = gamma[i_cell] * factord + factor * gammad[i_cell]
            auxres2 = factor * gamma[i_cell]
            i_k_node = i_cell_to_node[i_k, i_cell]
            CUDA.@atomic resd[i_k_node] += auxresd
            CUDA.@atomic res[i_k_node] += auxres
            CUDA.@atomic res2d[i_k_node] += auxres2d
            CUDA.@atomic res2[i_k_node] += auxres2
        end
        aeresk_stack_d[(div(i_ncell - 1, 1) + 1) * (div(4 - 1, 1) + 1) * (div(4 - 1, 1) + 1) + (((i_cell - 1) * (div(4 - 1, 1) + 1) + (i_loc - 1)) + 1)] = aereskd
        aeresk_stack[(div(i_ncell - 1, 1) + 1) * (div(4 - 1, 1) + 1) * (div(4 - 1, 1) + 1) + (((i_cell - 1) * (div(4 - 1, 1) + 1) + (i_loc - 1)) + 1)] = aeresk
        factor_stack_d[(div(i_ncell - 1, 1) + 1) * (div(4 - 1, 1) + 1) * (div(4 - 1, 1) + 1) + (((i_cell - 1) * (div(4 - 1, 1) + 1) + (i_loc - 1)) + 1)] = factord
        factor_stack[(div(i_ncell - 1, 1) + 1) * (div(4 - 1, 1) + 1) * (div(4 - 1, 1) + 1) + (((i_cell - 1) * (div(4 - 1, 1) + 1) + (i_loc - 1)) + 1)] = factor
    end
    aeresk_stack_d[((div(i_ncell - 1, 1) + 1) * (div(4 - 1, 1) + 1) * (div(4 - 1, 1) + 1) + (div(i_ncell - 1, 1) + 1) * (div(4 - 1, 1) + 1)) + ((i_cell - 1) + 1)] = aereskd
    aeresk_stack[((div(i_ncell - 1, 1) + 1) * (div(4 - 1, 1) + 1) * (div(4 - 1, 1) + 1) + (div(i_ncell - 1, 1) + 1) * (div(4 - 1, 1) + 1)) + ((i_cell - 1) + 1)] = aeresk
    aerex_stack_d[(div(i_ncell - 1, 1) + 1) * (div(4 - 1, 1) + 1) + ((i_cell - 1) + 1)] = aerexd
    aerex_stack[(div(i_ncell - 1, 1) + 1) * (div(4 - 1, 1) + 1) + ((i_cell - 1) + 1)] = aerex
    aerey_stack_d[(div(i_ncell - 1, 1) + 1) * (div(4 - 1, 1) + 1) + ((i_cell - 1) + 1)] = aereyd
    aerey_stack[(div(i_ncell - 1, 1) + 1) * (div(4 - 1, 1) + 1) + ((i_cell - 1) + 1)] = aerey
    aerez_stack_d[(div(i_ncell - 1, 1) + 1) * (div(4 - 1, 1) + 1) + ((i_cell - 1) + 1)] = aerezd
    aerez_stack[(div(i_ncell - 1, 1) + 1) * (div(4 - 1, 1) + 1) + ((i_cell - 1) + 1)] = aerez
    cavgx_stack_d[((div(i_ncell - 1, 1) + 1) + (div(i_ncell - 1, 1) + 1) * (div(4 - 1, 1) + 1)) + ((i_cell - 1) + 1)] = cavgxd
    cavgx_stack[((div(i_ncell - 1, 1) + 1) + (div(i_ncell - 1, 1) + 1) * (div(4 - 1, 1) + 1)) + ((i_cell - 1) + 1)] = cavgx
    cavgy_stack_d[((div(i_ncell - 1, 1) + 1) + (div(i_ncell - 1, 1) + 1) * (div(4 - 1, 1) + 1)) + ((i_cell - 1) + 1)] = cavgyd
    cavgy_stack[((div(i_ncell - 1, 1) + 1) + (div(i_ncell - 1, 1) + 1) * (div(4 - 1, 1) + 1)) + ((i_cell - 1) + 1)] = cavgy
    cavgz_stack_d[((div(i_ncell - 1, 1) + 1) + (div(i_ncell - 1, 1) + 1) * (div(4 - 1, 1) + 1)) + ((i_cell - 1) + 1)] = cavgzd
    cavgz_stack[((div(i_ncell - 1, 1) + 1) + (div(i_ncell - 1, 1) + 1) * (div(4 - 1, 1) + 1)) + ((i_cell - 1) + 1)] = cavgz
    factor_stack_d[((div(i_ncell - 1, 1) + 1) * (div(4 - 1, 1) + 1) * (div(4 - 1, 1) + 1) + (div(i_ncell - 1, 1) + 1) * (div(4 - 1, 1) + 1)) + ((i_cell - 1) + 1)] = factord
    factor_stack[((div(i_ncell - 1, 1) + 1) * (div(4 - 1, 1) + 1) * (div(4 - 1, 1) + 1) + (div(i_ncell - 1, 1) + 1) * (div(4 - 1, 1) + 1)) + ((i_cell - 1) + 1)] = factor
    re_stack_d[(div(i_ncell - 1, 1) + 1) * (div(4 - 1, 1) + 1) + ((i_cell - 1) + 1)] = red
    re_stack[(div(i_ncell - 1, 1) + 1) * (div(4 - 1, 1) + 1) + ((i_cell - 1) + 1)] = re
    vere_stack_d[(div(i_ncell - 1, 1) + 1) * (div(4 - 1, 1) + 1) + ((i_cell - 1) + 1)] = vered
    vere_stack[(div(i_ncell - 1, 1) + 1) * (div(4 - 1, 1) + 1) + ((i_cell - 1) + 1)] = vere
    return nothing
end

function cuda_kernel_ttgc_hv_2!(i_node_perio, npernode_half, res, resd, resperio, resperiod)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(npernode_half - 1, 1) + 1
        return nothing
    end
    k = 1 + (__tid - 1)
    i1 = i_node_perio[k, 1]
    i2 = i_node_perio[k, 2]
    resperiod[k, 1] = resd[i1]
    resperio[k, 1] = res[i1]
    resperiod[k, 2] = resd[i2]
    resperio[k, 2] = res[i2]
    return nothing
end

function cuda_kernel_ttgc_hv_3!(i_node_perio, npernode_half, res, resd, resperio, resperiod)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(npernode_half - 1, 1) + 1
        return nothing
    end
    k = 1 + (__tid - 1)
    i1 = i_node_perio[k, 1]
    i2 = i_node_perio[k, 2]
    CUDA.@atomic resd[i1] += resperiod[k, 2]
    CUDA.@atomic res[i1] += resperio[k, 2]
    CUDA.@atomic resd[i2] += resperiod[k, 1]
    CUDA.@atomic res[i2] += resperio[k, 1]
    return nothing
end

function cuda_kernel_ttgc_hv_4!(i_nnode, node_vol, node_vold, res, resd, up, upd)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(i_nnode - 1, 1) + 1
        return nothing
    end
    i_node = 1 + (__tid - 1)
    upd[i_node] = (1.0 / node_vol[i_node]) * resd[i_node] + -(res[i_node] / node_vol[i_node] ^ 2) * node_vold[i_node]
    up[i_node] = res[i_node] / node_vol[i_node]
    return nothing
end

function cuda_kernel_ttgc_hv_5!(i_nnode, i_seq_, mup, mup_stack, mup_stack_d, mupd)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(i_nnode - 1, 1) + 1
        return nothing
    end
    i_node = 1 + (__tid - 1)
    mup_stack_d[((i_seq_ - 1) * (div(i_nnode - 1, 1) + 1) + (i_node - 1)) + 1] = mupd[i_node]
    mup_stack[((i_seq_ - 1) * (div(i_nnode - 1, 1) + 1) + (i_node - 1)) + 1] = mup[i_node]
    mupd[i_node] = 0.0
    mup[i_node] = 0.0
    return nothing
end

function cuda_kernel_ttgc_hv_6!(auxu_stack, auxu_stack_d, cell_vol, cell_vold, i_cell_to_node, i_ncell, i_njac, i_nnode, i_seq_, mup, mup_stack, mup_stack_d, mupd, up, upd)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(i_ncell - 1, 1) + 1
        return nothing
    end
    i_cell = 1 + (__tid - 1)
    i_node1 = i_cell_to_node[1, i_cell]
    i_node2 = i_cell_to_node[2, i_cell]
    i_node3 = i_cell_to_node[3, i_cell]
    i_node4 = i_cell_to_node[4, i_cell]
    auxu_stack_d[((i_seq_ - 1) * (div(i_ncell - 1, 1) + 1) + (i_cell - 1)) + 1] = auxud
    auxu_stack[((i_seq_ - 1) * (div(i_ncell - 1, 1) + 1) + (i_cell - 1)) + 1] = auxu
    auxud = ((upd[i_node1] + upd[i_node2]) + upd[i_node3]) + upd[i_node4]
    auxu = up[i_node1] + up[i_node2] + up[i_node3] + up[i_node4]
    mup_stack_d[(div(i_njac - 1, 1) + 1) * (div(i_nnode - 1, 1) + 1) + (((i_seq_ - 1) * (div(i_ncell - 1, 1) + 1) + (i_cell - 1)) + 1)] = mupd[i_node1]
    mup_stack[(div(i_njac - 1, 1) + 1) * (div(i_nnode - 1, 1) + 1) + (((i_seq_ - 1) * (div(i_ncell - 1, 1) + 1) + (i_cell - 1)) + 1)] = mup[i_node1]
    CUDA.@atomic mupd[i_node1] += 0.05 * (cell_vol[i_cell] * (auxud + upd[i_node1]) + (auxu + up[i_node1]) * cell_vold[i_cell])
    CUDA.@atomic mup[i_node1] += ((auxu + up[i_node1]) * cell_vol[i_cell]) / 20
    mup_stack_d[((div(i_njac - 1, 1) + 1) * (div(i_nnode - 1, 1) + 1) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (((i_seq_ - 1) * (div(i_ncell - 1, 1) + 1) + (i_cell - 1)) + 1)] = mupd[i_node2]
    mup_stack[((div(i_njac - 1, 1) + 1) * (div(i_nnode - 1, 1) + 1) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (((i_seq_ - 1) * (div(i_ncell - 1, 1) + 1) + (i_cell - 1)) + 1)] = mup[i_node2]
    CUDA.@atomic mupd[i_node2] += 0.05 * (cell_vol[i_cell] * (auxud + upd[i_node2]) + (auxu + up[i_node2]) * cell_vold[i_cell])
    CUDA.@atomic mup[i_node2] += ((auxu + up[i_node2]) * cell_vol[i_cell]) / 20
    mup_stack_d[(((div(i_njac - 1, 1) + 1) * (div(i_nnode - 1, 1) + 1) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (((i_seq_ - 1) * (div(i_ncell - 1, 1) + 1) + (i_cell - 1)) + 1)] = mupd[i_node3]
    mup_stack[(((div(i_njac - 1, 1) + 1) * (div(i_nnode - 1, 1) + 1) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (((i_seq_ - 1) * (div(i_ncell - 1, 1) + 1) + (i_cell - 1)) + 1)] = mup[i_node3]
    CUDA.@atomic mupd[i_node3] += 0.05 * (cell_vol[i_cell] * (auxud + upd[i_node3]) + (auxu + up[i_node3]) * cell_vold[i_cell])
    CUDA.@atomic mup[i_node3] += ((auxu + up[i_node3]) * cell_vol[i_cell]) / 20
    mup_stack_d[((((div(i_njac - 1, 1) + 1) * (div(i_nnode - 1, 1) + 1) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (((i_seq_ - 1) * (div(i_ncell - 1, 1) + 1) + (i_cell - 1)) + 1)] = mupd[i_node4]
    mup_stack[((((div(i_njac - 1, 1) + 1) * (div(i_nnode - 1, 1) + 1) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (((i_seq_ - 1) * (div(i_ncell - 1, 1) + 1) + (i_cell - 1)) + 1)] = mup[i_node4]
    CUDA.@atomic mupd[i_node4] += 0.05 * (cell_vol[i_cell] * (auxud + upd[i_node4]) + (auxu + up[i_node4]) * cell_vold[i_cell])
    CUDA.@atomic mup[i_node4] += ((auxu + up[i_node4]) * cell_vol[i_cell]) / 20
    return nothing
end

function cuda_kernel_ttgc_hv_7!(i_node_perio, mup, mupd, npernode_half, resperio, resperiod)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(npernode_half - 1, 1) + 1
        return nothing
    end
    k = 1 + (__tid - 1)
    i1 = i_node_perio[k, 1]
    i2 = i_node_perio[k, 2]
    resperiod[k, 1] = mupd[i1]
    resperio[k, 1] = mup[i1]
    resperiod[k, 2] = mupd[i2]
    resperio[k, 2] = mup[i2]
    return nothing
end

function cuda_kernel_ttgc_hv_8!(i_ncell, i_njac, i_nnode, i_node_perio, i_seq_, mup, mup_stack, mup_stack_d, mupd, npernode_half, resperio, resperiod)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(npernode_half - 1, 1) + 1
        return nothing
    end
    k = 1 + (__tid - 1)
    i1 = i_node_perio[k, 1]
    i2 = i_node_perio[k, 2]
    mup_stack_d[(((((div(i_njac - 1, 1) + 1) * (div(i_nnode - 1, 1) + 1) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (((i_seq_ - 1) * (div(npernode_half - 1, 1) + 1) + (k - 1)) + 1)] = mupd[i1]
    mup_stack[(((((div(i_njac - 1, 1) + 1) * (div(i_nnode - 1, 1) + 1) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (((i_seq_ - 1) * (div(npernode_half - 1, 1) + 1) + (k - 1)) + 1)] = mup[i1]
    CUDA.@atomic mupd[i1] += resperiod[k, 2]
    CUDA.@atomic mup[i1] += resperio[k, 2]
    mup_stack_d[((((((div(i_njac - 1, 1) + 1) * (div(i_nnode - 1, 1) + 1) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(npernode_half - 1, 1) + 1)) + (((i_seq_ - 1) * (div(npernode_half - 1, 1) + 1) + (k - 1)) + 1)] = mupd[i2]
    mup_stack[((((((div(i_njac - 1, 1) + 1) * (div(i_nnode - 1, 1) + 1) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(npernode_half - 1, 1) + 1)) + (((i_seq_ - 1) * (div(npernode_half - 1, 1) + 1) + (k - 1)) + 1)] = mup[i2]
    CUDA.@atomic mupd[i2] += resperiod[k, 1]
    CUDA.@atomic mup[i2] += resperio[k, 1]
    return nothing
end

function cuda_kernel_ttgc_hv_9!(i_nnode, i_seq_, mup, mupd, node_vol, node_vold, res, resd, up, up_stack, up_stack_d, upd)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(i_nnode - 1, 1) + 1
        return nothing
    end
    i_node = 1 + (__tid - 1)
    up_stack_d[((i_seq_ - 1) * (div(i_nnode - 1, 1) + 1) + (i_node - 1)) + 1] = upd[i_node]
    up_stack[((i_seq_ - 1) * (div(i_nnode - 1, 1) + 1) + (i_node - 1)) + 1] = up[i_node]
    upd[i_node] = upd[i_node] + ((1.0 / node_vol[i_node]) * (resd[i_node] + -(mupd[i_node])) + -((res[i_node] - mup[i_node]) / node_vol[i_node] ^ 2) * node_vold[i_node])
    up[i_node] = up[i_node] + (res[i_node] - mup[i_node]) / node_vol[i_node]
    return nothing
end

function cuda_kernel_ttgc_hv_10!(c, cd, dt, dtd, i_cell_to_node, i_ncell, res2, res2d, skx, skxd, sky, skyd, u, ud, up, upd, vere_stack, vere_stack_d)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(i_ncell - 1, 1) + 1
        return nothing
    end
    i_cell = 1 + (__tid - 1)
    for i_loc = 1:4
        i_node = i_cell_to_node[i_loc, i_cell]
        vere_stack_d[((div(i_ncell - 1, 1) + 1) * (div(4 - 1, 1) + 1) + (div(i_ncell - 1, 1) + 1)) + (((i_cell - 1) * (div(4 - 1, 1) + 1) + (i_loc - 1)) + 1)] = vered
        vere_stack[((div(i_ncell - 1, 1) + 1) * (div(4 - 1, 1) + 1) + (div(i_ncell - 1, 1) + 1)) + (((i_cell - 1) * (div(4 - 1, 1) + 1) + (i_loc - 1)) + 1)] = vere
        vered = (-(1.0 / 3.0) * (c[1, i_node] * skx[i_loc, i_cell] + c[2, i_node] * sky[i_loc, i_cell])) * (ud[i_node] + upd[i_node]) + (-(1.0 / 3.0) * (u[i_node] + up[i_node])) * ((skx[i_loc, i_cell] * cd[1, i_node] + c[1, i_node] * skxd[i_loc, i_cell]) + (sky[i_loc, i_cell] * cd[2, i_node] + c[2, i_node] * skyd[i_loc, i_cell]))
        vere = -(1.0 / 3.0) * (u[i_node] + up[i_node]) * (c[1, i_node] * skx[i_loc, i_cell] + c[2, i_node] * sky[i_loc, i_cell])
        resid = vere * -(0.25dtd) + -(dt / 4) * vered
        resi = -(dt / 4) * vere
        for i_k = 1:4
            i_k_node = i_cell_to_node[i_k, i_cell]
            CUDA.@atomic res2d[i_k_node] += resid
            CUDA.@atomic res2[i_k_node] += resi
        end
    end
    vere_stack_d[(((div(i_ncell - 1, 1) + 1) * (div(4 - 1, 1) + 1) + (div(i_ncell - 1, 1) + 1)) + (div(i_ncell - 1, 1) + 1) * (div(4 - 1, 1) + 1)) + ((i_cell - 1) + 1)] = vered
    vere_stack[(((div(i_ncell - 1, 1) + 1) * (div(4 - 1, 1) + 1) + (div(i_ncell - 1, 1) + 1)) + (div(i_ncell - 1, 1) + 1) * (div(4 - 1, 1) + 1)) + ((i_cell - 1) + 1)] = vere
    return nothing
end

function cuda_kernel_ttgc_hv_11!(i_node_perio, npernode_half, res2, res2d, resperio, resperiod)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(npernode_half - 1, 1) + 1
        return nothing
    end
    k = 1 + (__tid - 1)
    i1 = i_node_perio[k, 1]
    i2 = i_node_perio[k, 2]
    resperiod[k, 1] = res2d[i1]
    resperio[k, 1] = res2[i1]
    resperiod[k, 2] = res2d[i2]
    resperio[k, 2] = res2[i2]
    return nothing
end

function cuda_kernel_ttgc_hv_12!(i_node_perio, npernode_half, res2, res2d, resperio, resperiod)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(npernode_half - 1, 1) + 1
        return nothing
    end
    k = 1 + (__tid - 1)
    i1 = i_node_perio[k, 1]
    i2 = i_node_perio[k, 2]
    CUDA.@atomic res2d[i1] += resperiod[k, 2]
    CUDA.@atomic res2[i1] += resperio[k, 2]
    CUDA.@atomic res2d[i2] += resperiod[k, 1]
    CUDA.@atomic res2[i2] += resperio[k, 1]
    return nothing
end

function cuda_kernel_ttgc_hv_13!(i_njac, i_nnode, node_vol, node_vold, res2, res2d, up, up_stack, up_stack_d, upd)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(i_nnode - 1, 1) + 1
        return nothing
    end
    i_node = 1 + (__tid - 1)
    up_stack_d[(div(i_njac - 1, 1) + 1) * (div(i_nnode - 1, 1) + 1) + ((i_node - 1) + 1)] = upd[i_node]
    up_stack[(div(i_njac - 1, 1) + 1) * (div(i_nnode - 1, 1) + 1) + ((i_node - 1) + 1)] = up[i_node]
    upd[i_node] = (1.0 / node_vol[i_node]) * res2d[i_node] + -(res2[i_node] / node_vol[i_node] ^ 2) * node_vold[i_node]
    up[i_node] = res2[i_node] / node_vol[i_node]
    return nothing
end

function cuda_kernel_ttgc_hv_14!(i_ncell, i_njac, i_nnode, i_seq_, mup, mup_stack, mup_stack_d, mupd, npernode_half)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(i_nnode - 1, 1) + 1
        return nothing
    end
    i_node = 1 + (__tid - 1)
    mup_stack_d[(((((((div(i_njac - 1, 1) + 1) * (div(i_nnode - 1, 1) + 1) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(npernode_half - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(npernode_half - 1, 1) + 1)) + (((i_seq_ - 1) * (div(i_nnode - 1, 1) + 1) + (i_node - 1)) + 1)] = mupd[i_node]
    mup_stack[(((((((div(i_njac - 1, 1) + 1) * (div(i_nnode - 1, 1) + 1) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(npernode_half - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(npernode_half - 1, 1) + 1)) + (((i_seq_ - 1) * (div(i_nnode - 1, 1) + 1) + (i_node - 1)) + 1)] = mup[i_node]
    mupd[i_node] = 0.0
    mup[i_node] = 0.0
    return nothing
end

function cuda_kernel_ttgc_hv_15!(auxu_stack, auxu_stack_d, cell_vol, cell_vold, i_cell_to_node, i_ncell, i_njac, i_nnode, i_seq_, mup, mup_stack, mup_stack_d, mupd, npernode_half, up, upd)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(i_ncell - 1, 1) + 1
        return nothing
    end
    i_cell = 1 + (__tid - 1)
    i_node1 = i_cell_to_node[1, i_cell]
    i_node2 = i_cell_to_node[2, i_cell]
    i_node3 = i_cell_to_node[3, i_cell]
    i_node4 = i_cell_to_node[4, i_cell]
    auxu_stack_d[((div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1) + (div(i_njac - 1, 1) + 1)) + (((i_seq_ - 1) * (div(i_ncell - 1, 1) + 1) + (i_cell - 1)) + 1)] = auxud
    auxu_stack[((div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1) + (div(i_njac - 1, 1) + 1)) + (((i_seq_ - 1) * (div(i_ncell - 1, 1) + 1) + (i_cell - 1)) + 1)] = auxu
    auxud = ((upd[i_node1] + upd[i_node2]) + upd[i_node3]) + upd[i_node4]
    auxu = up[i_node1] + up[i_node2] + up[i_node3] + up[i_node4]
    mup_stack_d[((((((((div(i_njac - 1, 1) + 1) * (div(i_nnode - 1, 1) + 1) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(npernode_half - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(npernode_half - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_nnode - 1, 1) + 1)) + (((i_seq_ - 1) * (div(i_ncell - 1, 1) + 1) + (i_cell - 1)) + 1)] = mupd[i_node1]
    mup_stack[((((((((div(i_njac - 1, 1) + 1) * (div(i_nnode - 1, 1) + 1) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(npernode_half - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(npernode_half - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_nnode - 1, 1) + 1)) + (((i_seq_ - 1) * (div(i_ncell - 1, 1) + 1) + (i_cell - 1)) + 1)] = mup[i_node1]
    CUDA.@atomic mupd[i_node1] += 0.05 * (cell_vol[i_cell] * (auxud + upd[i_node1]) + (auxu + up[i_node1]) * cell_vold[i_cell])
    CUDA.@atomic mup[i_node1] += ((auxu + up[i_node1]) * cell_vol[i_cell]) / 20
    mup_stack_d[(((((((((div(i_njac - 1, 1) + 1) * (div(i_nnode - 1, 1) + 1) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(npernode_half - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(npernode_half - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_nnode - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (((i_seq_ - 1) * (div(i_ncell - 1, 1) + 1) + (i_cell - 1)) + 1)] = mupd[i_node2]
    mup_stack[(((((((((div(i_njac - 1, 1) + 1) * (div(i_nnode - 1, 1) + 1) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(npernode_half - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(npernode_half - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_nnode - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (((i_seq_ - 1) * (div(i_ncell - 1, 1) + 1) + (i_cell - 1)) + 1)] = mup[i_node2]
    CUDA.@atomic mupd[i_node2] += 0.05 * (cell_vol[i_cell] * (auxud + upd[i_node2]) + (auxu + up[i_node2]) * cell_vold[i_cell])
    CUDA.@atomic mup[i_node2] += ((auxu + up[i_node2]) * cell_vol[i_cell]) / 20
    mup_stack_d[((((((((((div(i_njac - 1, 1) + 1) * (div(i_nnode - 1, 1) + 1) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(npernode_half - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(npernode_half - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_nnode - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (((i_seq_ - 1) * (div(i_ncell - 1, 1) + 1) + (i_cell - 1)) + 1)] = mupd[i_node3]
    mup_stack[((((((((((div(i_njac - 1, 1) + 1) * (div(i_nnode - 1, 1) + 1) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(npernode_half - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(npernode_half - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_nnode - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (((i_seq_ - 1) * (div(i_ncell - 1, 1) + 1) + (i_cell - 1)) + 1)] = mup[i_node3]
    CUDA.@atomic mupd[i_node3] += 0.05 * (cell_vol[i_cell] * (auxud + upd[i_node3]) + (auxu + up[i_node3]) * cell_vold[i_cell])
    CUDA.@atomic mup[i_node3] += ((auxu + up[i_node3]) * cell_vol[i_cell]) / 20
    mup_stack_d[(((((((((((div(i_njac - 1, 1) + 1) * (div(i_nnode - 1, 1) + 1) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(npernode_half - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(npernode_half - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_nnode - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (((i_seq_ - 1) * (div(i_ncell - 1, 1) + 1) + (i_cell - 1)) + 1)] = mupd[i_node4]
    mup_stack[(((((((((((div(i_njac - 1, 1) + 1) * (div(i_nnode - 1, 1) + 1) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(npernode_half - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(npernode_half - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_nnode - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (((i_seq_ - 1) * (div(i_ncell - 1, 1) + 1) + (i_cell - 1)) + 1)] = mup[i_node4]
    CUDA.@atomic mupd[i_node4] += 0.05 * (cell_vol[i_cell] * (auxud + upd[i_node4]) + (auxu + up[i_node4]) * cell_vold[i_cell])
    CUDA.@atomic mup[i_node4] += ((auxu + up[i_node4]) * cell_vol[i_cell]) / 20
    return nothing
end

function cuda_kernel_ttgc_hv_16!(i_node_perio, mup, mupd, npernode_half, resperio, resperiod)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(npernode_half - 1, 1) + 1
        return nothing
    end
    k = 1 + (__tid - 1)
    i1 = i_node_perio[k, 1]
    i2 = i_node_perio[k, 2]
    resperiod[k, 1] = mupd[i1]
    resperio[k, 1] = mup[i1]
    resperiod[k, 2] = mupd[i2]
    resperio[k, 2] = mup[i2]
    return nothing
end

function cuda_kernel_ttgc_hv_17!(i_ncell, i_njac, i_nnode, i_node_perio, i_seq_, mup, mup_stack, mup_stack_d, mupd, npernode_half, resperio, resperiod)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(npernode_half - 1, 1) + 1
        return nothing
    end
    k = 1 + (__tid - 1)
    i1 = i_node_perio[k, 1]
    i2 = i_node_perio[k, 2]
    mup_stack_d[((((((((((((div(i_njac - 1, 1) + 1) * (div(i_nnode - 1, 1) + 1) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(npernode_half - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(npernode_half - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_nnode - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (((i_seq_ - 1) * (div(npernode_half - 1, 1) + 1) + (k - 1)) + 1)] = mupd[i1]
    mup_stack[((((((((((((div(i_njac - 1, 1) + 1) * (div(i_nnode - 1, 1) + 1) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(npernode_half - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(npernode_half - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_nnode - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (((i_seq_ - 1) * (div(npernode_half - 1, 1) + 1) + (k - 1)) + 1)] = mup[i1]
    CUDA.@atomic mupd[i1] += resperiod[k, 2]
    CUDA.@atomic mup[i1] += resperio[k, 2]
    mup_stack_d[(((((((((((((div(i_njac - 1, 1) + 1) * (div(i_nnode - 1, 1) + 1) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(npernode_half - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(npernode_half - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_nnode - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(npernode_half - 1, 1) + 1)) + (((i_seq_ - 1) * (div(npernode_half - 1, 1) + 1) + (k - 1)) + 1)] = mupd[i2]
    mup_stack[(((((((((((((div(i_njac - 1, 1) + 1) * (div(i_nnode - 1, 1) + 1) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(npernode_half - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(npernode_half - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_nnode - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(npernode_half - 1, 1) + 1)) + (((i_seq_ - 1) * (div(npernode_half - 1, 1) + 1) + (k - 1)) + 1)] = mup[i2]
    CUDA.@atomic mupd[i2] += resperiod[k, 1]
    CUDA.@atomic mup[i2] += resperio[k, 1]
    return nothing
end

function cuda_kernel_ttgc_hv_18!(i_njac, i_nnode, i_seq_, mup, mupd, node_vol, node_vold, res2, res2d, up, up_stack, up_stack_d, upd)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(i_nnode - 1, 1) + 1
        return nothing
    end
    i_node = 1 + (__tid - 1)
    up_stack_d[((div(i_njac - 1, 1) + 1) * (div(i_nnode - 1, 1) + 1) + (div(i_nnode - 1, 1) + 1)) + (((i_seq_ - 1) * (div(i_nnode - 1, 1) + 1) + (i_node - 1)) + 1)] = upd[i_node]
    up_stack[((div(i_njac - 1, 1) + 1) * (div(i_nnode - 1, 1) + 1) + (div(i_nnode - 1, 1) + 1)) + (((i_seq_ - 1) * (div(i_nnode - 1, 1) + 1) + (i_node - 1)) + 1)] = up[i_node]
    upd[i_node] = upd[i_node] + ((1.0 / node_vol[i_node]) * (res2d[i_node] + -(mupd[i_node])) + -((res2[i_node] - mup[i_node]) / node_vol[i_node] ^ 2) * node_vold[i_node])
    up[i_node] = up[i_node] + (res2[i_node] - mup[i_node]) / node_vol[i_node]
    return nothing
end

function cuda_kernel_ttgc_hv_19!(i_nnode, loss, lossd, u, ud, up, upd, uref, urefd)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(i_nnode - 1, 1) + 1
        return nothing
    end
    i_seq_node = 1 + (__tid - 1)
    CUDA.@atomic lossd[1] += (2 * ((u[i_seq_node] + up[i_seq_node]) - uref[i_seq_node])) * ((ud[i_seq_node] + upd[i_seq_node]) + -(urefd[i_seq_node]))
    CUDA.@atomic loss[1] += ((u[i_seq_node] + up[i_seq_node]) - uref[i_seq_node]) ^ 2
    return nothing
end

function cuda_kernel_ttgc_hv_20!(i_nnode, lossb, lossbd, u, ub, ubd, ud, up, upb, upbd, upd, uref, urefb, urefbd, urefd)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(1 - i_nnode, -1) + 1
        return nothing
    end
    i_seq_node = i_nnode + (__tid - 1) * -1
    ubd[i_seq_node] = ubd[i_seq_node] + (lossb[1] * (2 * ((ud[i_seq_node] + upd[i_seq_node]) + -(urefd[i_seq_node]))) + (2 * ((u[i_seq_node] + up[i_seq_node]) - uref[i_seq_node])) * lossbd[1])
    ub[i_seq_node] = ub[i_seq_node] + (2 * ((u[i_seq_node] + up[i_seq_node]) - uref[i_seq_node])) * lossb[1]
    upbd[i_seq_node] = upbd[i_seq_node] + (lossb[1] * (2 * ((ud[i_seq_node] + upd[i_seq_node]) + -(urefd[i_seq_node]))) + (2 * ((u[i_seq_node] + up[i_seq_node]) - uref[i_seq_node])) * lossbd[1])
    upb[i_seq_node] = upb[i_seq_node] + (2 * ((u[i_seq_node] + up[i_seq_node]) - uref[i_seq_node])) * lossb[1]
    urefbd[i_seq_node] = urefbd[i_seq_node] + -((lossb[1] * (2 * ((ud[i_seq_node] + upd[i_seq_node]) + -(urefd[i_seq_node]))) + (2 * ((u[i_seq_node] + up[i_seq_node]) - uref[i_seq_node])) * lossbd[1]))
    urefb[i_seq_node] = urefb[i_seq_node] + -((2 * ((u[i_seq_node] + up[i_seq_node]) - uref[i_seq_node])) * lossb[1])
    return nothing
end

function cuda_kernel_ttgc_hv_21!(i_njac, i_nnode, i_seq_, mup, mupb, mupbd, mupd, node_vol, node_volb, node_volbd, node_vold, res2, res2b, res2bd, res2d, up, up_stack, up_stack_d, upb, upbd, upd)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(1 - i_nnode, -1) + 1
        return nothing
    end
    i_node = i_nnode + (__tid - 1) * -1
    upd[i_node] = up_stack_d[((div(i_njac - 1, 1) + 1) * (div(i_nnode - 1, 1) + 1) + (div(i_nnode - 1, 1) + 1)) + (((i_seq_ - 1) * (div(i_nnode - 1, 1) + 1) + (i_node - 1)) + 1)]
    up[i_node] = up_stack[((div(i_njac - 1, 1) + 1) * (div(i_nnode - 1, 1) + 1) + (div(i_nnode - 1, 1) + 1)) + (((i_seq_ - 1) * (div(i_nnode - 1, 1) + 1) + (i_node - 1)) + 1)]
    res2bd[i_node] = res2bd[i_node] + (upb[i_node] * (-(1.0 / node_vol[i_node] ^ 2) * node_vold[i_node]) + (1.0 / node_vol[i_node]) * upbd[i_node])
    res2b[i_node] = res2b[i_node] + (1.0 / node_vol[i_node]) * upb[i_node]
    mupbd[i_node] = mupbd[i_node] + -((upb[i_node] * (-(1.0 / node_vol[i_node] ^ 2) * node_vold[i_node]) + (1.0 / node_vol[i_node]) * upbd[i_node]))
    mupb[i_node] = mupb[i_node] + -((1.0 / node_vol[i_node]) * upb[i_node])
    node_volbd[i_node] = node_volbd[i_node] + (upb[i_node] * -(((1.0 / node_vol[i_node] ^ 2) * (res2d[i_node] + -(mupd[i_node])) + -((res2[i_node] - mup[i_node]) / (node_vol[i_node] ^ 2) ^ 2) * ((2 * node_vol[i_node]) * node_vold[i_node]))) + -((res2[i_node] - mup[i_node]) / node_vol[i_node] ^ 2) * upbd[i_node])
    node_volb[i_node] = node_volb[i_node] + -((res2[i_node] - mup[i_node]) / node_vol[i_node] ^ 2) * upb[i_node]
    return nothing
end

function cuda_kernel_ttgc_hv_22!(i_ncell, i_njac, i_nnode, i_node_perio, i_seq_, mup, mup_stack, mup_stack_d, mupb, mupbd, mupd, npernode_half, resperiob, resperiobd)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(1 - npernode_half, -1) + 1
        return nothing
    end
    k = npernode_half + (__tid - 1) * -1
    i1 = i_node_perio[k, 1]
    i2 = i_node_perio[k, 2]
    mupd[i2] = mup_stack_d[(((((((((((((div(i_njac - 1, 1) + 1) * (div(i_nnode - 1, 1) + 1) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(npernode_half - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(npernode_half - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_nnode - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(npernode_half - 1, 1) + 1)) + (((i_seq_ - 1) * (div(npernode_half - 1, 1) + 1) + (k - 1)) + 1)]
    mup[i2] = mup_stack[(((((((((((((div(i_njac - 1, 1) + 1) * (div(i_nnode - 1, 1) + 1) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(npernode_half - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(npernode_half - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_nnode - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(npernode_half - 1, 1) + 1)) + (((i_seq_ - 1) * (div(npernode_half - 1, 1) + 1) + (k - 1)) + 1)]
    resperiobd[k, 1] = resperiobd[k, 1] + mupbd[i2]
    resperiob[k, 1] = resperiob[k, 1] + mupb[i2]
    mupd[i1] = mup_stack_d[((((((((((((div(i_njac - 1, 1) + 1) * (div(i_nnode - 1, 1) + 1) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(npernode_half - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(npernode_half - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_nnode - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (((i_seq_ - 1) * (div(npernode_half - 1, 1) + 1) + (k - 1)) + 1)]
    mup[i1] = mup_stack[((((((((((((div(i_njac - 1, 1) + 1) * (div(i_nnode - 1, 1) + 1) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(npernode_half - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(npernode_half - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_nnode - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (((i_seq_ - 1) * (div(npernode_half - 1, 1) + 1) + (k - 1)) + 1)]
    resperiobd[k, 2] = resperiobd[k, 2] + mupbd[i1]
    resperiob[k, 2] = resperiob[k, 2] + mupb[i1]
    return nothing
end

function cuda_kernel_ttgc_hv_23!(i_node_perio, mupb, mupbd, npernode_half, resperiob, resperiobd)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(npernode_half - 1, 1) + 1
        return nothing
    end
    k = 1 + (__tid - 1)
    i1 = i_node_perio[k, 1]
    i2 = i_node_perio[k, 2]
    CUDA.@atomic mupbd[i2] += resperiobd[k, 2]
    CUDA.@atomic mupb[i2] += resperiob[k, 2]
    resperiobd[k, 2] = 0.0
    resperiob[k, 2] = 0.0
    CUDA.@atomic mupbd[i1] += resperiobd[k, 1]
    CUDA.@atomic mupb[i1] += resperiob[k, 1]
    resperiobd[k, 1] = 0.0
    resperiob[k, 1] = 0.0
    return nothing
end

function cuda_kernel_ttgc_hv_24!(auxu_stack, auxu_stack_d, cell_vol, cell_volb, cell_volbd, cell_vold, i_cell_to_node, i_ncell, i_njac, i_nnode, i_seq_, mup, mup_stack, mup_stack_d, mupb, mupbd, mupd, npernode_half, up, upb, upbd, upd)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(1 - i_ncell, -1) + 1
        return nothing
    end
    i_cell = i_ncell + (__tid - 1) * -1
    i_node1 = i_cell_to_node[1, i_cell]
    i_node2 = i_cell_to_node[2, i_cell]
    i_node3 = i_cell_to_node[3, i_cell]
    i_node4 = i_cell_to_node[4, i_cell]
    mupd[i_node4] = mup_stack_d[(((((((((((div(i_njac - 1, 1) + 1) * (div(i_nnode - 1, 1) + 1) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(npernode_half - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(npernode_half - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_nnode - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (((i_seq_ - 1) * (div(i_ncell - 1, 1) + 1) + (i_cell - 1)) + 1)]
    mup[i_node4] = mup_stack[(((((((((((div(i_njac - 1, 1) + 1) * (div(i_nnode - 1, 1) + 1) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(npernode_half - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(npernode_half - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_nnode - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (((i_seq_ - 1) * (div(i_ncell - 1, 1) + 1) + (i_cell - 1)) + 1)]
    auxubd = auxubd + ((0.05 * mupb[i_node4]) * cell_vold[i_cell] + cell_vol[i_cell] * (0.05 * mupbd[i_node4]))
    auxub = auxub + cell_vol[i_cell] * (0.05 * mupb[i_node4])
    CUDA.@atomic upbd[i_node4] += (0.05 * mupb[i_node4]) * cell_vold[i_cell] + cell_vol[i_cell] * (0.05 * mupbd[i_node4])
    CUDA.@atomic upb[i_node4] += cell_vol[i_cell] * (0.05 * mupb[i_node4])
    cell_volbd[i_cell] = cell_volbd[i_cell] + ((0.05 * mupb[i_node4]) * (auxud + upd[i_node4]) + (auxu + up[i_node4]) * (0.05 * mupbd[i_node4]))
    cell_volb[i_cell] = cell_volb[i_cell] + (auxu + up[i_node4]) * (0.05 * mupb[i_node4])
    mupd[i_node3] = mup_stack_d[((((((((((div(i_njac - 1, 1) + 1) * (div(i_nnode - 1, 1) + 1) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(npernode_half - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(npernode_half - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_nnode - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (((i_seq_ - 1) * (div(i_ncell - 1, 1) + 1) + (i_cell - 1)) + 1)]
    mup[i_node3] = mup_stack[((((((((((div(i_njac - 1, 1) + 1) * (div(i_nnode - 1, 1) + 1) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(npernode_half - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(npernode_half - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_nnode - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (((i_seq_ - 1) * (div(i_ncell - 1, 1) + 1) + (i_cell - 1)) + 1)]
    auxubd = auxubd + ((0.05 * mupb[i_node3]) * cell_vold[i_cell] + cell_vol[i_cell] * (0.05 * mupbd[i_node3]))
    auxub = auxub + cell_vol[i_cell] * (0.05 * mupb[i_node3])
    CUDA.@atomic upbd[i_node3] += (0.05 * mupb[i_node3]) * cell_vold[i_cell] + cell_vol[i_cell] * (0.05 * mupbd[i_node3])
    CUDA.@atomic upb[i_node3] += cell_vol[i_cell] * (0.05 * mupb[i_node3])
    cell_volbd[i_cell] = cell_volbd[i_cell] + ((0.05 * mupb[i_node3]) * (auxud + upd[i_node3]) + (auxu + up[i_node3]) * (0.05 * mupbd[i_node3]))
    cell_volb[i_cell] = cell_volb[i_cell] + (auxu + up[i_node3]) * (0.05 * mupb[i_node3])
    mupd[i_node2] = mup_stack_d[(((((((((div(i_njac - 1, 1) + 1) * (div(i_nnode - 1, 1) + 1) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(npernode_half - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(npernode_half - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_nnode - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (((i_seq_ - 1) * (div(i_ncell - 1, 1) + 1) + (i_cell - 1)) + 1)]
    mup[i_node2] = mup_stack[(((((((((div(i_njac - 1, 1) + 1) * (div(i_nnode - 1, 1) + 1) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(npernode_half - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(npernode_half - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_nnode - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (((i_seq_ - 1) * (div(i_ncell - 1, 1) + 1) + (i_cell - 1)) + 1)]
    auxubd = auxubd + ((0.05 * mupb[i_node2]) * cell_vold[i_cell] + cell_vol[i_cell] * (0.05 * mupbd[i_node2]))
    auxub = auxub + cell_vol[i_cell] * (0.05 * mupb[i_node2])
    CUDA.@atomic upbd[i_node2] += (0.05 * mupb[i_node2]) * cell_vold[i_cell] + cell_vol[i_cell] * (0.05 * mupbd[i_node2])
    CUDA.@atomic upb[i_node2] += cell_vol[i_cell] * (0.05 * mupb[i_node2])
    cell_volbd[i_cell] = cell_volbd[i_cell] + ((0.05 * mupb[i_node2]) * (auxud + upd[i_node2]) + (auxu + up[i_node2]) * (0.05 * mupbd[i_node2]))
    cell_volb[i_cell] = cell_volb[i_cell] + (auxu + up[i_node2]) * (0.05 * mupb[i_node2])
    mupd[i_node1] = mup_stack_d[((((((((div(i_njac - 1, 1) + 1) * (div(i_nnode - 1, 1) + 1) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(npernode_half - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(npernode_half - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_nnode - 1, 1) + 1)) + (((i_seq_ - 1) * (div(i_ncell - 1, 1) + 1) + (i_cell - 1)) + 1)]
    mup[i_node1] = mup_stack[((((((((div(i_njac - 1, 1) + 1) * (div(i_nnode - 1, 1) + 1) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(npernode_half - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(npernode_half - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_nnode - 1, 1) + 1)) + (((i_seq_ - 1) * (div(i_ncell - 1, 1) + 1) + (i_cell - 1)) + 1)]
    auxubd = auxubd + ((0.05 * mupb[i_node1]) * cell_vold[i_cell] + cell_vol[i_cell] * (0.05 * mupbd[i_node1]))
    auxub = auxub + cell_vol[i_cell] * (0.05 * mupb[i_node1])
    CUDA.@atomic upbd[i_node1] += (0.05 * mupb[i_node1]) * cell_vold[i_cell] + cell_vol[i_cell] * (0.05 * mupbd[i_node1])
    CUDA.@atomic upb[i_node1] += cell_vol[i_cell] * (0.05 * mupb[i_node1])
    cell_volbd[i_cell] = cell_volbd[i_cell] + ((0.05 * mupb[i_node1]) * (auxud + upd[i_node1]) + (auxu + up[i_node1]) * (0.05 * mupbd[i_node1]))
    cell_volb[i_cell] = cell_volb[i_cell] + (auxu + up[i_node1]) * (0.05 * mupb[i_node1])
    auxud = auxu_stack_d[((div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1) + (div(i_njac - 1, 1) + 1)) + (((i_seq_ - 1) * (div(i_ncell - 1, 1) + 1) + (i_cell - 1)) + 1)]
    auxu = auxu_stack[((div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1) + (div(i_njac - 1, 1) + 1)) + (((i_seq_ - 1) * (div(i_ncell - 1, 1) + 1) + (i_cell - 1)) + 1)]
    CUDA.@atomic upbd[i_node1] += auxubd
    CUDA.@atomic upb[i_node1] += auxub
    CUDA.@atomic upbd[i_node2] += auxubd
    CUDA.@atomic upb[i_node2] += auxub
    CUDA.@atomic upbd[i_node3] += auxubd
    CUDA.@atomic upb[i_node3] += auxub
    CUDA.@atomic upbd[i_node4] += auxubd
    CUDA.@atomic upb[i_node4] += auxub
    auxubd = 0.0
    auxub = 0.0
    return nothing
end

function cuda_kernel_ttgc_hv_25!(i_ncell, i_njac, i_nnode, i_seq_, mup, mup_stack, mup_stack_d, mupb, mupbd, mupd, npernode_half)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(1 - i_nnode, -1) + 1
        return nothing
    end
    i_node = i_nnode + (__tid - 1) * -1
    mupd[i_node] = mup_stack_d[(((((((div(i_njac - 1, 1) + 1) * (div(i_nnode - 1, 1) + 1) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(npernode_half - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(npernode_half - 1, 1) + 1)) + (((i_seq_ - 1) * (div(i_nnode - 1, 1) + 1) + (i_node - 1)) + 1)]
    mup[i_node] = mup_stack[(((((((div(i_njac - 1, 1) + 1) * (div(i_nnode - 1, 1) + 1) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(npernode_half - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(npernode_half - 1, 1) + 1)) + (((i_seq_ - 1) * (div(i_nnode - 1, 1) + 1) + (i_node - 1)) + 1)]
    mupbd[i_node] = 0.0
    mupb[i_node] = 0.0
    return nothing
end

function cuda_kernel_ttgc_hv_26!(i_njac, i_nnode, node_vol, node_volb, node_volbd, node_vold, res2, res2b, res2bd, res2d, up, up_stack, up_stack_d, upb, upbd, upd)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(1 - i_nnode, -1) + 1
        return nothing
    end
    i_node = i_nnode + (__tid - 1) * -1
    upd[i_node] = up_stack_d[(div(i_njac - 1, 1) + 1) * (div(i_nnode - 1, 1) + 1) + ((i_node - 1) + 1)]
    up[i_node] = up_stack[(div(i_njac - 1, 1) + 1) * (div(i_nnode - 1, 1) + 1) + ((i_node - 1) + 1)]
    res2bd[i_node] = res2bd[i_node] + (upb[i_node] * (-(1.0 / node_vol[i_node] ^ 2) * node_vold[i_node]) + (1.0 / node_vol[i_node]) * upbd[i_node])
    res2b[i_node] = res2b[i_node] + (1.0 / node_vol[i_node]) * upb[i_node]
    node_volbd[i_node] = node_volbd[i_node] + (upb[i_node] * -(((1.0 / node_vol[i_node] ^ 2) * res2d[i_node] + -(res2[i_node] / (node_vol[i_node] ^ 2) ^ 2) * ((2 * node_vol[i_node]) * node_vold[i_node]))) + -(res2[i_node] / node_vol[i_node] ^ 2) * upbd[i_node])
    node_volb[i_node] = node_volb[i_node] + -(res2[i_node] / node_vol[i_node] ^ 2) * upb[i_node]
    upbd[i_node] = 0.0
    upb[i_node] = 0.0
    return nothing
end

function cuda_kernel_ttgc_hv_27!(i_node_perio, npernode_half, res2b, res2bd, resperiob, resperiobd)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(npernode_half - 1, 1) + 1
        return nothing
    end
    k = 1 + (__tid - 1)
    i1 = i_node_perio[k, 1]
    i2 = i_node_perio[k, 2]
    resperiobd[k, 1] = resperiobd[k, 1] + res2bd[i2]
    resperiob[k, 1] = resperiob[k, 1] + res2b[i2]
    resperiobd[k, 2] = resperiobd[k, 2] + res2bd[i1]
    resperiob[k, 2] = resperiob[k, 2] + res2b[i1]
    return nothing
end

function cuda_kernel_ttgc_hv_28!(i_node_perio, npernode_half, res2b, res2bd, resperiob, resperiobd)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(npernode_half - 1, 1) + 1
        return nothing
    end
    k = 1 + (__tid - 1)
    i1 = i_node_perio[k, 1]
    i2 = i_node_perio[k, 2]
    CUDA.@atomic res2bd[i2] += resperiobd[k, 2]
    CUDA.@atomic res2b[i2] += resperiob[k, 2]
    resperiobd[k, 2] = 0.0
    resperiob[k, 2] = 0.0
    CUDA.@atomic res2bd[i1] += resperiobd[k, 1]
    CUDA.@atomic res2b[i1] += resperiob[k, 1]
    resperiobd[k, 1] = 0.0
    resperiob[k, 1] = 0.0
    return nothing
end

function cuda_kernel_ttgc_hv_29!(c, cb, cbd, cd, dt, dtb, dtbd, dtd, i_cell_to_node, i_ncell, res2b, res2bd, skx, skxb, skxbd, skxd, sky, skyb, skybd, skyd, u, ub, ubd, ud, up, upb, upbd, upd, vere_stack, vere_stack_d)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(1 - i_ncell, -1) + 1
        return nothing
    end
    i_cell = i_ncell + (__tid - 1) * -1
    vered = vere_stack_d[(((div(i_ncell - 1, 1) + 1) * (div(4 - 1, 1) + 1) + (div(i_ncell - 1, 1) + 1)) + (div(i_ncell - 1, 1) + 1) * (div(4 - 1, 1) + 1)) + ((i_cell - 1) + 1)]
    vere = vere_stack[(((div(i_ncell - 1, 1) + 1) * (div(4 - 1, 1) + 1) + (div(i_ncell - 1, 1) + 1)) + (div(i_ncell - 1, 1) + 1) * (div(4 - 1, 1) + 1)) + ((i_cell - 1) + 1)]
    for i_loc = 4:-1:1
        i_node = i_cell_to_node[i_loc, i_cell]
        for i_k = 1:4
            i_k_node = i_cell_to_node[i_k, i_cell]
            resibd = resibd + res2bd[i_k_node]
            resib = resib + res2b[i_k_node]
        end
        CUDA.@atomic dtbd[1] += 0.25 * -((resib * vered + vere * resibd))
        CUDA.@atomic dtb[1] += 0.25 * -(vere * resib)
        verebd = verebd + (resib * -(0.25dtd) + -(dt / 4) * resibd)
        vereb = vereb + -(dt / 4) * resib
        resibd = 0.0
        resib = 0.0
        vered = vere_stack_d[((div(i_ncell - 1, 1) + 1) * (div(4 - 1, 1) + 1) + (div(i_ncell - 1, 1) + 1)) + (((i_cell - 1) * (div(4 - 1, 1) + 1) + (i_loc - 1)) + 1)]
        vere = vere_stack[((div(i_ncell - 1, 1) + 1) * (div(4 - 1, 1) + 1) + (div(i_ncell - 1, 1) + 1)) + (((i_cell - 1) * (div(4 - 1, 1) + 1) + (i_loc - 1)) + 1)]
        CUDA.@atomic ubd[i_node] += vereb * (-(1.0 / 3.0) * ((skx[i_loc, i_cell] * cd[1, i_node] + c[1, i_node] * skxd[i_loc, i_cell]) + (sky[i_loc, i_cell] * cd[2, i_node] + c[2, i_node] * skyd[i_loc, i_cell]))) + (-(1.0 / 3.0) * (c[1, i_node] * skx[i_loc, i_cell] + c[2, i_node] * sky[i_loc, i_cell])) * verebd
        CUDA.@atomic ub[i_node] += (-(1.0 / 3.0) * (c[1, i_node] * skx[i_loc, i_cell] + c[2, i_node] * sky[i_loc, i_cell])) * vereb
        CUDA.@atomic upbd[i_node] += vereb * (-(1.0 / 3.0) * ((skx[i_loc, i_cell] * cd[1, i_node] + c[1, i_node] * skxd[i_loc, i_cell]) + (sky[i_loc, i_cell] * cd[2, i_node] + c[2, i_node] * skyd[i_loc, i_cell]))) + (-(1.0 / 3.0) * (c[1, i_node] * skx[i_loc, i_cell] + c[2, i_node] * sky[i_loc, i_cell])) * verebd
        CUDA.@atomic upb[i_node] += (-(1.0 / 3.0) * (c[1, i_node] * skx[i_loc, i_cell] + c[2, i_node] * sky[i_loc, i_cell])) * vereb
        CUDA.@atomic cbd[1, i_node] += ((-(1.0 / 3.0) * (u[i_node] + up[i_node])) * vereb) * skxd[i_loc, i_cell] + skx[i_loc, i_cell] * (vereb * (-(1.0 / 3.0) * (ud[i_node] + upd[i_node])) + (-(1.0 / 3.0) * (u[i_node] + up[i_node])) * verebd)
        CUDA.@atomic cb[1, i_node] += skx[i_loc, i_cell] * ((-(1.0 / 3.0) * (u[i_node] + up[i_node])) * vereb)
        skxbd[i_loc, i_cell] = skxbd[i_loc, i_cell] + (((-(1.0 / 3.0) * (u[i_node] + up[i_node])) * vereb) * cd[1, i_node] + c[1, i_node] * (vereb * (-(1.0 / 3.0) * (ud[i_node] + upd[i_node])) + (-(1.0 / 3.0) * (u[i_node] + up[i_node])) * verebd))
        skxb[i_loc, i_cell] = skxb[i_loc, i_cell] + c[1, i_node] * ((-(1.0 / 3.0) * (u[i_node] + up[i_node])) * vereb)
        CUDA.@atomic cbd[2, i_node] += ((-(1.0 / 3.0) * (u[i_node] + up[i_node])) * vereb) * skyd[i_loc, i_cell] + sky[i_loc, i_cell] * (vereb * (-(1.0 / 3.0) * (ud[i_node] + upd[i_node])) + (-(1.0 / 3.0) * (u[i_node] + up[i_node])) * verebd)
        CUDA.@atomic cb[2, i_node] += sky[i_loc, i_cell] * ((-(1.0 / 3.0) * (u[i_node] + up[i_node])) * vereb)
        skybd[i_loc, i_cell] = skybd[i_loc, i_cell] + (((-(1.0 / 3.0) * (u[i_node] + up[i_node])) * vereb) * cd[2, i_node] + c[2, i_node] * (vereb * (-(1.0 / 3.0) * (ud[i_node] + upd[i_node])) + (-(1.0 / 3.0) * (u[i_node] + up[i_node])) * verebd))
        skyb[i_loc, i_cell] = skyb[i_loc, i_cell] + c[2, i_node] * ((-(1.0 / 3.0) * (u[i_node] + up[i_node])) * vereb)
        verebd = 0.0
        vereb = 0.0
    end
    return nothing
end

function cuda_kernel_ttgc_hv_30!(i_nnode, i_seq_, mup, mupb, mupbd, mupd, node_vol, node_volb, node_volbd, node_vold, res, resb, resbd, resd, up, up_stack, up_stack_d, upb, upbd, upd)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(1 - i_nnode, -1) + 1
        return nothing
    end
    i_node = i_nnode + (__tid - 1) * -1
    upd[i_node] = up_stack_d[((i_seq_ - 1) * (div(i_nnode - 1, 1) + 1) + (i_node - 1)) + 1]
    up[i_node] = up_stack[((i_seq_ - 1) * (div(i_nnode - 1, 1) + 1) + (i_node - 1)) + 1]
    resbd[i_node] = resbd[i_node] + (upb[i_node] * (-(1.0 / node_vol[i_node] ^ 2) * node_vold[i_node]) + (1.0 / node_vol[i_node]) * upbd[i_node])
    resb[i_node] = resb[i_node] + (1.0 / node_vol[i_node]) * upb[i_node]
    mupbd[i_node] = mupbd[i_node] + -((upb[i_node] * (-(1.0 / node_vol[i_node] ^ 2) * node_vold[i_node]) + (1.0 / node_vol[i_node]) * upbd[i_node]))
    mupb[i_node] = mupb[i_node] + -((1.0 / node_vol[i_node]) * upb[i_node])
    node_volbd[i_node] = node_volbd[i_node] + (upb[i_node] * -(((1.0 / node_vol[i_node] ^ 2) * (resd[i_node] + -(mupd[i_node])) + -((res[i_node] - mup[i_node]) / (node_vol[i_node] ^ 2) ^ 2) * ((2 * node_vol[i_node]) * node_vold[i_node]))) + -((res[i_node] - mup[i_node]) / node_vol[i_node] ^ 2) * upbd[i_node])
    node_volb[i_node] = node_volb[i_node] + -((res[i_node] - mup[i_node]) / node_vol[i_node] ^ 2) * upb[i_node]
    return nothing
end

function cuda_kernel_ttgc_hv_31!(i_ncell, i_njac, i_nnode, i_node_perio, i_seq_, mup, mup_stack, mup_stack_d, mupb, mupbd, mupd, npernode_half, resperiob, resperiobd)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(1 - npernode_half, -1) + 1
        return nothing
    end
    k = npernode_half + (__tid - 1) * -1
    i1 = i_node_perio[k, 1]
    i2 = i_node_perio[k, 2]
    mupd[i2] = mup_stack_d[((((((div(i_njac - 1, 1) + 1) * (div(i_nnode - 1, 1) + 1) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(npernode_half - 1, 1) + 1)) + (((i_seq_ - 1) * (div(npernode_half - 1, 1) + 1) + (k - 1)) + 1)]
    mup[i2] = mup_stack[((((((div(i_njac - 1, 1) + 1) * (div(i_nnode - 1, 1) + 1) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(npernode_half - 1, 1) + 1)) + (((i_seq_ - 1) * (div(npernode_half - 1, 1) + 1) + (k - 1)) + 1)]
    resperiobd[k, 1] = resperiobd[k, 1] + mupbd[i2]
    resperiob[k, 1] = resperiob[k, 1] + mupb[i2]
    mupd[i1] = mup_stack_d[(((((div(i_njac - 1, 1) + 1) * (div(i_nnode - 1, 1) + 1) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (((i_seq_ - 1) * (div(npernode_half - 1, 1) + 1) + (k - 1)) + 1)]
    mup[i1] = mup_stack[(((((div(i_njac - 1, 1) + 1) * (div(i_nnode - 1, 1) + 1) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (((i_seq_ - 1) * (div(npernode_half - 1, 1) + 1) + (k - 1)) + 1)]
    resperiobd[k, 2] = resperiobd[k, 2] + mupbd[i1]
    resperiob[k, 2] = resperiob[k, 2] + mupb[i1]
    return nothing
end

function cuda_kernel_ttgc_hv_32!(i_node_perio, mupb, mupbd, npernode_half, resperiob, resperiobd)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(npernode_half - 1, 1) + 1
        return nothing
    end
    k = 1 + (__tid - 1)
    i1 = i_node_perio[k, 1]
    i2 = i_node_perio[k, 2]
    CUDA.@atomic mupbd[i2] += resperiobd[k, 2]
    CUDA.@atomic mupb[i2] += resperiob[k, 2]
    resperiobd[k, 2] = 0.0
    resperiob[k, 2] = 0.0
    CUDA.@atomic mupbd[i1] += resperiobd[k, 1]
    CUDA.@atomic mupb[i1] += resperiob[k, 1]
    resperiobd[k, 1] = 0.0
    resperiob[k, 1] = 0.0
    return nothing
end

function cuda_kernel_ttgc_hv_33!(auxu_stack, auxu_stack_d, cell_vol, cell_volb, cell_volbd, cell_vold, i_cell_to_node, i_ncell, i_njac, i_nnode, i_seq_, mup, mup_stack, mup_stack_d, mupb, mupbd, mupd, up, upb, upbd, upd)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(1 - i_ncell, -1) + 1
        return nothing
    end
    i_cell = i_ncell + (__tid - 1) * -1
    i_node1 = i_cell_to_node[1, i_cell]
    i_node2 = i_cell_to_node[2, i_cell]
    i_node3 = i_cell_to_node[3, i_cell]
    i_node4 = i_cell_to_node[4, i_cell]
    mupd[i_node4] = mup_stack_d[((((div(i_njac - 1, 1) + 1) * (div(i_nnode - 1, 1) + 1) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (((i_seq_ - 1) * (div(i_ncell - 1, 1) + 1) + (i_cell - 1)) + 1)]
    mup[i_node4] = mup_stack[((((div(i_njac - 1, 1) + 1) * (div(i_nnode - 1, 1) + 1) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (((i_seq_ - 1) * (div(i_ncell - 1, 1) + 1) + (i_cell - 1)) + 1)]
    auxubd = auxubd + ((0.05 * mupb[i_node4]) * cell_vold[i_cell] + cell_vol[i_cell] * (0.05 * mupbd[i_node4]))
    auxub = auxub + cell_vol[i_cell] * (0.05 * mupb[i_node4])
    CUDA.@atomic upbd[i_node4] += (0.05 * mupb[i_node4]) * cell_vold[i_cell] + cell_vol[i_cell] * (0.05 * mupbd[i_node4])
    CUDA.@atomic upb[i_node4] += cell_vol[i_cell] * (0.05 * mupb[i_node4])
    cell_volbd[i_cell] = cell_volbd[i_cell] + ((0.05 * mupb[i_node4]) * (auxud + upd[i_node4]) + (auxu + up[i_node4]) * (0.05 * mupbd[i_node4]))
    cell_volb[i_cell] = cell_volb[i_cell] + (auxu + up[i_node4]) * (0.05 * mupb[i_node4])
    mupd[i_node3] = mup_stack_d[(((div(i_njac - 1, 1) + 1) * (div(i_nnode - 1, 1) + 1) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (((i_seq_ - 1) * (div(i_ncell - 1, 1) + 1) + (i_cell - 1)) + 1)]
    mup[i_node3] = mup_stack[(((div(i_njac - 1, 1) + 1) * (div(i_nnode - 1, 1) + 1) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (((i_seq_ - 1) * (div(i_ncell - 1, 1) + 1) + (i_cell - 1)) + 1)]
    auxubd = auxubd + ((0.05 * mupb[i_node3]) * cell_vold[i_cell] + cell_vol[i_cell] * (0.05 * mupbd[i_node3]))
    auxub = auxub + cell_vol[i_cell] * (0.05 * mupb[i_node3])
    CUDA.@atomic upbd[i_node3] += (0.05 * mupb[i_node3]) * cell_vold[i_cell] + cell_vol[i_cell] * (0.05 * mupbd[i_node3])
    CUDA.@atomic upb[i_node3] += cell_vol[i_cell] * (0.05 * mupb[i_node3])
    cell_volbd[i_cell] = cell_volbd[i_cell] + ((0.05 * mupb[i_node3]) * (auxud + upd[i_node3]) + (auxu + up[i_node3]) * (0.05 * mupbd[i_node3]))
    cell_volb[i_cell] = cell_volb[i_cell] + (auxu + up[i_node3]) * (0.05 * mupb[i_node3])
    mupd[i_node2] = mup_stack_d[((div(i_njac - 1, 1) + 1) * (div(i_nnode - 1, 1) + 1) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (((i_seq_ - 1) * (div(i_ncell - 1, 1) + 1) + (i_cell - 1)) + 1)]
    mup[i_node2] = mup_stack[((div(i_njac - 1, 1) + 1) * (div(i_nnode - 1, 1) + 1) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (((i_seq_ - 1) * (div(i_ncell - 1, 1) + 1) + (i_cell - 1)) + 1)]
    auxubd = auxubd + ((0.05 * mupb[i_node2]) * cell_vold[i_cell] + cell_vol[i_cell] * (0.05 * mupbd[i_node2]))
    auxub = auxub + cell_vol[i_cell] * (0.05 * mupb[i_node2])
    CUDA.@atomic upbd[i_node2] += (0.05 * mupb[i_node2]) * cell_vold[i_cell] + cell_vol[i_cell] * (0.05 * mupbd[i_node2])
    CUDA.@atomic upb[i_node2] += cell_vol[i_cell] * (0.05 * mupb[i_node2])
    cell_volbd[i_cell] = cell_volbd[i_cell] + ((0.05 * mupb[i_node2]) * (auxud + upd[i_node2]) + (auxu + up[i_node2]) * (0.05 * mupbd[i_node2]))
    cell_volb[i_cell] = cell_volb[i_cell] + (auxu + up[i_node2]) * (0.05 * mupb[i_node2])
    mupd[i_node1] = mup_stack_d[(div(i_njac - 1, 1) + 1) * (div(i_nnode - 1, 1) + 1) + (((i_seq_ - 1) * (div(i_ncell - 1, 1) + 1) + (i_cell - 1)) + 1)]
    mup[i_node1] = mup_stack[(div(i_njac - 1, 1) + 1) * (div(i_nnode - 1, 1) + 1) + (((i_seq_ - 1) * (div(i_ncell - 1, 1) + 1) + (i_cell - 1)) + 1)]
    auxubd = auxubd + ((0.05 * mupb[i_node1]) * cell_vold[i_cell] + cell_vol[i_cell] * (0.05 * mupbd[i_node1]))
    auxub = auxub + cell_vol[i_cell] * (0.05 * mupb[i_node1])
    CUDA.@atomic upbd[i_node1] += (0.05 * mupb[i_node1]) * cell_vold[i_cell] + cell_vol[i_cell] * (0.05 * mupbd[i_node1])
    CUDA.@atomic upb[i_node1] += cell_vol[i_cell] * (0.05 * mupb[i_node1])
    cell_volbd[i_cell] = cell_volbd[i_cell] + ((0.05 * mupb[i_node1]) * (auxud + upd[i_node1]) + (auxu + up[i_node1]) * (0.05 * mupbd[i_node1]))
    cell_volb[i_cell] = cell_volb[i_cell] + (auxu + up[i_node1]) * (0.05 * mupb[i_node1])
    auxud = auxu_stack_d[((i_seq_ - 1) * (div(i_ncell - 1, 1) + 1) + (i_cell - 1)) + 1]
    auxu = auxu_stack[((i_seq_ - 1) * (div(i_ncell - 1, 1) + 1) + (i_cell - 1)) + 1]
    CUDA.@atomic upbd[i_node1] += auxubd
    CUDA.@atomic upb[i_node1] += auxub
    CUDA.@atomic upbd[i_node2] += auxubd
    CUDA.@atomic upb[i_node2] += auxub
    CUDA.@atomic upbd[i_node3] += auxubd
    CUDA.@atomic upb[i_node3] += auxub
    CUDA.@atomic upbd[i_node4] += auxubd
    CUDA.@atomic upb[i_node4] += auxub
    auxubd = 0.0
    auxub = 0.0
    return nothing
end

function cuda_kernel_ttgc_hv_34!(i_nnode, i_seq_, mup, mup_stack, mup_stack_d, mupb, mupbd, mupd)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(1 - i_nnode, -1) + 1
        return nothing
    end
    i_node = i_nnode + (__tid - 1) * -1
    mupd[i_node] = mup_stack_d[((i_seq_ - 1) * (div(i_nnode - 1, 1) + 1) + (i_node - 1)) + 1]
    mup[i_node] = mup_stack[((i_seq_ - 1) * (div(i_nnode - 1, 1) + 1) + (i_node - 1)) + 1]
    mupbd[i_node] = 0.0
    mupb[i_node] = 0.0
    return nothing
end

function cuda_kernel_ttgc_hv_35!(i_nnode, node_vol, node_volb, node_volbd, node_vold, res, resb, resbd, resd, upb, upbd)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(i_nnode - 1, 1) + 1
        return nothing
    end
    i_node = 1 + (__tid - 1)
    resbd[i_node] = resbd[i_node] + (upb[i_node] * (-(1.0 / node_vol[i_node] ^ 2) * node_vold[i_node]) + (1.0 / node_vol[i_node]) * upbd[i_node])
    resb[i_node] = resb[i_node] + (1.0 / node_vol[i_node]) * upb[i_node]
    node_volbd[i_node] = node_volbd[i_node] + (upb[i_node] * -(((1.0 / node_vol[i_node] ^ 2) * resd[i_node] + -(res[i_node] / (node_vol[i_node] ^ 2) ^ 2) * ((2 * node_vol[i_node]) * node_vold[i_node]))) + -(res[i_node] / node_vol[i_node] ^ 2) * upbd[i_node])
    node_volb[i_node] = node_volb[i_node] + -(res[i_node] / node_vol[i_node] ^ 2) * upb[i_node]
    upbd[i_node] = 0.0
    upb[i_node] = 0.0
    return nothing
end

function cuda_kernel_ttgc_hv_36!(i_node_perio, npernode_half, resb, resbd, resperiob, resperiobd)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(npernode_half - 1, 1) + 1
        return nothing
    end
    k = 1 + (__tid - 1)
    i1 = i_node_perio[k, 1]
    i2 = i_node_perio[k, 2]
    resperiobd[k, 1] = resperiobd[k, 1] + resbd[i2]
    resperiob[k, 1] = resperiob[k, 1] + resb[i2]
    resperiobd[k, 2] = resperiobd[k, 2] + resbd[i1]
    resperiob[k, 2] = resperiob[k, 2] + resb[i1]
    return nothing
end

function cuda_kernel_ttgc_hv_37!(i_node_perio, npernode_half, resb, resbd, resperiob, resperiobd)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(npernode_half - 1, 1) + 1
        return nothing
    end
    k = 1 + (__tid - 1)
    i1 = i_node_perio[k, 1]
    i2 = i_node_perio[k, 2]
    CUDA.@atomic resbd[i2] += resperiobd[k, 2]
    CUDA.@atomic resb[i2] += resperiob[k, 2]
    resperiobd[k, 2] = 0.0
    resperiob[k, 2] = 0.0
    CUDA.@atomic resbd[i1] += resperiobd[k, 1]
    CUDA.@atomic resb[i1] += resperiob[k, 1]
    resperiobd[k, 1] = 0.0
    resperiob[k, 1] = 0.0
    return nothing
end

function cuda_kernel_ttgc_hv_38!(aeresk_stack, aeresk_stack_d, aerex_stack, aerex_stack_d, aerey_stack, aerey_stack_d, aerez_stack, aerez_stack_d, beta, betab, betabd, betad, c, cavgx_stack, cavgx_stack_d, cavgy_stack, cavgy_stack_d, cavgz_stack, cavgz_stack_d, cb, cbd, cd, cell_vol, cell_volb, cell_volbd, cell_vold, dt, dtb, dtbd, dtd, factor_stack, factor_stack_d, gamma, gammab, gammabd, gammad, i_cell_to_node, i_ncell, re_stack, re_stack_d, res2b, res2bd, resb, resbd, skx, skxb, skxbd, skxd, sky, skyb, skybd, skyd, skz, skzb, skzbd, skzd, u, ub, ubd, ud, vere_stack, vere_stack_d)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(1 - i_ncell, -1) + 1
        return nothing
    end
    i_cell = i_ncell + (__tid - 1) * -1
    aereskd = aeresk_stack_d[((div(i_ncell - 1, 1) + 1) * (div(4 - 1, 1) + 1) * (div(4 - 1, 1) + 1) + (div(i_ncell - 1, 1) + 1) * (div(4 - 1, 1) + 1)) + ((i_cell - 1) + 1)]
    aeresk = aeresk_stack[((div(i_ncell - 1, 1) + 1) * (div(4 - 1, 1) + 1) * (div(4 - 1, 1) + 1) + (div(i_ncell - 1, 1) + 1) * (div(4 - 1, 1) + 1)) + ((i_cell - 1) + 1)]
    aerexd = aerex_stack_d[(div(i_ncell - 1, 1) + 1) * (div(4 - 1, 1) + 1) + ((i_cell - 1) + 1)]
    aerex = aerex_stack[(div(i_ncell - 1, 1) + 1) * (div(4 - 1, 1) + 1) + ((i_cell - 1) + 1)]
    aereyd = aerey_stack_d[(div(i_ncell - 1, 1) + 1) * (div(4 - 1, 1) + 1) + ((i_cell - 1) + 1)]
    aerey = aerey_stack[(div(i_ncell - 1, 1) + 1) * (div(4 - 1, 1) + 1) + ((i_cell - 1) + 1)]
    aerezd = aerez_stack_d[(div(i_ncell - 1, 1) + 1) * (div(4 - 1, 1) + 1) + ((i_cell - 1) + 1)]
    aerez = aerez_stack[(div(i_ncell - 1, 1) + 1) * (div(4 - 1, 1) + 1) + ((i_cell - 1) + 1)]
    cavgxd = cavgx_stack_d[((div(i_ncell - 1, 1) + 1) + (div(i_ncell - 1, 1) + 1) * (div(4 - 1, 1) + 1)) + ((i_cell - 1) + 1)]
    cavgx = cavgx_stack[((div(i_ncell - 1, 1) + 1) + (div(i_ncell - 1, 1) + 1) * (div(4 - 1, 1) + 1)) + ((i_cell - 1) + 1)]
    cavgyd = cavgy_stack_d[((div(i_ncell - 1, 1) + 1) + (div(i_ncell - 1, 1) + 1) * (div(4 - 1, 1) + 1)) + ((i_cell - 1) + 1)]
    cavgy = cavgy_stack[((div(i_ncell - 1, 1) + 1) + (div(i_ncell - 1, 1) + 1) * (div(4 - 1, 1) + 1)) + ((i_cell - 1) + 1)]
    cavgzd = cavgz_stack_d[((div(i_ncell - 1, 1) + 1) + (div(i_ncell - 1, 1) + 1) * (div(4 - 1, 1) + 1)) + ((i_cell - 1) + 1)]
    cavgz = cavgz_stack[((div(i_ncell - 1, 1) + 1) + (div(i_ncell - 1, 1) + 1) * (div(4 - 1, 1) + 1)) + ((i_cell - 1) + 1)]
    factord = factor_stack_d[((div(i_ncell - 1, 1) + 1) * (div(4 - 1, 1) + 1) * (div(4 - 1, 1) + 1) + (div(i_ncell - 1, 1) + 1) * (div(4 - 1, 1) + 1)) + ((i_cell - 1) + 1)]
    factor = factor_stack[((div(i_ncell - 1, 1) + 1) * (div(4 - 1, 1) + 1) * (div(4 - 1, 1) + 1) + (div(i_ncell - 1, 1) + 1) * (div(4 - 1, 1) + 1)) + ((i_cell - 1) + 1)]
    red = re_stack_d[(div(i_ncell - 1, 1) + 1) * (div(4 - 1, 1) + 1) + ((i_cell - 1) + 1)]
    re = re_stack[(div(i_ncell - 1, 1) + 1) * (div(4 - 1, 1) + 1) + ((i_cell - 1) + 1)]
    vered = vere_stack_d[(div(i_ncell - 1, 1) + 1) * (div(4 - 1, 1) + 1) + ((i_cell - 1) + 1)]
    vere = vere_stack[(div(i_ncell - 1, 1) + 1) * (div(4 - 1, 1) + 1) + ((i_cell - 1) + 1)]
    for i_loc = 4:-1:1
        aereskd = aeresk_stack_d[(div(i_ncell - 1, 1) + 1) * (div(4 - 1, 1) + 1) * (div(4 - 1, 1) + 1) + (((i_cell - 1) * (div(4 - 1, 1) + 1) + (i_loc - 1)) + 1)]
        aeresk = aeresk_stack[(div(i_ncell - 1, 1) + 1) * (div(4 - 1, 1) + 1) * (div(4 - 1, 1) + 1) + (((i_cell - 1) * (div(4 - 1, 1) + 1) + (i_loc - 1)) + 1)]
        factord = factor_stack_d[(div(i_ncell - 1, 1) + 1) * (div(4 - 1, 1) + 1) * (div(4 - 1, 1) + 1) + (((i_cell - 1) * (div(4 - 1, 1) + 1) + (i_loc - 1)) + 1)]
        factor = factor_stack[(div(i_ncell - 1, 1) + 1) * (div(4 - 1, 1) + 1) * (div(4 - 1, 1) + 1) + (((i_cell - 1) * (div(4 - 1, 1) + 1) + (i_loc - 1)) + 1)]
        i_node = i_cell_to_node[i_loc, i_cell]
        for i_k = 4:-1:1
            i_k_node = i_cell_to_node[i_k, i_cell]
            auxres2bd = auxres2bd + res2bd[i_k_node]
            auxres2b = auxres2b + res2b[i_k_node]
            auxresbd = auxresbd + resbd[i_k_node]
            auxresb = auxresb + resb[i_k_node]
            factorbd = factorbd + (auxres2b * gammad[i_cell] + gamma[i_cell] * auxres2bd)
            factorb = factorb + gamma[i_cell] * auxres2b
            gammabd[i_cell] = gammabd[i_cell] + (auxres2b * factord + factor * auxres2bd)
            gammab[i_cell] = gammab[i_cell] + factor * auxres2b
            auxres2bd = 0.0
            auxres2b = 0.0
            resibd = resibd + auxresbd
            resib = resib + auxresb
            factorbd = factorbd + (auxresb * betad[i_cell] + beta[i_cell] * auxresbd)
            factorb = factorb + beta[i_cell] * auxresb
            betabd[i_cell] = betabd[i_cell] + (auxresb * factord + factor * auxresbd)
            betab[i_cell] = betab[i_cell] + factor * auxresb
            auxresbd = 0.0
            auxresb = 0.0
            factord = factor_stack_d[((i_cell - 1) * ((div(4 - 1, 1) + 1) * (div(4 - 1, 1) + 1)) + (i_loc - 1) * (div(4 - 1, 1) + 1) + (i_k - 1)) + 1]
            factor = factor_stack[((i_cell - 1) * ((div(4 - 1, 1) + 1) * (div(4 - 1, 1) + 1)) + (i_loc - 1) * (div(4 - 1, 1) + 1) + (i_k - 1)) + 1]
            CUDA.@atomic dtbd[1] += (0.3333333333333333 * (aeresk * factorb)) * (2dtd) + (2dt) * (0.3333333333333333 * (factorb * aereskd + aeresk * factorbd))
            CUDA.@atomic dtb[1] += (2dt) * (0.3333333333333333 * (aeresk * factorb))
            aereskbd = aereskbd + (factorb * (0.3333333333333333 * ((2dt) * dtd)) + (dt ^ 2 / 3) * factorbd)
            aereskb = aereskb + (dt ^ 2 / 3) * factorb
            factorbd = 0.0
            factorb = 0.0
            aereskd = aeresk_stack_d[((i_cell - 1) * ((div(4 - 1, 1) + 1) * (div(4 - 1, 1) + 1)) + (i_loc - 1) * (div(4 - 1, 1) + 1) + (i_k - 1)) + 1]
            aeresk = aeresk_stack[((i_cell - 1) * ((div(4 - 1, 1) + 1) * (div(4 - 1, 1) + 1)) + (i_loc - 1) * (div(4 - 1, 1) + 1) + (i_k - 1)) + 1]
            aerexbd = aerexbd + (aereskb * skxd[i_k, i_cell] + skx[i_k, i_cell] * aereskbd)
            aerexb = aerexb + skx[i_k, i_cell] * aereskb
            skxbd[i_k, i_cell] = skxbd[i_k, i_cell] + (aereskb * aerexd + aerex * aereskbd)
            skxb[i_k, i_cell] = skxb[i_k, i_cell] + aerex * aereskb
            aereybd = aereybd + (aereskb * skyd[i_k, i_cell] + sky[i_k, i_cell] * aereskbd)
            aereyb = aereyb + sky[i_k, i_cell] * aereskb
            skybd[i_k, i_cell] = skybd[i_k, i_cell] + (aereskb * aereyd + aerey * aereskbd)
            skyb[i_k, i_cell] = skyb[i_k, i_cell] + aerey * aereskb
            aerezbd = aerezbd + (aereskb * skzd[i_k, i_cell] + skz[i_k, i_cell] * aereskbd)
            aerezb = aerezb + skz[i_k, i_cell] * aereskb
            skzbd[i_k, i_cell] = skzbd[i_k, i_cell] + (aereskb * aerezd + aerez * aereskbd)
            skzb[i_k, i_cell] = skzb[i_k, i_cell] + aerez * aereskb
            aereskbd = 0.0
            aereskb = 0.0
        end
        CUDA.@atomic dtbd[1] += 0.25 * -((resib * (vere * -(gammad[i_cell]) + (0.5 - gamma[i_cell]) * vered) + ((0.5 - gamma[i_cell]) * vere) * resibd))
        CUDA.@atomic dtb[1] += 0.25 * -(((0.5 - gamma[i_cell]) * vere) * resib)
        gammabd[i_cell] = gammabd[i_cell] + -((resib * (vere * -(0.25dtd) + -(dt / 4) * vered) + (-(dt / 4) * vere) * resibd))
        gammab[i_cell] = gammab[i_cell] + -((-(dt / 4) * vere) * resib)
        verebd = verebd + (resib * ((0.5 - gamma[i_cell]) * -(0.25dtd) + -(dt / 4) * -(gammad[i_cell])) + (-(dt / 4) * (0.5 - gamma[i_cell])) * resibd)
        vereb = vereb + (-(dt / 4) * (0.5 - gamma[i_cell])) * resib
        resibd = 0.0
        resib = 0.0
        aerezd = aerez_stack_d[((i_cell - 1) * (div(4 - 1, 1) + 1) + (i_loc - 1)) + 1]
        aerez = aerez_stack[((i_cell - 1) * (div(4 - 1, 1) + 1) + (i_loc - 1)) + 1]
        cavgzbd = cavgzbd + (aerezb * red + re * aerezbd)
        cavgzb = cavgzb + re * aerezb
        rebd = rebd + (aerezb * cavgzd + cavgz * aerezbd)
        reb = reb + cavgz * aerezb
        aerezbd = 0.0
        aerezb = 0.0
        aereyd = aerey_stack_d[((i_cell - 1) * (div(4 - 1, 1) + 1) + (i_loc - 1)) + 1]
        aerey = aerey_stack[((i_cell - 1) * (div(4 - 1, 1) + 1) + (i_loc - 1)) + 1]
        cavgybd = cavgybd + (aereyb * red + re * aereybd)
        cavgyb = cavgyb + re * aereyb
        rebd = rebd + (aereyb * cavgyd + cavgy * aereybd)
        reb = reb + cavgy * aereyb
        aereybd = 0.0
        aereyb = 0.0
        aerexd = aerex_stack_d[((i_cell - 1) * (div(4 - 1, 1) + 1) + (i_loc - 1)) + 1]
        aerex = aerex_stack[((i_cell - 1) * (div(4 - 1, 1) + 1) + (i_loc - 1)) + 1]
        cavgxbd = cavgxbd + (aerexb * red + re * aerexbd)
        cavgxb = cavgxb + re * aerexb
        rebd = rebd + (aerexb * cavgxd + cavgx * aerexbd)
        reb = reb + cavgx * aerexb
        aerexbd = 0.0
        aerexb = 0.0
        red = re_stack_d[((i_cell - 1) * (div(4 - 1, 1) + 1) + (i_loc - 1)) + 1]
        re = re_stack[((i_cell - 1) * (div(4 - 1, 1) + 1) + (i_loc - 1)) + 1]
        verebd = verebd + (reb * (-(1.0 / cell_vol[i_cell] ^ 2) * cell_vold[i_cell]) + (1.0 / cell_vol[i_cell]) * rebd)
        vereb = vereb + (1.0 / cell_vol[i_cell]) * reb
        cell_volbd[i_cell] = cell_volbd[i_cell] + (reb * -(((1.0 / cell_vol[i_cell] ^ 2) * vered + -(vere / (cell_vol[i_cell] ^ 2) ^ 2) * ((2 * cell_vol[i_cell]) * cell_vold[i_cell]))) + -(vere / cell_vol[i_cell] ^ 2) * rebd)
        cell_volb[i_cell] = cell_volb[i_cell] + -(vere / cell_vol[i_cell] ^ 2) * reb
        rebd = 0.0
        reb = 0.0
        vered = vere_stack_d[((i_cell - 1) * (div(4 - 1, 1) + 1) + (i_loc - 1)) + 1]
        vere = vere_stack[((i_cell - 1) * (div(4 - 1, 1) + 1) + (i_loc - 1)) + 1]
        CUDA.@atomic ubd[i_node] += vereb * (-(1.0 / 3.0) * (((skx[i_loc, i_cell] * cd[1, i_node] + c[1, i_node] * skxd[i_loc, i_cell]) + (sky[i_loc, i_cell] * cd[2, i_node] + c[2, i_node] * skyd[i_loc, i_cell])) + (skz[i_loc, i_cell] * cd[3, i_node] + c[3, i_node] * skzd[i_loc, i_cell]))) + (-(1.0 / 3.0) * (c[1, i_node] * skx[i_loc, i_cell] + c[2, i_node] * sky[i_loc, i_cell] + c[3, i_node] * skz[i_loc, i_cell])) * verebd
        CUDA.@atomic ub[i_node] += (-(1.0 / 3.0) * (c[1, i_node] * skx[i_loc, i_cell] + c[2, i_node] * sky[i_loc, i_cell] + c[3, i_node] * skz[i_loc, i_cell])) * vereb
        CUDA.@atomic cbd[1, i_node] += ((-(1.0 / 3.0) * u[i_node]) * vereb) * skxd[i_loc, i_cell] + skx[i_loc, i_cell] * (vereb * (-(1.0 / 3.0) * ud[i_node]) + (-(1.0 / 3.0) * u[i_node]) * verebd)
        CUDA.@atomic cb[1, i_node] += skx[i_loc, i_cell] * ((-(1.0 / 3.0) * u[i_node]) * vereb)
        skxbd[i_loc, i_cell] = skxbd[i_loc, i_cell] + (((-(1.0 / 3.0) * u[i_node]) * vereb) * cd[1, i_node] + c[1, i_node] * (vereb * (-(1.0 / 3.0) * ud[i_node]) + (-(1.0 / 3.0) * u[i_node]) * verebd))
        skxb[i_loc, i_cell] = skxb[i_loc, i_cell] + c[1, i_node] * ((-(1.0 / 3.0) * u[i_node]) * vereb)
        CUDA.@atomic cbd[2, i_node] += ((-(1.0 / 3.0) * u[i_node]) * vereb) * skyd[i_loc, i_cell] + sky[i_loc, i_cell] * (vereb * (-(1.0 / 3.0) * ud[i_node]) + (-(1.0 / 3.0) * u[i_node]) * verebd)
        CUDA.@atomic cb[2, i_node] += sky[i_loc, i_cell] * ((-(1.0 / 3.0) * u[i_node]) * vereb)
        skybd[i_loc, i_cell] = skybd[i_loc, i_cell] + (((-(1.0 / 3.0) * u[i_node]) * vereb) * cd[2, i_node] + c[2, i_node] * (vereb * (-(1.0 / 3.0) * ud[i_node]) + (-(1.0 / 3.0) * u[i_node]) * verebd))
        skyb[i_loc, i_cell] = skyb[i_loc, i_cell] + c[2, i_node] * ((-(1.0 / 3.0) * u[i_node]) * vereb)
        CUDA.@atomic cbd[3, i_node] += ((-(1.0 / 3.0) * u[i_node]) * vereb) * skzd[i_loc, i_cell] + skz[i_loc, i_cell] * (vereb * (-(1.0 / 3.0) * ud[i_node]) + (-(1.0 / 3.0) * u[i_node]) * verebd)
        CUDA.@atomic cb[3, i_node] += skz[i_loc, i_cell] * ((-(1.0 / 3.0) * u[i_node]) * vereb)
        skzbd[i_loc, i_cell] = skzbd[i_loc, i_cell] + (((-(1.0 / 3.0) * u[i_node]) * vereb) * cd[3, i_node] + c[3, i_node] * (vereb * (-(1.0 / 3.0) * ud[i_node]) + (-(1.0 / 3.0) * u[i_node]) * verebd))
        skzb[i_loc, i_cell] = skzb[i_loc, i_cell] + c[3, i_node] * ((-(1.0 / 3.0) * u[i_node]) * vereb)
        verebd = 0.0
        vereb = 0.0
    end
    for i_loc = 4:-1:1
        i_node = i_cell_to_node[i_loc, i_cell]
        cavgzd = cavgz_stack_d[(div(i_ncell - 1, 1) + 1) + (((i_cell - 1) * (div(4 - 1, 1) + 1) + (i_loc - 1)) + 1)]
        cavgz = cavgz_stack[(div(i_ncell - 1, 1) + 1) + (((i_cell - 1) * (div(4 - 1, 1) + 1) + (i_loc - 1)) + 1)]
        CUDA.@atomic cbd[3, i_node] += 0.25cavgzbd
        CUDA.@atomic cb[3, i_node] += 0.25cavgzb
        cavgyd = cavgy_stack_d[(div(i_ncell - 1, 1) + 1) + (((i_cell - 1) * (div(4 - 1, 1) + 1) + (i_loc - 1)) + 1)]
        cavgy = cavgy_stack[(div(i_ncell - 1, 1) + 1) + (((i_cell - 1) * (div(4 - 1, 1) + 1) + (i_loc - 1)) + 1)]
        CUDA.@atomic cbd[2, i_node] += 0.25cavgybd
        CUDA.@atomic cb[2, i_node] += 0.25cavgyb
        cavgxd = cavgx_stack_d[(div(i_ncell - 1, 1) + 1) + (((i_cell - 1) * (div(4 - 1, 1) + 1) + (i_loc - 1)) + 1)]
        cavgx = cavgx_stack[(div(i_ncell - 1, 1) + 1) + (((i_cell - 1) * (div(4 - 1, 1) + 1) + (i_loc - 1)) + 1)]
        CUDA.@atomic cbd[1, i_node] += 0.25cavgxbd
        CUDA.@atomic cb[1, i_node] += 0.25cavgxb
    end
    cavgzd = cavgz_stack_d[(i_cell - 1) + 1]
    cavgz = cavgz_stack[(i_cell - 1) + 1]
    cavgzbd = 0.0
    cavgzb = 0.0
    cavgyd = cavgy_stack_d[(i_cell - 1) + 1]
    cavgy = cavgy_stack[(i_cell - 1) + 1]
    cavgybd = 0.0
    cavgyb = 0.0
    cavgxd = cavgx_stack_d[(i_cell - 1) + 1]
    cavgx = cavgx_stack[(i_cell - 1) + 1]
    cavgxbd = 0.0
    cavgxb = 0.0
    return nothing
end

function cuda_kernel_ttgc_1!(beta, c, cell_vol, dt, gamma, i_cell_to_node, i_ncell, res, res2, skx, sky, skz, u)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(i_ncell - 1, 1) + 1
        return nothing
    end
    i_cell = 1 + (__tid - 1)
    cavgx = 0.0
    cavgy = 0.0
    cavgz = 0.0
    for i_loc = 1:4
        i_node = i_cell_to_node[i_loc, i_cell]
        cavgx = cavgx + c[1, i_node] / 4
        cavgy = cavgy + c[2, i_node] / 4
        cavgz = cavgz + c[3, i_node] / 4
    end
    for i_loc = 1:4
        i_node = i_cell_to_node[i_loc, i_cell]
        vere = -(1.0 / 3.0) * u[i_node] * (c[1, i_node] * skx[i_loc, i_cell] + c[2, i_node] * sky[i_loc, i_cell] + c[3, i_node] * skz[i_loc, i_cell])
        re = vere / cell_vol[i_cell]
        aerex = cavgx * re
        aerey = cavgy * re
        aerez = cavgz * re
        resi = -(dt / 4) * (0.5 - gamma[i_cell]) * vere
        for i_k = 1:4
            aeresk = aerex * skx[i_k, i_cell] + aerey * sky[i_k, i_cell] + aerez * skz[i_k, i_cell]
            factor = (dt ^ 2 / 3) * aeresk
            auxres = resi + factor * beta[i_cell]
            auxres2 = factor * gamma[i_cell]
            i_k_node = i_cell_to_node[i_k, i_cell]
            CUDA.@atomic res[i_k_node] += auxres
            CUDA.@atomic res2[i_k_node] += auxres2
        end
    end
    return nothing
end

function cuda_kernel_ttgc_2!(i_node_perio, npernode_half, res, resperio)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(npernode_half - 1, 1) + 1
        return nothing
    end
    k = 1 + (__tid - 1)
    i1 = i_node_perio[k, 1]
    i2 = i_node_perio[k, 2]
    resperio[k, 1] = res[i1]
    resperio[k, 2] = res[i2]
    return nothing
end

function cuda_kernel_ttgc_3!(i_node_perio, npernode_half, res, resperio)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(npernode_half - 1, 1) + 1
        return nothing
    end
    k = 1 + (__tid - 1)
    i1 = i_node_perio[k, 1]
    i2 = i_node_perio[k, 2]
    CUDA.@atomic res[i1] += resperio[k, 2]
    CUDA.@atomic res[i2] += resperio[k, 1]
    return nothing
end

function cuda_kernel_ttgc_4!(i_nnode, node_vol, res, up)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(i_nnode - 1, 1) + 1
        return nothing
    end
    i_node = 1 + (__tid - 1)
    up[i_node] = res[i_node] / node_vol[i_node]
    return nothing
end

function cuda_kernel_ttgc_5!(i_nnode, mup)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(i_nnode - 1, 1) + 1
        return nothing
    end
    i_node = 1 + (__tid - 1)
    mup[i_node] = 0.0
    return nothing
end

function cuda_kernel_ttgc_6!(cell_vol, i_cell_to_node, i_ncell, mup, up)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(i_ncell - 1, 1) + 1
        return nothing
    end
    i_cell = 1 + (__tid - 1)
    i_node1 = i_cell_to_node[1, i_cell]
    i_node2 = i_cell_to_node[2, i_cell]
    i_node3 = i_cell_to_node[3, i_cell]
    i_node4 = i_cell_to_node[4, i_cell]
    auxu = up[i_node1] + up[i_node2] + up[i_node3] + up[i_node4]
    CUDA.@atomic mup[i_node1] += ((auxu + up[i_node1]) * cell_vol[i_cell]) / 20
    CUDA.@atomic mup[i_node2] += ((auxu + up[i_node2]) * cell_vol[i_cell]) / 20
    CUDA.@atomic mup[i_node3] += ((auxu + up[i_node3]) * cell_vol[i_cell]) / 20
    CUDA.@atomic mup[i_node4] += ((auxu + up[i_node4]) * cell_vol[i_cell]) / 20
    return nothing
end

function cuda_kernel_ttgc_7!(i_node_perio, mup, npernode_half, resperio)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(npernode_half - 1, 1) + 1
        return nothing
    end
    k = 1 + (__tid - 1)
    i1 = i_node_perio[k, 1]
    i2 = i_node_perio[k, 2]
    resperio[k, 1] = mup[i1]
    resperio[k, 2] = mup[i2]
    return nothing
end

function cuda_kernel_ttgc_8!(i_node_perio, mup, npernode_half, resperio)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(npernode_half - 1, 1) + 1
        return nothing
    end
    k = 1 + (__tid - 1)
    i1 = i_node_perio[k, 1]
    i2 = i_node_perio[k, 2]
    CUDA.@atomic mup[i1] += resperio[k, 2]
    CUDA.@atomic mup[i2] += resperio[k, 1]
    return nothing
end

function cuda_kernel_ttgc_9!(i_nnode, mup, node_vol, res, up)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(i_nnode - 1, 1) + 1
        return nothing
    end
    i_node = 1 + (__tid - 1)
    up[i_node] = up[i_node] + (res[i_node] - mup[i_node]) / node_vol[i_node]
    return nothing
end

function cuda_kernel_ttgc_10!(c, dt, i_cell_to_node, i_ncell, res2, skx, sky, u, up)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(i_ncell - 1, 1) + 1
        return nothing
    end
    i_cell = 1 + (__tid - 1)
    for i_loc = 1:4
        i_node = i_cell_to_node[i_loc, i_cell]
        vere = -(1.0 / 3.0) * (u[i_node] + up[i_node]) * (c[1, i_node] * skx[i_loc, i_cell] + c[2, i_node] * sky[i_loc, i_cell])
        resi = -(dt / 4) * vere
        for i_k = 1:4
            i_k_node = i_cell_to_node[i_k, i_cell]
            CUDA.@atomic res2[i_k_node] += resi
        end
    end
    return nothing
end

function cuda_kernel_ttgc_11!(i_node_perio, npernode_half, res2, resperio)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(npernode_half - 1, 1) + 1
        return nothing
    end
    k = 1 + (__tid - 1)
    i1 = i_node_perio[k, 1]
    i2 = i_node_perio[k, 2]
    resperio[k, 1] = res2[i1]
    resperio[k, 2] = res2[i2]
    return nothing
end

function cuda_kernel_ttgc_12!(i_node_perio, npernode_half, res2, resperio)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(npernode_half - 1, 1) + 1
        return nothing
    end
    k = 1 + (__tid - 1)
    i1 = i_node_perio[k, 1]
    i2 = i_node_perio[k, 2]
    CUDA.@atomic res2[i1] += resperio[k, 2]
    CUDA.@atomic res2[i2] += resperio[k, 1]
    return nothing
end

function cuda_kernel_ttgc_13!(i_nnode, node_vol, res2, up)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(i_nnode - 1, 1) + 1
        return nothing
    end
    i_node = 1 + (__tid - 1)
    up[i_node] = res2[i_node] / node_vol[i_node]
    return nothing
end

function cuda_kernel_ttgc_14!(i_nnode, mup)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(i_nnode - 1, 1) + 1
        return nothing
    end
    i_node = 1 + (__tid - 1)
    mup[i_node] = 0.0
    return nothing
end

function cuda_kernel_ttgc_15!(cell_vol, i_cell_to_node, i_ncell, mup, up)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(i_ncell - 1, 1) + 1
        return nothing
    end
    i_cell = 1 + (__tid - 1)
    i_node1 = i_cell_to_node[1, i_cell]
    i_node2 = i_cell_to_node[2, i_cell]
    i_node3 = i_cell_to_node[3, i_cell]
    i_node4 = i_cell_to_node[4, i_cell]
    auxu = up[i_node1] + up[i_node2] + up[i_node3] + up[i_node4]
    CUDA.@atomic mup[i_node1] += ((auxu + up[i_node1]) * cell_vol[i_cell]) / 20
    CUDA.@atomic mup[i_node2] += ((auxu + up[i_node2]) * cell_vol[i_cell]) / 20
    CUDA.@atomic mup[i_node3] += ((auxu + up[i_node3]) * cell_vol[i_cell]) / 20
    CUDA.@atomic mup[i_node4] += ((auxu + up[i_node4]) * cell_vol[i_cell]) / 20
    return nothing
end

function cuda_kernel_ttgc_16!(i_node_perio, mup, npernode_half, resperio)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(npernode_half - 1, 1) + 1
        return nothing
    end
    k = 1 + (__tid - 1)
    i1 = i_node_perio[k, 1]
    i2 = i_node_perio[k, 2]
    resperio[k, 1] = mup[i1]
    resperio[k, 2] = mup[i2]
    return nothing
end

function cuda_kernel_ttgc_17!(i_node_perio, mup, npernode_half, resperio)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(npernode_half - 1, 1) + 1
        return nothing
    end
    k = 1 + (__tid - 1)
    i1 = i_node_perio[k, 1]
    i2 = i_node_perio[k, 2]
    CUDA.@atomic mup[i1] += resperio[k, 2]
    CUDA.@atomic mup[i2] += resperio[k, 1]
    return nothing
end

function cuda_kernel_ttgc_18!(i_nnode, mup, node_vol, res2, up)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(i_nnode - 1, 1) + 1
        return nothing
    end
    i_node = 1 + (__tid - 1)
    up[i_node] = up[i_node] + (res2[i_node] - mup[i_node]) / node_vol[i_node]
    return nothing
end

function initstacks_ttgc_b_cuda(i_ncell, i_njac, i_nnode, npernode_half)
    cavgx_stack = CuArray{Float64}(undef, (((div(i_ncell - 1, 1) + 1) + (div(i_ncell - 1, 1) + 1) * (div(4 - 1, 1) + 1)) + (div(i_ncell - 1, 1) + 1)) + 1)
    cavgy_stack = CuArray{Float64}(undef, (((div(i_ncell - 1, 1) + 1) + (div(i_ncell - 1, 1) + 1) * (div(4 - 1, 1) + 1)) + (div(i_ncell - 1, 1) + 1)) + 1)
    cavgz_stack = CuArray{Float64}(undef, (((div(i_ncell - 1, 1) + 1) + (div(i_ncell - 1, 1) + 1) * (div(4 - 1, 1) + 1)) + (div(i_ncell - 1, 1) + 1)) + 1)
    vere_stack = CuArray{Float64}(undef, ((((div(i_ncell - 1, 1) + 1) * (div(4 - 1, 1) + 1) + (div(i_ncell - 1, 1) + 1)) + (div(i_ncell - 1, 1) + 1) * (div(4 - 1, 1) + 1)) + (div(i_ncell - 1, 1) + 1)) + 1)
    re_stack = CuArray{Float64}(undef, ((div(i_ncell - 1, 1) + 1) * (div(4 - 1, 1) + 1) + (div(i_ncell - 1, 1) + 1)) + 1)
    aerex_stack = CuArray{Float64}(undef, ((div(i_ncell - 1, 1) + 1) * (div(4 - 1, 1) + 1) + (div(i_ncell - 1, 1) + 1)) + 1)
    aerey_stack = CuArray{Float64}(undef, ((div(i_ncell - 1, 1) + 1) * (div(4 - 1, 1) + 1) + (div(i_ncell - 1, 1) + 1)) + 1)
    aerez_stack = CuArray{Float64}(undef, ((div(i_ncell - 1, 1) + 1) * (div(4 - 1, 1) + 1) + (div(i_ncell - 1, 1) + 1)) + 1)
    aeresk_stack = CuArray{Float64}(undef, (((div(i_ncell - 1, 1) + 1) * (div(4 - 1, 1) + 1) * (div(4 - 1, 1) + 1) + (div(i_ncell - 1, 1) + 1) * (div(4 - 1, 1) + 1)) + (div(i_ncell - 1, 1) + 1)) + 1)
    factor_stack = CuArray{Float64}(undef, (((div(i_ncell - 1, 1) + 1) * (div(4 - 1, 1) + 1) * (div(4 - 1, 1) + 1) + (div(i_ncell - 1, 1) + 1) * (div(4 - 1, 1) + 1)) + (div(i_ncell - 1, 1) + 1)) + 1)
    mup_stack = CuArray{Float64}(undef, (((((((((((((div(i_njac - 1, 1) + 1) * (div(i_nnode - 1, 1) + 1) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(npernode_half - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(npernode_half - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_nnode - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(npernode_half - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(npernode_half - 1, 1) + 1))
    auxu_stack = CuArray{Float64}(undef, ((((div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1) + (div(i_njac - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1)) + 1)
    up_stack = CuArray{Float64}(undef, ((div(i_njac - 1, 1) + 1) * (div(i_nnode - 1, 1) + 1) + (div(i_nnode - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_nnode - 1, 1) + 1))
    return (cavgx_stack, cavgy_stack, cavgz_stack, vere_stack, re_stack, aerex_stack, aerey_stack, aerez_stack, aeresk_stack, factor_stack, mup_stack, auxu_stack, up_stack)
end

function ttgc_hv_cuda(u, ub, uref, urefb, i_cell_to_node, cell_vol, cell_volb, node_vol, node_volb, skx, skxb, sky, skyb, skz, skzb, i_ncell, i_nnode, c, cb, dt, dtb, beta, betab, gamma, gammab, i_njac, res, resb, res2, res2b, up, upb, mup, mupb, npernode_half, resperio, resperiob, i_node_perio, loss, lossb, ud, ubd, urefd, urefbd, cell_vold, cell_volbd, node_vold, node_volbd, skxd, skxbd, skyd, skybd, skzd, skzbd, cd, cbd, dtd, dtbd, betad, betabd, gammad, gammabd, resd, resbd, res2d, res2bd, upd, upbd, mupd, mupbd, resperiod, resperiobd, lossd, lossbd, cavgx_stack, cavgy_stack, cavgz_stack, vere_stack, re_stack, aerex_stack, aerey_stack, aerez_stack, aeresk_stack, factor_stack, mup_stack, auxu_stack, up_stack)
    dtb = CuArray([dtb])
    dtbd = CuArray([dtbd])
    nthread_per_block = 256
    cavgx_stack_d = CuArray{Float64}(undef, (((div(i_ncell - 1, 1) + 1) + (div(i_ncell - 1, 1) + 1) * (div(4 - 1, 1) + 1)) + (div(i_ncell - 1, 1) + 1)) + 1)
    cavgy_stack_d = CuArray{Float64}(undef, (((div(i_ncell - 1, 1) + 1) + (div(i_ncell - 1, 1) + 1) * (div(4 - 1, 1) + 1)) + (div(i_ncell - 1, 1) + 1)) + 1)
    cavgz_stack_d = CuArray{Float64}(undef, (((div(i_ncell - 1, 1) + 1) + (div(i_ncell - 1, 1) + 1) * (div(4 - 1, 1) + 1)) + (div(i_ncell - 1, 1) + 1)) + 1)
    vere_stack_d = CuArray{Float64}(undef, ((((div(i_ncell - 1, 1) + 1) * (div(4 - 1, 1) + 1) + (div(i_ncell - 1, 1) + 1)) + (div(i_ncell - 1, 1) + 1) * (div(4 - 1, 1) + 1)) + (div(i_ncell - 1, 1) + 1)) + 1)
    re_stack_d = CuArray{Float64}(undef, ((div(i_ncell - 1, 1) + 1) * (div(4 - 1, 1) + 1) + (div(i_ncell - 1, 1) + 1)) + 1)
    aerex_stack_d = CuArray{Float64}(undef, ((div(i_ncell - 1, 1) + 1) * (div(4 - 1, 1) + 1) + (div(i_ncell - 1, 1) + 1)) + 1)
    aerey_stack_d = CuArray{Float64}(undef, ((div(i_ncell - 1, 1) + 1) * (div(4 - 1, 1) + 1) + (div(i_ncell - 1, 1) + 1)) + 1)
    aerez_stack_d = CuArray{Float64}(undef, ((div(i_ncell - 1, 1) + 1) * (div(4 - 1, 1) + 1) + (div(i_ncell - 1, 1) + 1)) + 1)
    aeresk_stack_d = CuArray{Float64}(undef, (((div(i_ncell - 1, 1) + 1) * (div(4 - 1, 1) + 1) * (div(4 - 1, 1) + 1) + (div(i_ncell - 1, 1) + 1) * (div(4 - 1, 1) + 1)) + (div(i_ncell - 1, 1) + 1)) + 1)
    factor_stack_d = CuArray{Float64}(undef, (((div(i_ncell - 1, 1) + 1) * (div(4 - 1, 1) + 1) * (div(4 - 1, 1) + 1) + (div(i_ncell - 1, 1) + 1) * (div(4 - 1, 1) + 1)) + (div(i_ncell - 1, 1) + 1)) + 1)
    mup_stack_d = CuArray{Float64}(undef, (((((((((((((div(i_njac - 1, 1) + 1) * (div(i_nnode - 1, 1) + 1) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(npernode_half - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(npernode_half - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_nnode - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(npernode_half - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(npernode_half - 1, 1) + 1))
    auxu_stack_d = CuArray{Float64}(undef, ((((div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1) + (div(i_njac - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1)) + 1)
    up_stack_d = CuArray{Float64}(undef, ((div(i_njac - 1, 1) + 1) * (div(i_nnode - 1, 1) + 1) + (div(i_nnode - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_nnode - 1, 1) + 1))
    aeresk = 0.0
    aerex = 0.0
    aerey = 0.0
    aerez = 0.0
    auxres = 0.0
    auxres2 = 0.0
    auxu = 0.0
    cavgx = 0.0
    cavgy = 0.0
    cavgz = 0.0
    factor = 0.0
    re = 0.0
    resi = 0.0
    vere = 0.0
    aereskb = 0.0
    aerexb = 0.0
    aereyb = 0.0
    aerezb = 0.0
    auxresb = 0.0
    auxres2b = 0.0
    auxub = 0.0
    cavgxb = 0.0
    cavgyb = 0.0
    cavgzb = 0.0
    factorb = 0.0
    reb = 0.0
    resib = 0.0
    vereb = 0.0
    aereskd = 0.0
    aereskbd = 0.0
    aerexd = 0.0
    aerexbd = 0.0
    aereyd = 0.0
    aereybd = 0.0
    aerezd = 0.0
    aerezbd = 0.0
    auxresd = 0.0
    auxresbd = 0.0
    auxres2d = 0.0
    auxres2bd = 0.0
    auxud = 0.0
    auxubd = 0.0
    cavgxd = 0.0
    cavgxbd = 0.0
    cavgyd = 0.0
    cavgybd = 0.0
    cavgzd = 0.0
    cavgzbd = 0.0
    factord = 0.0
    factorbd = 0.0
    red = 0.0
    rebd = 0.0
    resid = 0.0
    resibd = 0.0
    vered = 0.0
    verebd = 0.0
    @cuda threads = nthread_per_block blocks = cld(div(i_ncell - 1, 1) + 1, nthread_per_block) cuda_kernel_ttgc_hv_1!(aeresk_stack, aeresk_stack_d, aerex_stack, aerex_stack_d, aerey_stack, aerey_stack_d, aerez_stack, aerez_stack_d, beta, betad, c, cavgx_stack, cavgx_stack_d, cavgy_stack, cavgy_stack_d, cavgz_stack, cavgz_stack_d, cd, cell_vol, cell_vold, dt, dtd, factor_stack, factor_stack_d, gamma, gammad, i_cell_to_node, i_ncell, re_stack, re_stack_d, res, res2, res2d, resd, skx, skxd, sky, skyd, skz, skzd, u, ud, vere_stack, vere_stack_d)
    @cuda threads = nthread_per_block blocks = cld(div(npernode_half - 1, 1) + 1, nthread_per_block) cuda_kernel_ttgc_hv_2!(i_node_perio, npernode_half, res, resd, resperio, resperiod)
    @cuda threads = nthread_per_block blocks = cld(div(npernode_half - 1, 1) + 1, nthread_per_block) cuda_kernel_ttgc_hv_3!(i_node_perio, npernode_half, res, resd, resperio, resperiod)
    @cuda threads = nthread_per_block blocks = cld(div(i_nnode - 1, 1) + 1, nthread_per_block) cuda_kernel_ttgc_hv_4!(i_nnode, node_vol, node_vold, res, resd, up, upd)
    for i_seq_ = 1:i_njac
        @cuda threads = nthread_per_block blocks = cld(div(i_nnode - 1, 1) + 1, nthread_per_block) cuda_kernel_ttgc_hv_5!(i_nnode, i_seq_, mup, mup_stack, mup_stack_d, mupd)
        @cuda threads = nthread_per_block blocks = cld(div(i_ncell - 1, 1) + 1, nthread_per_block) cuda_kernel_ttgc_hv_6!(auxu_stack, auxu_stack_d, cell_vol, cell_vold, i_cell_to_node, i_ncell, i_njac, i_nnode, i_seq_, mup, mup_stack, mup_stack_d, mupd, up, upd)
        @cuda threads = nthread_per_block blocks = cld(div(npernode_half - 1, 1) + 1, nthread_per_block) cuda_kernel_ttgc_hv_7!(i_node_perio, mup, mupd, npernode_half, resperio, resperiod)
        @cuda threads = nthread_per_block blocks = cld(div(npernode_half - 1, 1) + 1, nthread_per_block) cuda_kernel_ttgc_hv_8!(i_ncell, i_njac, i_nnode, i_node_perio, i_seq_, mup, mup_stack, mup_stack_d, mupd, npernode_half, resperio, resperiod)
        @cuda threads = nthread_per_block blocks = cld(div(i_nnode - 1, 1) + 1, nthread_per_block) cuda_kernel_ttgc_hv_9!(i_nnode, i_seq_, mup, mupd, node_vol, node_vold, res, resd, up, up_stack, up_stack_d, upd)
        CUDA.@allowscalar begin
                auxu_stack_d[(div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1) + ((i_seq_ - 1) + 1)] = auxud
                auxu_stack[(div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1) + ((i_seq_ - 1) + 1)] = auxu
            end
    end
    @cuda threads = nthread_per_block blocks = cld(div(i_ncell - 1, 1) + 1, nthread_per_block) cuda_kernel_ttgc_hv_10!(c, cd, dt, dtd, i_cell_to_node, i_ncell, res2, res2d, skx, skxd, sky, skyd, u, ud, up, upd, vere_stack, vere_stack_d)
    @cuda threads = nthread_per_block blocks = cld(div(npernode_half - 1, 1) + 1, nthread_per_block) cuda_kernel_ttgc_hv_11!(i_node_perio, npernode_half, res2, res2d, resperio, resperiod)
    @cuda threads = nthread_per_block blocks = cld(div(npernode_half - 1, 1) + 1, nthread_per_block) cuda_kernel_ttgc_hv_12!(i_node_perio, npernode_half, res2, res2d, resperio, resperiod)
    @cuda threads = nthread_per_block blocks = cld(div(i_nnode - 1, 1) + 1, nthread_per_block) cuda_kernel_ttgc_hv_13!(i_njac, i_nnode, node_vol, node_vold, res2, res2d, up, up_stack, up_stack_d, upd)
    for i_seq_ = 1:i_njac
        @cuda threads = nthread_per_block blocks = cld(div(i_nnode - 1, 1) + 1, nthread_per_block) cuda_kernel_ttgc_hv_14!(i_ncell, i_njac, i_nnode, i_seq_, mup, mup_stack, mup_stack_d, mupd, npernode_half)
        @cuda threads = nthread_per_block blocks = cld(div(i_ncell - 1, 1) + 1, nthread_per_block) cuda_kernel_ttgc_hv_15!(auxu_stack, auxu_stack_d, cell_vol, cell_vold, i_cell_to_node, i_ncell, i_njac, i_nnode, i_seq_, mup, mup_stack, mup_stack_d, mupd, npernode_half, up, upd)
        @cuda threads = nthread_per_block blocks = cld(div(npernode_half - 1, 1) + 1, nthread_per_block) cuda_kernel_ttgc_hv_16!(i_node_perio, mup, mupd, npernode_half, resperio, resperiod)
        @cuda threads = nthread_per_block blocks = cld(div(npernode_half - 1, 1) + 1, nthread_per_block) cuda_kernel_ttgc_hv_17!(i_ncell, i_njac, i_nnode, i_node_perio, i_seq_, mup, mup_stack, mup_stack_d, mupd, npernode_half, resperio, resperiod)
        @cuda threads = nthread_per_block blocks = cld(div(i_nnode - 1, 1) + 1, nthread_per_block) cuda_kernel_ttgc_hv_18!(i_njac, i_nnode, i_seq_, mup, mupd, node_vol, node_vold, res2, res2d, up, up_stack, up_stack_d, upd)
        CUDA.@allowscalar begin
                auxu_stack_d[(((div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1) + (div(i_njac - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + ((i_seq_ - 1) + 1)] = auxud
                auxu_stack[(((div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1) + (div(i_njac - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + ((i_seq_ - 1) + 1)] = auxu
            end
    end
    @cuda threads = nthread_per_block blocks = cld(div(i_nnode - 1, 1) + 1, nthread_per_block) cuda_kernel_ttgc_hv_19!(i_nnode, loss, lossd, u, ud, up, upd, uref, urefd)
    CUDA.@allowscalar begin
            aeresk_stack_d[(((div(i_ncell - 1, 1) + 1) * (div(4 - 1, 1) + 1) * (div(4 - 1, 1) + 1) + (div(i_ncell - 1, 1) + 1) * (div(4 - 1, 1) + 1)) + (div(i_ncell - 1, 1) + 1)) + 1] = aereskd
            aeresk_stack[(((div(i_ncell - 1, 1) + 1) * (div(4 - 1, 1) + 1) * (div(4 - 1, 1) + 1) + (div(i_ncell - 1, 1) + 1) * (div(4 - 1, 1) + 1)) + (div(i_ncell - 1, 1) + 1)) + 1] = aeresk
            aerex_stack_d[((div(i_ncell - 1, 1) + 1) * (div(4 - 1, 1) + 1) + (div(i_ncell - 1, 1) + 1)) + 1] = aerexd
            aerex_stack[((div(i_ncell - 1, 1) + 1) * (div(4 - 1, 1) + 1) + (div(i_ncell - 1, 1) + 1)) + 1] = aerex
            aerey_stack_d[((div(i_ncell - 1, 1) + 1) * (div(4 - 1, 1) + 1) + (div(i_ncell - 1, 1) + 1)) + 1] = aereyd
            aerey_stack[((div(i_ncell - 1, 1) + 1) * (div(4 - 1, 1) + 1) + (div(i_ncell - 1, 1) + 1)) + 1] = aerey
            aerez_stack_d[((div(i_ncell - 1, 1) + 1) * (div(4 - 1, 1) + 1) + (div(i_ncell - 1, 1) + 1)) + 1] = aerezd
            aerez_stack[((div(i_ncell - 1, 1) + 1) * (div(4 - 1, 1) + 1) + (div(i_ncell - 1, 1) + 1)) + 1] = aerez
            auxu_stack_d[((((div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1) + (div(i_njac - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1)) + 1] = auxud
            auxu_stack[((((div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1) + (div(i_njac - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1)) + 1] = auxu
            cavgx_stack_d[(((div(i_ncell - 1, 1) + 1) + (div(i_ncell - 1, 1) + 1) * (div(4 - 1, 1) + 1)) + (div(i_ncell - 1, 1) + 1)) + 1] = cavgxd
            cavgx_stack[(((div(i_ncell - 1, 1) + 1) + (div(i_ncell - 1, 1) + 1) * (div(4 - 1, 1) + 1)) + (div(i_ncell - 1, 1) + 1)) + 1] = cavgx
            cavgy_stack_d[(((div(i_ncell - 1, 1) + 1) + (div(i_ncell - 1, 1) + 1) * (div(4 - 1, 1) + 1)) + (div(i_ncell - 1, 1) + 1)) + 1] = cavgyd
            cavgy_stack[(((div(i_ncell - 1, 1) + 1) + (div(i_ncell - 1, 1) + 1) * (div(4 - 1, 1) + 1)) + (div(i_ncell - 1, 1) + 1)) + 1] = cavgy
            cavgz_stack_d[(((div(i_ncell - 1, 1) + 1) + (div(i_ncell - 1, 1) + 1) * (div(4 - 1, 1) + 1)) + (div(i_ncell - 1, 1) + 1)) + 1] = cavgzd
            cavgz_stack[(((div(i_ncell - 1, 1) + 1) + (div(i_ncell - 1, 1) + 1) * (div(4 - 1, 1) + 1)) + (div(i_ncell - 1, 1) + 1)) + 1] = cavgz
            factor_stack_d[(((div(i_ncell - 1, 1) + 1) * (div(4 - 1, 1) + 1) * (div(4 - 1, 1) + 1) + (div(i_ncell - 1, 1) + 1) * (div(4 - 1, 1) + 1)) + (div(i_ncell - 1, 1) + 1)) + 1] = factord
            factor_stack[(((div(i_ncell - 1, 1) + 1) * (div(4 - 1, 1) + 1) * (div(4 - 1, 1) + 1) + (div(i_ncell - 1, 1) + 1) * (div(4 - 1, 1) + 1)) + (div(i_ncell - 1, 1) + 1)) + 1] = factor
            re_stack_d[((div(i_ncell - 1, 1) + 1) * (div(4 - 1, 1) + 1) + (div(i_ncell - 1, 1) + 1)) + 1] = red
            re_stack[((div(i_ncell - 1, 1) + 1) * (div(4 - 1, 1) + 1) + (div(i_ncell - 1, 1) + 1)) + 1] = re
            vere_stack_d[((((div(i_ncell - 1, 1) + 1) * (div(4 - 1, 1) + 1) + (div(i_ncell - 1, 1) + 1)) + (div(i_ncell - 1, 1) + 1) * (div(4 - 1, 1) + 1)) + (div(i_ncell - 1, 1) + 1)) + 1] = vered
            vere_stack[((((div(i_ncell - 1, 1) + 1) * (div(4 - 1, 1) + 1) + (div(i_ncell - 1, 1) + 1)) + (div(i_ncell - 1, 1) + 1) * (div(4 - 1, 1) + 1)) + (div(i_ncell - 1, 1) + 1)) + 1] = vere
            aereskd = aeresk_stack_d[(((div(i_ncell - 1, 1) + 1) * (div(4 - 1, 1) + 1) * (div(4 - 1, 1) + 1) + (div(i_ncell - 1, 1) + 1) * (div(4 - 1, 1) + 1)) + (div(i_ncell - 1, 1) + 1)) + 1]
            aeresk = aeresk_stack[(((div(i_ncell - 1, 1) + 1) * (div(4 - 1, 1) + 1) * (div(4 - 1, 1) + 1) + (div(i_ncell - 1, 1) + 1) * (div(4 - 1, 1) + 1)) + (div(i_ncell - 1, 1) + 1)) + 1]
            aerexd = aerex_stack_d[((div(i_ncell - 1, 1) + 1) * (div(4 - 1, 1) + 1) + (div(i_ncell - 1, 1) + 1)) + 1]
            aerex = aerex_stack[((div(i_ncell - 1, 1) + 1) * (div(4 - 1, 1) + 1) + (div(i_ncell - 1, 1) + 1)) + 1]
            aereyd = aerey_stack_d[((div(i_ncell - 1, 1) + 1) * (div(4 - 1, 1) + 1) + (div(i_ncell - 1, 1) + 1)) + 1]
            aerey = aerey_stack[((div(i_ncell - 1, 1) + 1) * (div(4 - 1, 1) + 1) + (div(i_ncell - 1, 1) + 1)) + 1]
            aerezd = aerez_stack_d[((div(i_ncell - 1, 1) + 1) * (div(4 - 1, 1) + 1) + (div(i_ncell - 1, 1) + 1)) + 1]
            aerez = aerez_stack[((div(i_ncell - 1, 1) + 1) * (div(4 - 1, 1) + 1) + (div(i_ncell - 1, 1) + 1)) + 1]
            auxud = auxu_stack_d[((((div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1) + (div(i_njac - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1)) + 1]
            auxu = auxu_stack[((((div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1) + (div(i_njac - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1)) + 1]
            cavgxd = cavgx_stack_d[(((div(i_ncell - 1, 1) + 1) + (div(i_ncell - 1, 1) + 1) * (div(4 - 1, 1) + 1)) + (div(i_ncell - 1, 1) + 1)) + 1]
            cavgx = cavgx_stack[(((div(i_ncell - 1, 1) + 1) + (div(i_ncell - 1, 1) + 1) * (div(4 - 1, 1) + 1)) + (div(i_ncell - 1, 1) + 1)) + 1]
            cavgyd = cavgy_stack_d[(((div(i_ncell - 1, 1) + 1) + (div(i_ncell - 1, 1) + 1) * (div(4 - 1, 1) + 1)) + (div(i_ncell - 1, 1) + 1)) + 1]
            cavgy = cavgy_stack[(((div(i_ncell - 1, 1) + 1) + (div(i_ncell - 1, 1) + 1) * (div(4 - 1, 1) + 1)) + (div(i_ncell - 1, 1) + 1)) + 1]
            cavgzd = cavgz_stack_d[(((div(i_ncell - 1, 1) + 1) + (div(i_ncell - 1, 1) + 1) * (div(4 - 1, 1) + 1)) + (div(i_ncell - 1, 1) + 1)) + 1]
            cavgz = cavgz_stack[(((div(i_ncell - 1, 1) + 1) + (div(i_ncell - 1, 1) + 1) * (div(4 - 1, 1) + 1)) + (div(i_ncell - 1, 1) + 1)) + 1]
            factord = factor_stack_d[(((div(i_ncell - 1, 1) + 1) * (div(4 - 1, 1) + 1) * (div(4 - 1, 1) + 1) + (div(i_ncell - 1, 1) + 1) * (div(4 - 1, 1) + 1)) + (div(i_ncell - 1, 1) + 1)) + 1]
            factor = factor_stack[(((div(i_ncell - 1, 1) + 1) * (div(4 - 1, 1) + 1) * (div(4 - 1, 1) + 1) + (div(i_ncell - 1, 1) + 1) * (div(4 - 1, 1) + 1)) + (div(i_ncell - 1, 1) + 1)) + 1]
            red = re_stack_d[((div(i_ncell - 1, 1) + 1) * (div(4 - 1, 1) + 1) + (div(i_ncell - 1, 1) + 1)) + 1]
            re = re_stack[((div(i_ncell - 1, 1) + 1) * (div(4 - 1, 1) + 1) + (div(i_ncell - 1, 1) + 1)) + 1]
            vered = vere_stack_d[((((div(i_ncell - 1, 1) + 1) * (div(4 - 1, 1) + 1) + (div(i_ncell - 1, 1) + 1)) + (div(i_ncell - 1, 1) + 1) * (div(4 - 1, 1) + 1)) + (div(i_ncell - 1, 1) + 1)) + 1]
            vere = vere_stack[((((div(i_ncell - 1, 1) + 1) * (div(4 - 1, 1) + 1) + (div(i_ncell - 1, 1) + 1)) + (div(i_ncell - 1, 1) + 1) * (div(4 - 1, 1) + 1)) + (div(i_ncell - 1, 1) + 1)) + 1]
        end
    @cuda threads = nthread_per_block blocks = cld(div(1 - i_nnode, -1) + 1, nthread_per_block) cuda_kernel_ttgc_hv_20!(i_nnode, lossb, lossbd, u, ub, ubd, ud, up, upb, upbd, upd, uref, urefb, urefbd, urefd)
    for i_seq_ = i_njac:-1:1
        CUDA.@allowscalar begin
                auxud = auxu_stack_d[(((div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1) + (div(i_njac - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + ((i_seq_ - 1) + 1)]
                auxu = auxu_stack[(((div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1) + (div(i_njac - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + ((i_seq_ - 1) + 1)]
            end
        @cuda threads = nthread_per_block blocks = cld(div(1 - i_nnode, -1) + 1, nthread_per_block) cuda_kernel_ttgc_hv_21!(i_njac, i_nnode, i_seq_, mup, mupb, mupbd, mupd, node_vol, node_volb, node_volbd, node_vold, res2, res2b, res2bd, res2d, up, up_stack, up_stack_d, upb, upbd, upd)
        @cuda threads = nthread_per_block blocks = cld(div(1 - npernode_half, -1) + 1, nthread_per_block) cuda_kernel_ttgc_hv_22!(i_ncell, i_njac, i_nnode, i_node_perio, i_seq_, mup, mup_stack, mup_stack_d, mupb, mupbd, mupd, npernode_half, resperiob, resperiobd)
        @cuda threads = nthread_per_block blocks = cld(div(npernode_half - 1, 1) + 1, nthread_per_block) cuda_kernel_ttgc_hv_23!(i_node_perio, mupb, mupbd, npernode_half, resperiob, resperiobd)
        @cuda threads = nthread_per_block blocks = cld(div(1 - i_ncell, -1) + 1, nthread_per_block) cuda_kernel_ttgc_hv_24!(auxu_stack, auxu_stack_d, cell_vol, cell_volb, cell_volbd, cell_vold, i_cell_to_node, i_ncell, i_njac, i_nnode, i_seq_, mup, mup_stack, mup_stack_d, mupb, mupbd, mupd, npernode_half, up, upb, upbd, upd)
        @cuda threads = nthread_per_block blocks = cld(div(1 - i_nnode, -1) + 1, nthread_per_block) cuda_kernel_ttgc_hv_25!(i_ncell, i_njac, i_nnode, i_seq_, mup, mup_stack, mup_stack_d, mupb, mupbd, mupd, npernode_half)
    end
    @cuda threads = nthread_per_block blocks = cld(div(1 - i_nnode, -1) + 1, nthread_per_block) cuda_kernel_ttgc_hv_26!(i_njac, i_nnode, node_vol, node_volb, node_volbd, node_vold, res2, res2b, res2bd, res2d, up, up_stack, up_stack_d, upb, upbd, upd)
    @cuda threads = nthread_per_block blocks = cld(div(npernode_half - 1, 1) + 1, nthread_per_block) cuda_kernel_ttgc_hv_27!(i_node_perio, npernode_half, res2b, res2bd, resperiob, resperiobd)
    @cuda threads = nthread_per_block blocks = cld(div(npernode_half - 1, 1) + 1, nthread_per_block) cuda_kernel_ttgc_hv_28!(i_node_perio, npernode_half, res2b, res2bd, resperiob, resperiobd)
    @cuda threads = nthread_per_block blocks = cld(div(1 - i_ncell, -1) + 1, nthread_per_block) cuda_kernel_ttgc_hv_29!(c, cb, cbd, cd, dt, dtb, dtbd, dtd, i_cell_to_node, i_ncell, res2b, res2bd, skx, skxb, skxbd, skxd, sky, skyb, skybd, skyd, u, ub, ubd, ud, up, upb, upbd, upd, vere_stack, vere_stack_d)
    for i_seq_ = i_njac:-1:1
        CUDA.@allowscalar begin
                auxud = auxu_stack_d[(div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1) + ((i_seq_ - 1) + 1)]
                auxu = auxu_stack[(div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1) + ((i_seq_ - 1) + 1)]
            end
        @cuda threads = nthread_per_block blocks = cld(div(1 - i_nnode, -1) + 1, nthread_per_block) cuda_kernel_ttgc_hv_30!(i_nnode, i_seq_, mup, mupb, mupbd, mupd, node_vol, node_volb, node_volbd, node_vold, res, resb, resbd, resd, up, up_stack, up_stack_d, upb, upbd, upd)
        @cuda threads = nthread_per_block blocks = cld(div(1 - npernode_half, -1) + 1, nthread_per_block) cuda_kernel_ttgc_hv_31!(i_ncell, i_njac, i_nnode, i_node_perio, i_seq_, mup, mup_stack, mup_stack_d, mupb, mupbd, mupd, npernode_half, resperiob, resperiobd)
        @cuda threads = nthread_per_block blocks = cld(div(npernode_half - 1, 1) + 1, nthread_per_block) cuda_kernel_ttgc_hv_32!(i_node_perio, mupb, mupbd, npernode_half, resperiob, resperiobd)
        @cuda threads = nthread_per_block blocks = cld(div(1 - i_ncell, -1) + 1, nthread_per_block) cuda_kernel_ttgc_hv_33!(auxu_stack, auxu_stack_d, cell_vol, cell_volb, cell_volbd, cell_vold, i_cell_to_node, i_ncell, i_njac, i_nnode, i_seq_, mup, mup_stack, mup_stack_d, mupb, mupbd, mupd, up, upb, upbd, upd)
        @cuda threads = nthread_per_block blocks = cld(div(1 - i_nnode, -1) + 1, nthread_per_block) cuda_kernel_ttgc_hv_34!(i_nnode, i_seq_, mup, mup_stack, mup_stack_d, mupb, mupbd, mupd)
    end
    @cuda threads = nthread_per_block blocks = cld(div(i_nnode - 1, 1) + 1, nthread_per_block) cuda_kernel_ttgc_hv_35!(i_nnode, node_vol, node_volb, node_volbd, node_vold, res, resb, resbd, resd, upb, upbd)
    @cuda threads = nthread_per_block blocks = cld(div(npernode_half - 1, 1) + 1, nthread_per_block) cuda_kernel_ttgc_hv_36!(i_node_perio, npernode_half, resb, resbd, resperiob, resperiobd)
    @cuda threads = nthread_per_block blocks = cld(div(npernode_half - 1, 1) + 1, nthread_per_block) cuda_kernel_ttgc_hv_37!(i_node_perio, npernode_half, resb, resbd, resperiob, resperiobd)
    @cuda threads = nthread_per_block blocks = cld(div(1 - i_ncell, -1) + 1, nthread_per_block) cuda_kernel_ttgc_hv_38!(aeresk_stack, aeresk_stack_d, aerex_stack, aerex_stack_d, aerey_stack, aerey_stack_d, aerez_stack, aerez_stack_d, beta, betab, betabd, betad, c, cavgx_stack, cavgx_stack_d, cavgy_stack, cavgy_stack_d, cavgz_stack, cavgz_stack_d, cb, cbd, cd, cell_vol, cell_volb, cell_volbd, cell_vold, dt, dtb, dtbd, dtd, factor_stack, factor_stack_d, gamma, gammab, gammabd, gammad, i_cell_to_node, i_ncell, re_stack, re_stack_d, res2b, res2bd, resb, resbd, skx, skxb, skxbd, skxd, sky, skyb, skybd, skyd, skz, skzb, skzbd, skzd, u, ub, ubd, ud, vere_stack, vere_stack_d)
    dtb = (Array(dtb))[1]
    dtbd = (Array(dtbd))[1]
    return (dtb, dtbd)
end

function ttgc_cuda(u, uref, i_cell_to_node, cell_vol, node_vol, skx, sky, skz, i_ncell, i_nnode, c, dt, beta, gamma, i_njac, res, res2, up, mup, npernode_half, resperio, i_node_perio, loss)
    nthread_per_block = 256
    @cuda threads = nthread_per_block blocks = cld(div(i_ncell - 1, 1) + 1, nthread_per_block) cuda_kernel_ttgc_1!(beta, c, cell_vol, dt, gamma, i_cell_to_node, i_ncell, res, res2, skx, sky, skz, u)
    @cuda threads = nthread_per_block blocks = cld(div(npernode_half - 1, 1) + 1, nthread_per_block) cuda_kernel_ttgc_2!(i_node_perio, npernode_half, res, resperio)
    @cuda threads = nthread_per_block blocks = cld(div(npernode_half - 1, 1) + 1, nthread_per_block) cuda_kernel_ttgc_3!(i_node_perio, npernode_half, res, resperio)
    @cuda threads = nthread_per_block blocks = cld(div(i_nnode - 1, 1) + 1, nthread_per_block) cuda_kernel_ttgc_4!(i_nnode, node_vol, res, up)
    for i_seq_ = 1:i_njac
        @cuda threads = nthread_per_block blocks = cld(div(i_nnode - 1, 1) + 1, nthread_per_block) cuda_kernel_ttgc_5!(i_nnode, mup)
        @cuda threads = nthread_per_block blocks = cld(div(i_ncell - 1, 1) + 1, nthread_per_block) cuda_kernel_ttgc_6!(cell_vol, i_cell_to_node, i_ncell, mup, up)
        @cuda threads = nthread_per_block blocks = cld(div(npernode_half - 1, 1) + 1, nthread_per_block) cuda_kernel_ttgc_7!(i_node_perio, mup, npernode_half, resperio)
        @cuda threads = nthread_per_block blocks = cld(div(npernode_half - 1, 1) + 1, nthread_per_block) cuda_kernel_ttgc_8!(i_node_perio, mup, npernode_half, resperio)
        @cuda threads = nthread_per_block blocks = cld(div(i_nnode - 1, 1) + 1, nthread_per_block) cuda_kernel_ttgc_9!(i_nnode, mup, node_vol, res, up)
    end
    @cuda threads = nthread_per_block blocks = cld(div(i_ncell - 1, 1) + 1, nthread_per_block) cuda_kernel_ttgc_10!(c, dt, i_cell_to_node, i_ncell, res2, skx, sky, u, up)
    @cuda threads = nthread_per_block blocks = cld(div(npernode_half - 1, 1) + 1, nthread_per_block) cuda_kernel_ttgc_11!(i_node_perio, npernode_half, res2, resperio)
    @cuda threads = nthread_per_block blocks = cld(div(npernode_half - 1, 1) + 1, nthread_per_block) cuda_kernel_ttgc_12!(i_node_perio, npernode_half, res2, resperio)
    @cuda threads = nthread_per_block blocks = cld(div(i_nnode - 1, 1) + 1, nthread_per_block) cuda_kernel_ttgc_13!(i_nnode, node_vol, res2, up)
    for i_seq_ = 1:i_njac
        @cuda threads = nthread_per_block blocks = cld(div(i_nnode - 1, 1) + 1, nthread_per_block) cuda_kernel_ttgc_14!(i_nnode, mup)
        @cuda threads = nthread_per_block blocks = cld(div(i_ncell - 1, 1) + 1, nthread_per_block) cuda_kernel_ttgc_15!(cell_vol, i_cell_to_node, i_ncell, mup, up)
        @cuda threads = nthread_per_block blocks = cld(div(npernode_half - 1, 1) + 1, nthread_per_block) cuda_kernel_ttgc_16!(i_node_perio, mup, npernode_half, resperio)
        @cuda threads = nthread_per_block blocks = cld(div(npernode_half - 1, 1) + 1, nthread_per_block) cuda_kernel_ttgc_17!(i_node_perio, mup, npernode_half, resperio)
        @cuda threads = nthread_per_block blocks = cld(div(i_nnode - 1, 1) + 1, nthread_per_block) cuda_kernel_ttgc_18!(i_nnode, mup, node_vol, res2, up)
    end
    CUDA.@allowscalar begin
            loss[1] = loss[1] + mapreduce(((__mr_1, __mr_2, __mr_3)->((__mr_1 + __mr_2) - __mr_3) ^ 2), +, u, up, uref)
        end
    return nothing
end
