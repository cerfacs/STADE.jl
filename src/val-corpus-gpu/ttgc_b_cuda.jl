import Pkg
haskey(Pkg.project().dependencies, "CUDA") || Pkg.add("CUDA")
using CUDA
using LinearAlgebra
CUDA.allowscalar(false)

function cuda_kernel_ttgc_b_1!(aeresk_stack, aerex_stack, aerey_stack, aerez_stack, beta, c, cavgx_stack, cavgy_stack, cavgz_stack, cell_vol, dt, factor_stack, gamma, i_cell_to_node, i_ncell, re_stack, res, res2, skx, sky, skz, u, vere_stack)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(i_ncell - 1, 1) + 1
        return nothing
    end
    i_cell = 1 + (__tid - 1)
    cavgx_stack[(i_cell - 1) + 1] = cavgx
    cavgx = 0.0
    cavgy_stack[(i_cell - 1) + 1] = cavgy
    cavgy = 0.0
    cavgz_stack[(i_cell - 1) + 1] = cavgz
    cavgz = 0.0
    for i_loc = 1:4
        i_node = i_cell_to_node[i_loc, i_cell]
        cavgx_stack[(div(i_ncell - 1, 1) + 1) + (((i_cell - 1) * (div(4 - 1, 1) + 1) + (i_loc - 1)) + 1)] = cavgx
        cavgx = cavgx + c[1, i_node] / 4
        cavgy_stack[(div(i_ncell - 1, 1) + 1) + (((i_cell - 1) * (div(4 - 1, 1) + 1) + (i_loc - 1)) + 1)] = cavgy
        cavgy = cavgy + c[2, i_node] / 4
        cavgz_stack[(div(i_ncell - 1, 1) + 1) + (((i_cell - 1) * (div(4 - 1, 1) + 1) + (i_loc - 1)) + 1)] = cavgz
        cavgz = cavgz + c[3, i_node] / 4
    end
    for i_loc = 1:4
        i_node = i_cell_to_node[i_loc, i_cell]
        vere_stack[((i_cell - 1) * (div(4 - 1, 1) + 1) + (i_loc - 1)) + 1] = vere
        vere = -(1.0 / 3.0) * u[i_node] * (c[1, i_node] * skx[i_loc, i_cell] + c[2, i_node] * sky[i_loc, i_cell] + c[3, i_node] * skz[i_loc, i_cell])
        re_stack[((i_cell - 1) * (div(4 - 1, 1) + 1) + (i_loc - 1)) + 1] = re
        re = vere / cell_vol[i_cell]
        aerex_stack[((i_cell - 1) * (div(4 - 1, 1) + 1) + (i_loc - 1)) + 1] = aerex
        aerex = cavgx * re
        aerey_stack[((i_cell - 1) * (div(4 - 1, 1) + 1) + (i_loc - 1)) + 1] = aerey
        aerey = cavgy * re
        aerez_stack[((i_cell - 1) * (div(4 - 1, 1) + 1) + (i_loc - 1)) + 1] = aerez
        aerez = cavgz * re
        resi = -(dt / 4) * (0.5 - gamma[i_cell]) * vere
        for i_k = 1:4
            aeresk_stack[((i_cell - 1) * ((div(4 - 1, 1) + 1) * (div(4 - 1, 1) + 1)) + (i_loc - 1) * (div(4 - 1, 1) + 1) + (i_k - 1)) + 1] = aeresk
            aeresk = aerex * skx[i_k, i_cell] + aerey * sky[i_k, i_cell] + aerez * skz[i_k, i_cell]
            factor_stack[((i_cell - 1) * ((div(4 - 1, 1) + 1) * (div(4 - 1, 1) + 1)) + (i_loc - 1) * (div(4 - 1, 1) + 1) + (i_k - 1)) + 1] = factor
            factor = (dt ^ 2 / 3) * aeresk
            auxres = resi + factor * beta[i_cell]
            auxres2 = factor * gamma[i_cell]
            i_k_node = i_cell_to_node[i_k, i_cell]
            CUDA.@atomic res[i_k_node] += auxres
            CUDA.@atomic res2[i_k_node] += auxres2
        end
        aeresk_stack[(div(i_ncell - 1, 1) + 1) * (div(4 - 1, 1) + 1) * (div(4 - 1, 1) + 1) + (((i_cell - 1) * (div(4 - 1, 1) + 1) + (i_loc - 1)) + 1)] = aeresk
        factor_stack[(div(i_ncell - 1, 1) + 1) * (div(4 - 1, 1) + 1) * (div(4 - 1, 1) + 1) + (((i_cell - 1) * (div(4 - 1, 1) + 1) + (i_loc - 1)) + 1)] = factor
    end
    aeresk_stack[((div(i_ncell - 1, 1) + 1) * (div(4 - 1, 1) + 1) * (div(4 - 1, 1) + 1) + (div(i_ncell - 1, 1) + 1) * (div(4 - 1, 1) + 1)) + ((i_cell - 1) + 1)] = aeresk
    aerex_stack[(div(i_ncell - 1, 1) + 1) * (div(4 - 1, 1) + 1) + ((i_cell - 1) + 1)] = aerex
    aerey_stack[(div(i_ncell - 1, 1) + 1) * (div(4 - 1, 1) + 1) + ((i_cell - 1) + 1)] = aerey
    aerez_stack[(div(i_ncell - 1, 1) + 1) * (div(4 - 1, 1) + 1) + ((i_cell - 1) + 1)] = aerez
    cavgx_stack[((div(i_ncell - 1, 1) + 1) + (div(i_ncell - 1, 1) + 1) * (div(4 - 1, 1) + 1)) + ((i_cell - 1) + 1)] = cavgx
    cavgy_stack[((div(i_ncell - 1, 1) + 1) + (div(i_ncell - 1, 1) + 1) * (div(4 - 1, 1) + 1)) + ((i_cell - 1) + 1)] = cavgy
    cavgz_stack[((div(i_ncell - 1, 1) + 1) + (div(i_ncell - 1, 1) + 1) * (div(4 - 1, 1) + 1)) + ((i_cell - 1) + 1)] = cavgz
    factor_stack[((div(i_ncell - 1, 1) + 1) * (div(4 - 1, 1) + 1) * (div(4 - 1, 1) + 1) + (div(i_ncell - 1, 1) + 1) * (div(4 - 1, 1) + 1)) + ((i_cell - 1) + 1)] = factor
    re_stack[(div(i_ncell - 1, 1) + 1) * (div(4 - 1, 1) + 1) + ((i_cell - 1) + 1)] = re
    vere_stack[(div(i_ncell - 1, 1) + 1) * (div(4 - 1, 1) + 1) + ((i_cell - 1) + 1)] = vere
    return nothing
end

function cuda_kernel_ttgc_b_2!(i_node_perio, npernode_half, res, resperio)
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

function cuda_kernel_ttgc_b_3!(i_node_perio, npernode_half, res, resperio)
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

function cuda_kernel_ttgc_b_4!(i_nnode, node_vol, res, up)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(i_nnode - 1, 1) + 1
        return nothing
    end
    i_node = 1 + (__tid - 1)
    up[i_node] = res[i_node] / node_vol[i_node]
    return nothing
end

function cuda_kernel_ttgc_b_5!(i_nnode, i_seq_, mup, mup_stack)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(i_nnode - 1, 1) + 1
        return nothing
    end
    i_node = 1 + (__tid - 1)
    mup_stack[((i_seq_ - 1) * (div(i_nnode - 1, 1) + 1) + (i_node - 1)) + 1] = mup[i_node]
    mup[i_node] = 0.0
    return nothing
end

function cuda_kernel_ttgc_b_6!(auxu_stack, cell_vol, i_cell_to_node, i_ncell, i_njac, i_nnode, i_seq_, mup, mup_stack, up)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(i_ncell - 1, 1) + 1
        return nothing
    end
    i_cell = 1 + (__tid - 1)
    i_node1 = i_cell_to_node[1, i_cell]
    i_node2 = i_cell_to_node[2, i_cell]
    i_node3 = i_cell_to_node[3, i_cell]
    i_node4 = i_cell_to_node[4, i_cell]
    auxu_stack[((i_seq_ - 1) * (div(i_ncell - 1, 1) + 1) + (i_cell - 1)) + 1] = auxu
    auxu = up[i_node1] + up[i_node2] + up[i_node3] + up[i_node4]
    mup_stack[(div(i_njac - 1, 1) + 1) * (div(i_nnode - 1, 1) + 1) + (((i_seq_ - 1) * (div(i_ncell - 1, 1) + 1) + (i_cell - 1)) + 1)] = mup[i_node1]
    CUDA.@atomic mup[i_node1] += ((auxu + up[i_node1]) * cell_vol[i_cell]) / 20
    mup_stack[((div(i_njac - 1, 1) + 1) * (div(i_nnode - 1, 1) + 1) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (((i_seq_ - 1) * (div(i_ncell - 1, 1) + 1) + (i_cell - 1)) + 1)] = mup[i_node2]
    CUDA.@atomic mup[i_node2] += ((auxu + up[i_node2]) * cell_vol[i_cell]) / 20
    mup_stack[(((div(i_njac - 1, 1) + 1) * (div(i_nnode - 1, 1) + 1) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (((i_seq_ - 1) * (div(i_ncell - 1, 1) + 1) + (i_cell - 1)) + 1)] = mup[i_node3]
    CUDA.@atomic mup[i_node3] += ((auxu + up[i_node3]) * cell_vol[i_cell]) / 20
    mup_stack[((((div(i_njac - 1, 1) + 1) * (div(i_nnode - 1, 1) + 1) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (((i_seq_ - 1) * (div(i_ncell - 1, 1) + 1) + (i_cell - 1)) + 1)] = mup[i_node4]
    CUDA.@atomic mup[i_node4] += ((auxu + up[i_node4]) * cell_vol[i_cell]) / 20
    return nothing
end

function cuda_kernel_ttgc_b_7!(i_node_perio, mup, npernode_half, resperio)
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

function cuda_kernel_ttgc_b_8!(i_ncell, i_njac, i_nnode, i_node_perio, i_seq_, mup, mup_stack, npernode_half, resperio)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(npernode_half - 1, 1) + 1
        return nothing
    end
    k = 1 + (__tid - 1)
    i1 = i_node_perio[k, 1]
    i2 = i_node_perio[k, 2]
    mup_stack[(((((div(i_njac - 1, 1) + 1) * (div(i_nnode - 1, 1) + 1) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (((i_seq_ - 1) * (div(npernode_half - 1, 1) + 1) + (k - 1)) + 1)] = mup[i1]
    CUDA.@atomic mup[i1] += resperio[k, 2]
    mup_stack[((((((div(i_njac - 1, 1) + 1) * (div(i_nnode - 1, 1) + 1) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(npernode_half - 1, 1) + 1)) + (((i_seq_ - 1) * (div(npernode_half - 1, 1) + 1) + (k - 1)) + 1)] = mup[i2]
    CUDA.@atomic mup[i2] += resperio[k, 1]
    return nothing
end

function cuda_kernel_ttgc_b_9!(i_nnode, i_seq_, mup, node_vol, res, up, up_stack)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(i_nnode - 1, 1) + 1
        return nothing
    end
    i_node = 1 + (__tid - 1)
    up_stack[((i_seq_ - 1) * (div(i_nnode - 1, 1) + 1) + (i_node - 1)) + 1] = up[i_node]
    up[i_node] = up[i_node] + (res[i_node] - mup[i_node]) / node_vol[i_node]
    return nothing
end

function cuda_kernel_ttgc_b_10!(c, dt, i_cell_to_node, i_ncell, res2, skx, sky, u, up, vere_stack)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(i_ncell - 1, 1) + 1
        return nothing
    end
    i_cell = 1 + (__tid - 1)
    for i_loc = 1:4
        i_node = i_cell_to_node[i_loc, i_cell]
        vere_stack[((div(i_ncell - 1, 1) + 1) * (div(4 - 1, 1) + 1) + (div(i_ncell - 1, 1) + 1)) + (((i_cell - 1) * (div(4 - 1, 1) + 1) + (i_loc - 1)) + 1)] = vere
        vere = -(1.0 / 3.0) * (u[i_node] + up[i_node]) * (c[1, i_node] * skx[i_loc, i_cell] + c[2, i_node] * sky[i_loc, i_cell])
        resi = -(dt / 4) * vere
        for i_k = 1:4
            i_k_node = i_cell_to_node[i_k, i_cell]
            CUDA.@atomic res2[i_k_node] += resi
        end
    end
    vere_stack[(((div(i_ncell - 1, 1) + 1) * (div(4 - 1, 1) + 1) + (div(i_ncell - 1, 1) + 1)) + (div(i_ncell - 1, 1) + 1) * (div(4 - 1, 1) + 1)) + ((i_cell - 1) + 1)] = vere
    return nothing
end

function cuda_kernel_ttgc_b_11!(i_node_perio, npernode_half, res2, resperio)
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

function cuda_kernel_ttgc_b_12!(i_node_perio, npernode_half, res2, resperio)
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

function cuda_kernel_ttgc_b_13!(i_njac, i_nnode, node_vol, res2, up, up_stack)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(i_nnode - 1, 1) + 1
        return nothing
    end
    i_node = 1 + (__tid - 1)
    up_stack[(div(i_njac - 1, 1) + 1) * (div(i_nnode - 1, 1) + 1) + ((i_node - 1) + 1)] = up[i_node]
    up[i_node] = res2[i_node] / node_vol[i_node]
    return nothing
end

function cuda_kernel_ttgc_b_14!(i_ncell, i_njac, i_nnode, i_seq_, mup, mup_stack, npernode_half)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(i_nnode - 1, 1) + 1
        return nothing
    end
    i_node = 1 + (__tid - 1)
    mup_stack[(((((((div(i_njac - 1, 1) + 1) * (div(i_nnode - 1, 1) + 1) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(npernode_half - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(npernode_half - 1, 1) + 1)) + (((i_seq_ - 1) * (div(i_nnode - 1, 1) + 1) + (i_node - 1)) + 1)] = mup[i_node]
    mup[i_node] = 0.0
    return nothing
end

function cuda_kernel_ttgc_b_15!(auxu_stack, cell_vol, i_cell_to_node, i_ncell, i_njac, i_nnode, i_seq_, mup, mup_stack, npernode_half, up)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(i_ncell - 1, 1) + 1
        return nothing
    end
    i_cell = 1 + (__tid - 1)
    i_node1 = i_cell_to_node[1, i_cell]
    i_node2 = i_cell_to_node[2, i_cell]
    i_node3 = i_cell_to_node[3, i_cell]
    i_node4 = i_cell_to_node[4, i_cell]
    auxu_stack[((div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1) + (div(i_njac - 1, 1) + 1)) + (((i_seq_ - 1) * (div(i_ncell - 1, 1) + 1) + (i_cell - 1)) + 1)] = auxu
    auxu = up[i_node1] + up[i_node2] + up[i_node3] + up[i_node4]
    mup_stack[((((((((div(i_njac - 1, 1) + 1) * (div(i_nnode - 1, 1) + 1) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(npernode_half - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(npernode_half - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_nnode - 1, 1) + 1)) + (((i_seq_ - 1) * (div(i_ncell - 1, 1) + 1) + (i_cell - 1)) + 1)] = mup[i_node1]
    CUDA.@atomic mup[i_node1] += ((auxu + up[i_node1]) * cell_vol[i_cell]) / 20
    mup_stack[(((((((((div(i_njac - 1, 1) + 1) * (div(i_nnode - 1, 1) + 1) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(npernode_half - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(npernode_half - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_nnode - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (((i_seq_ - 1) * (div(i_ncell - 1, 1) + 1) + (i_cell - 1)) + 1)] = mup[i_node2]
    CUDA.@atomic mup[i_node2] += ((auxu + up[i_node2]) * cell_vol[i_cell]) / 20
    mup_stack[((((((((((div(i_njac - 1, 1) + 1) * (div(i_nnode - 1, 1) + 1) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(npernode_half - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(npernode_half - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_nnode - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (((i_seq_ - 1) * (div(i_ncell - 1, 1) + 1) + (i_cell - 1)) + 1)] = mup[i_node3]
    CUDA.@atomic mup[i_node3] += ((auxu + up[i_node3]) * cell_vol[i_cell]) / 20
    mup_stack[(((((((((((div(i_njac - 1, 1) + 1) * (div(i_nnode - 1, 1) + 1) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(npernode_half - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(npernode_half - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_nnode - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (((i_seq_ - 1) * (div(i_ncell - 1, 1) + 1) + (i_cell - 1)) + 1)] = mup[i_node4]
    CUDA.@atomic mup[i_node4] += ((auxu + up[i_node4]) * cell_vol[i_cell]) / 20
    return nothing
end

function cuda_kernel_ttgc_b_16!(i_node_perio, mup, npernode_half, resperio)
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

function cuda_kernel_ttgc_b_17!(i_ncell, i_njac, i_nnode, i_node_perio, i_seq_, mup, mup_stack, npernode_half, resperio)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(npernode_half - 1, 1) + 1
        return nothing
    end
    k = 1 + (__tid - 1)
    i1 = i_node_perio[k, 1]
    i2 = i_node_perio[k, 2]
    mup_stack[((((((((((((div(i_njac - 1, 1) + 1) * (div(i_nnode - 1, 1) + 1) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(npernode_half - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(npernode_half - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_nnode - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (((i_seq_ - 1) * (div(npernode_half - 1, 1) + 1) + (k - 1)) + 1)] = mup[i1]
    CUDA.@atomic mup[i1] += resperio[k, 2]
    mup_stack[(((((((((((((div(i_njac - 1, 1) + 1) * (div(i_nnode - 1, 1) + 1) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(npernode_half - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(npernode_half - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_nnode - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(npernode_half - 1, 1) + 1)) + (((i_seq_ - 1) * (div(npernode_half - 1, 1) + 1) + (k - 1)) + 1)] = mup[i2]
    CUDA.@atomic mup[i2] += resperio[k, 1]
    return nothing
end

function cuda_kernel_ttgc_b_18!(i_njac, i_nnode, i_seq_, mup, node_vol, res2, up, up_stack)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(i_nnode - 1, 1) + 1
        return nothing
    end
    i_node = 1 + (__tid - 1)
    up_stack[((div(i_njac - 1, 1) + 1) * (div(i_nnode - 1, 1) + 1) + (div(i_nnode - 1, 1) + 1)) + (((i_seq_ - 1) * (div(i_nnode - 1, 1) + 1) + (i_node - 1)) + 1)] = up[i_node]
    up[i_node] = up[i_node] + (res2[i_node] - mup[i_node]) / node_vol[i_node]
    return nothing
end

function cuda_kernel_ttgc_b_19!(i_nnode, lossb, u, ub, up, upb, uref, urefb)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(1 - i_nnode, -1) + 1
        return nothing
    end
    i_seq_node = i_nnode + (__tid - 1) * -1
    ub[i_seq_node] = ub[i_seq_node] + (2 * ((u[i_seq_node] + up[i_seq_node]) - uref[i_seq_node])) * lossb[1]
    upb[i_seq_node] = upb[i_seq_node] + (2 * ((u[i_seq_node] + up[i_seq_node]) - uref[i_seq_node])) * lossb[1]
    urefb[i_seq_node] = urefb[i_seq_node] + -((2 * ((u[i_seq_node] + up[i_seq_node]) - uref[i_seq_node])) * lossb[1])
    return nothing
end

function cuda_kernel_ttgc_b_20!(i_njac, i_nnode, i_seq_, mup, mupb, node_vol, node_volb, res2, res2b, up, up_stack, upb)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(1 - i_nnode, -1) + 1
        return nothing
    end
    i_node = i_nnode + (__tid - 1) * -1
    up[i_node] = up_stack[((div(i_njac - 1, 1) + 1) * (div(i_nnode - 1, 1) + 1) + (div(i_nnode - 1, 1) + 1)) + (((i_seq_ - 1) * (div(i_nnode - 1, 1) + 1) + (i_node - 1)) + 1)]
    res2b[i_node] = res2b[i_node] + (1.0 / node_vol[i_node]) * upb[i_node]
    mupb[i_node] = mupb[i_node] + -((1.0 / node_vol[i_node]) * upb[i_node])
    node_volb[i_node] = node_volb[i_node] + -((res2[i_node] - mup[i_node]) / node_vol[i_node] ^ 2) * upb[i_node]
    return nothing
end

function cuda_kernel_ttgc_b_21!(i_ncell, i_njac, i_nnode, i_node_perio, i_seq_, mup, mup_stack, mupb, npernode_half, resperiob)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(1 - npernode_half, -1) + 1
        return nothing
    end
    k = npernode_half + (__tid - 1) * -1
    i1 = i_node_perio[k, 1]
    i2 = i_node_perio[k, 2]
    mup[i2] = mup_stack[(((((((((((((div(i_njac - 1, 1) + 1) * (div(i_nnode - 1, 1) + 1) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(npernode_half - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(npernode_half - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_nnode - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(npernode_half - 1, 1) + 1)) + (((i_seq_ - 1) * (div(npernode_half - 1, 1) + 1) + (k - 1)) + 1)]
    resperiob[k, 1] = resperiob[k, 1] + mupb[i2]
    mup[i1] = mup_stack[((((((((((((div(i_njac - 1, 1) + 1) * (div(i_nnode - 1, 1) + 1) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(npernode_half - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(npernode_half - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_nnode - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (((i_seq_ - 1) * (div(npernode_half - 1, 1) + 1) + (k - 1)) + 1)]
    resperiob[k, 2] = resperiob[k, 2] + mupb[i1]
    return nothing
end

function cuda_kernel_ttgc_b_22!(i_node_perio, mupb, npernode_half, resperiob)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(npernode_half - 1, 1) + 1
        return nothing
    end
    k = 1 + (__tid - 1)
    i1 = i_node_perio[k, 1]
    i2 = i_node_perio[k, 2]
    CUDA.@atomic mupb[i2] += resperiob[k, 2]
    resperiob[k, 2] = 0.0
    CUDA.@atomic mupb[i1] += resperiob[k, 1]
    resperiob[k, 1] = 0.0
    return nothing
end

function cuda_kernel_ttgc_b_23!(auxu_stack, cell_vol, cell_volb, i_cell_to_node, i_ncell, i_njac, i_nnode, i_seq_, mup, mup_stack, mupb, npernode_half, up, upb)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(1 - i_ncell, -1) + 1
        return nothing
    end
    i_cell = i_ncell + (__tid - 1) * -1
    i_node1 = i_cell_to_node[1, i_cell]
    i_node2 = i_cell_to_node[2, i_cell]
    i_node3 = i_cell_to_node[3, i_cell]
    i_node4 = i_cell_to_node[4, i_cell]
    mup[i_node4] = mup_stack[(((((((((((div(i_njac - 1, 1) + 1) * (div(i_nnode - 1, 1) + 1) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(npernode_half - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(npernode_half - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_nnode - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (((i_seq_ - 1) * (div(i_ncell - 1, 1) + 1) + (i_cell - 1)) + 1)]
    auxub = auxub + cell_vol[i_cell] * (0.05 * mupb[i_node4])
    CUDA.@atomic upb[i_node4] += cell_vol[i_cell] * (0.05 * mupb[i_node4])
    cell_volb[i_cell] = cell_volb[i_cell] + (auxu + up[i_node4]) * (0.05 * mupb[i_node4])
    mup[i_node3] = mup_stack[((((((((((div(i_njac - 1, 1) + 1) * (div(i_nnode - 1, 1) + 1) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(npernode_half - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(npernode_half - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_nnode - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (((i_seq_ - 1) * (div(i_ncell - 1, 1) + 1) + (i_cell - 1)) + 1)]
    auxub = auxub + cell_vol[i_cell] * (0.05 * mupb[i_node3])
    CUDA.@atomic upb[i_node3] += cell_vol[i_cell] * (0.05 * mupb[i_node3])
    cell_volb[i_cell] = cell_volb[i_cell] + (auxu + up[i_node3]) * (0.05 * mupb[i_node3])
    mup[i_node2] = mup_stack[(((((((((div(i_njac - 1, 1) + 1) * (div(i_nnode - 1, 1) + 1) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(npernode_half - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(npernode_half - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_nnode - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (((i_seq_ - 1) * (div(i_ncell - 1, 1) + 1) + (i_cell - 1)) + 1)]
    auxub = auxub + cell_vol[i_cell] * (0.05 * mupb[i_node2])
    CUDA.@atomic upb[i_node2] += cell_vol[i_cell] * (0.05 * mupb[i_node2])
    cell_volb[i_cell] = cell_volb[i_cell] + (auxu + up[i_node2]) * (0.05 * mupb[i_node2])
    mup[i_node1] = mup_stack[((((((((div(i_njac - 1, 1) + 1) * (div(i_nnode - 1, 1) + 1) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(npernode_half - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(npernode_half - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_nnode - 1, 1) + 1)) + (((i_seq_ - 1) * (div(i_ncell - 1, 1) + 1) + (i_cell - 1)) + 1)]
    auxub = auxub + cell_vol[i_cell] * (0.05 * mupb[i_node1])
    CUDA.@atomic upb[i_node1] += cell_vol[i_cell] * (0.05 * mupb[i_node1])
    cell_volb[i_cell] = cell_volb[i_cell] + (auxu + up[i_node1]) * (0.05 * mupb[i_node1])
    auxu = auxu_stack[((div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1) + (div(i_njac - 1, 1) + 1)) + (((i_seq_ - 1) * (div(i_ncell - 1, 1) + 1) + (i_cell - 1)) + 1)]
    CUDA.@atomic upb[i_node1] += auxub
    CUDA.@atomic upb[i_node2] += auxub
    CUDA.@atomic upb[i_node3] += auxub
    CUDA.@atomic upb[i_node4] += auxub
    auxub = 0.0
    return nothing
end

function cuda_kernel_ttgc_b_24!(i_ncell, i_njac, i_nnode, i_seq_, mup, mup_stack, mupb, npernode_half)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(1 - i_nnode, -1) + 1
        return nothing
    end
    i_node = i_nnode + (__tid - 1) * -1
    mup[i_node] = mup_stack[(((((((div(i_njac - 1, 1) + 1) * (div(i_nnode - 1, 1) + 1) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(npernode_half - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(npernode_half - 1, 1) + 1)) + (((i_seq_ - 1) * (div(i_nnode - 1, 1) + 1) + (i_node - 1)) + 1)]
    mupb[i_node] = 0.0
    return nothing
end

function cuda_kernel_ttgc_b_25!(i_njac, i_nnode, node_vol, node_volb, res2, res2b, up, up_stack, upb)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(1 - i_nnode, -1) + 1
        return nothing
    end
    i_node = i_nnode + (__tid - 1) * -1
    up[i_node] = up_stack[(div(i_njac - 1, 1) + 1) * (div(i_nnode - 1, 1) + 1) + ((i_node - 1) + 1)]
    res2b[i_node] = res2b[i_node] + (1.0 / node_vol[i_node]) * upb[i_node]
    node_volb[i_node] = node_volb[i_node] + -(res2[i_node] / node_vol[i_node] ^ 2) * upb[i_node]
    upb[i_node] = 0.0
    return nothing
end

function cuda_kernel_ttgc_b_26!(i_node_perio, npernode_half, res2b, resperiob)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(npernode_half - 1, 1) + 1
        return nothing
    end
    k = 1 + (__tid - 1)
    i1 = i_node_perio[k, 1]
    i2 = i_node_perio[k, 2]
    resperiob[k, 1] = resperiob[k, 1] + res2b[i2]
    resperiob[k, 2] = resperiob[k, 2] + res2b[i1]
    return nothing
end

function cuda_kernel_ttgc_b_27!(i_node_perio, npernode_half, res2b, resperiob)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(npernode_half - 1, 1) + 1
        return nothing
    end
    k = 1 + (__tid - 1)
    i1 = i_node_perio[k, 1]
    i2 = i_node_perio[k, 2]
    CUDA.@atomic res2b[i2] += resperiob[k, 2]
    resperiob[k, 2] = 0.0
    CUDA.@atomic res2b[i1] += resperiob[k, 1]
    resperiob[k, 1] = 0.0
    return nothing
end

function cuda_kernel_ttgc_b_28!(c, cb, dt, dtb, i_cell_to_node, i_ncell, res2b, skx, skxb, sky, skyb, u, ub, up, upb, vere_stack)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(1 - i_ncell, -1) + 1
        return nothing
    end
    i_cell = i_ncell + (__tid - 1) * -1
    vere = vere_stack[(((div(i_ncell - 1, 1) + 1) * (div(4 - 1, 1) + 1) + (div(i_ncell - 1, 1) + 1)) + (div(i_ncell - 1, 1) + 1) * (div(4 - 1, 1) + 1)) + ((i_cell - 1) + 1)]
    for i_loc = 4:-1:1
        i_node = i_cell_to_node[i_loc, i_cell]
        for i_k = 1:4
            i_k_node = i_cell_to_node[i_k, i_cell]
            resib = resib + res2b[i_k_node]
        end
        CUDA.@atomic dtb[1] += 0.25 * -(vere * resib)
        vereb = vereb + -(dt / 4) * resib
        resib = 0.0
        vere = vere_stack[((div(i_ncell - 1, 1) + 1) * (div(4 - 1, 1) + 1) + (div(i_ncell - 1, 1) + 1)) + (((i_cell - 1) * (div(4 - 1, 1) + 1) + (i_loc - 1)) + 1)]
        CUDA.@atomic ub[i_node] += (-(1.0 / 3.0) * (c[1, i_node] * skx[i_loc, i_cell] + c[2, i_node] * sky[i_loc, i_cell])) * vereb
        CUDA.@atomic upb[i_node] += (-(1.0 / 3.0) * (c[1, i_node] * skx[i_loc, i_cell] + c[2, i_node] * sky[i_loc, i_cell])) * vereb
        CUDA.@atomic cb[1, i_node] += skx[i_loc, i_cell] * ((-(1.0 / 3.0) * (u[i_node] + up[i_node])) * vereb)
        skxb[i_loc, i_cell] = skxb[i_loc, i_cell] + c[1, i_node] * ((-(1.0 / 3.0) * (u[i_node] + up[i_node])) * vereb)
        CUDA.@atomic cb[2, i_node] += sky[i_loc, i_cell] * ((-(1.0 / 3.0) * (u[i_node] + up[i_node])) * vereb)
        skyb[i_loc, i_cell] = skyb[i_loc, i_cell] + c[2, i_node] * ((-(1.0 / 3.0) * (u[i_node] + up[i_node])) * vereb)
        vereb = 0.0
    end
    return nothing
end

function cuda_kernel_ttgc_b_29!(i_nnode, i_seq_, mup, mupb, node_vol, node_volb, res, resb, up, up_stack, upb)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(1 - i_nnode, -1) + 1
        return nothing
    end
    i_node = i_nnode + (__tid - 1) * -1
    up[i_node] = up_stack[((i_seq_ - 1) * (div(i_nnode - 1, 1) + 1) + (i_node - 1)) + 1]
    resb[i_node] = resb[i_node] + (1.0 / node_vol[i_node]) * upb[i_node]
    mupb[i_node] = mupb[i_node] + -((1.0 / node_vol[i_node]) * upb[i_node])
    node_volb[i_node] = node_volb[i_node] + -((res[i_node] - mup[i_node]) / node_vol[i_node] ^ 2) * upb[i_node]
    return nothing
end

function cuda_kernel_ttgc_b_30!(i_ncell, i_njac, i_nnode, i_node_perio, i_seq_, mup, mup_stack, mupb, npernode_half, resperiob)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(1 - npernode_half, -1) + 1
        return nothing
    end
    k = npernode_half + (__tid - 1) * -1
    i1 = i_node_perio[k, 1]
    i2 = i_node_perio[k, 2]
    mup[i2] = mup_stack[((((((div(i_njac - 1, 1) + 1) * (div(i_nnode - 1, 1) + 1) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(npernode_half - 1, 1) + 1)) + (((i_seq_ - 1) * (div(npernode_half - 1, 1) + 1) + (k - 1)) + 1)]
    resperiob[k, 1] = resperiob[k, 1] + mupb[i2]
    mup[i1] = mup_stack[(((((div(i_njac - 1, 1) + 1) * (div(i_nnode - 1, 1) + 1) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (((i_seq_ - 1) * (div(npernode_half - 1, 1) + 1) + (k - 1)) + 1)]
    resperiob[k, 2] = resperiob[k, 2] + mupb[i1]
    return nothing
end

function cuda_kernel_ttgc_b_31!(i_node_perio, mupb, npernode_half, resperiob)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(npernode_half - 1, 1) + 1
        return nothing
    end
    k = 1 + (__tid - 1)
    i1 = i_node_perio[k, 1]
    i2 = i_node_perio[k, 2]
    CUDA.@atomic mupb[i2] += resperiob[k, 2]
    resperiob[k, 2] = 0.0
    CUDA.@atomic mupb[i1] += resperiob[k, 1]
    resperiob[k, 1] = 0.0
    return nothing
end

function cuda_kernel_ttgc_b_32!(auxu_stack, cell_vol, cell_volb, i_cell_to_node, i_ncell, i_njac, i_nnode, i_seq_, mup, mup_stack, mupb, up, upb)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(1 - i_ncell, -1) + 1
        return nothing
    end
    i_cell = i_ncell + (__tid - 1) * -1
    i_node1 = i_cell_to_node[1, i_cell]
    i_node2 = i_cell_to_node[2, i_cell]
    i_node3 = i_cell_to_node[3, i_cell]
    i_node4 = i_cell_to_node[4, i_cell]
    mup[i_node4] = mup_stack[((((div(i_njac - 1, 1) + 1) * (div(i_nnode - 1, 1) + 1) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (((i_seq_ - 1) * (div(i_ncell - 1, 1) + 1) + (i_cell - 1)) + 1)]
    auxub = auxub + cell_vol[i_cell] * (0.05 * mupb[i_node4])
    CUDA.@atomic upb[i_node4] += cell_vol[i_cell] * (0.05 * mupb[i_node4])
    cell_volb[i_cell] = cell_volb[i_cell] + (auxu + up[i_node4]) * (0.05 * mupb[i_node4])
    mup[i_node3] = mup_stack[(((div(i_njac - 1, 1) + 1) * (div(i_nnode - 1, 1) + 1) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (((i_seq_ - 1) * (div(i_ncell - 1, 1) + 1) + (i_cell - 1)) + 1)]
    auxub = auxub + cell_vol[i_cell] * (0.05 * mupb[i_node3])
    CUDA.@atomic upb[i_node3] += cell_vol[i_cell] * (0.05 * mupb[i_node3])
    cell_volb[i_cell] = cell_volb[i_cell] + (auxu + up[i_node3]) * (0.05 * mupb[i_node3])
    mup[i_node2] = mup_stack[((div(i_njac - 1, 1) + 1) * (div(i_nnode - 1, 1) + 1) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (((i_seq_ - 1) * (div(i_ncell - 1, 1) + 1) + (i_cell - 1)) + 1)]
    auxub = auxub + cell_vol[i_cell] * (0.05 * mupb[i_node2])
    CUDA.@atomic upb[i_node2] += cell_vol[i_cell] * (0.05 * mupb[i_node2])
    cell_volb[i_cell] = cell_volb[i_cell] + (auxu + up[i_node2]) * (0.05 * mupb[i_node2])
    mup[i_node1] = mup_stack[(div(i_njac - 1, 1) + 1) * (div(i_nnode - 1, 1) + 1) + (((i_seq_ - 1) * (div(i_ncell - 1, 1) + 1) + (i_cell - 1)) + 1)]
    auxub = auxub + cell_vol[i_cell] * (0.05 * mupb[i_node1])
    CUDA.@atomic upb[i_node1] += cell_vol[i_cell] * (0.05 * mupb[i_node1])
    cell_volb[i_cell] = cell_volb[i_cell] + (auxu + up[i_node1]) * (0.05 * mupb[i_node1])
    auxu = auxu_stack[((i_seq_ - 1) * (div(i_ncell - 1, 1) + 1) + (i_cell - 1)) + 1]
    CUDA.@atomic upb[i_node1] += auxub
    CUDA.@atomic upb[i_node2] += auxub
    CUDA.@atomic upb[i_node3] += auxub
    CUDA.@atomic upb[i_node4] += auxub
    auxub = 0.0
    return nothing
end

function cuda_kernel_ttgc_b_33!(i_nnode, i_seq_, mup, mup_stack, mupb)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(1 - i_nnode, -1) + 1
        return nothing
    end
    i_node = i_nnode + (__tid - 1) * -1
    mup[i_node] = mup_stack[((i_seq_ - 1) * (div(i_nnode - 1, 1) + 1) + (i_node - 1)) + 1]
    mupb[i_node] = 0.0
    return nothing
end

function cuda_kernel_ttgc_b_34!(i_nnode, node_vol, node_volb, res, resb, upb)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(i_nnode - 1, 1) + 1
        return nothing
    end
    i_node = 1 + (__tid - 1)
    resb[i_node] = resb[i_node] + (1.0 / node_vol[i_node]) * upb[i_node]
    node_volb[i_node] = node_volb[i_node] + -(res[i_node] / node_vol[i_node] ^ 2) * upb[i_node]
    upb[i_node] = 0.0
    return nothing
end

function cuda_kernel_ttgc_b_35!(i_node_perio, npernode_half, resb, resperiob)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(npernode_half - 1, 1) + 1
        return nothing
    end
    k = 1 + (__tid - 1)
    i1 = i_node_perio[k, 1]
    i2 = i_node_perio[k, 2]
    resperiob[k, 1] = resperiob[k, 1] + resb[i2]
    resperiob[k, 2] = resperiob[k, 2] + resb[i1]
    return nothing
end

function cuda_kernel_ttgc_b_36!(i_node_perio, npernode_half, resb, resperiob)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(npernode_half - 1, 1) + 1
        return nothing
    end
    k = 1 + (__tid - 1)
    i1 = i_node_perio[k, 1]
    i2 = i_node_perio[k, 2]
    CUDA.@atomic resb[i2] += resperiob[k, 2]
    resperiob[k, 2] = 0.0
    CUDA.@atomic resb[i1] += resperiob[k, 1]
    resperiob[k, 1] = 0.0
    return nothing
end

function cuda_kernel_ttgc_b_37!(aeresk_stack, aerex_stack, aerey_stack, aerez_stack, beta, betab, c, cavgx_stack, cavgy_stack, cavgz_stack, cb, cell_vol, cell_volb, dt, dtb, factor_stack, gamma, gammab, i_cell_to_node, i_ncell, re_stack, res2b, resb, skx, skxb, sky, skyb, skz, skzb, u, ub, vere_stack)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(1 - i_ncell, -1) + 1
        return nothing
    end
    i_cell = i_ncell + (__tid - 1) * -1
    aeresk = aeresk_stack[((div(i_ncell - 1, 1) + 1) * (div(4 - 1, 1) + 1) * (div(4 - 1, 1) + 1) + (div(i_ncell - 1, 1) + 1) * (div(4 - 1, 1) + 1)) + ((i_cell - 1) + 1)]
    aerex = aerex_stack[(div(i_ncell - 1, 1) + 1) * (div(4 - 1, 1) + 1) + ((i_cell - 1) + 1)]
    aerey = aerey_stack[(div(i_ncell - 1, 1) + 1) * (div(4 - 1, 1) + 1) + ((i_cell - 1) + 1)]
    aerez = aerez_stack[(div(i_ncell - 1, 1) + 1) * (div(4 - 1, 1) + 1) + ((i_cell - 1) + 1)]
    cavgx = cavgx_stack[((div(i_ncell - 1, 1) + 1) + (div(i_ncell - 1, 1) + 1) * (div(4 - 1, 1) + 1)) + ((i_cell - 1) + 1)]
    cavgy = cavgy_stack[((div(i_ncell - 1, 1) + 1) + (div(i_ncell - 1, 1) + 1) * (div(4 - 1, 1) + 1)) + ((i_cell - 1) + 1)]
    cavgz = cavgz_stack[((div(i_ncell - 1, 1) + 1) + (div(i_ncell - 1, 1) + 1) * (div(4 - 1, 1) + 1)) + ((i_cell - 1) + 1)]
    factor = factor_stack[((div(i_ncell - 1, 1) + 1) * (div(4 - 1, 1) + 1) * (div(4 - 1, 1) + 1) + (div(i_ncell - 1, 1) + 1) * (div(4 - 1, 1) + 1)) + ((i_cell - 1) + 1)]
    re = re_stack[(div(i_ncell - 1, 1) + 1) * (div(4 - 1, 1) + 1) + ((i_cell - 1) + 1)]
    vere = vere_stack[(div(i_ncell - 1, 1) + 1) * (div(4 - 1, 1) + 1) + ((i_cell - 1) + 1)]
    for i_loc = 4:-1:1
        aeresk = aeresk_stack[(div(i_ncell - 1, 1) + 1) * (div(4 - 1, 1) + 1) * (div(4 - 1, 1) + 1) + (((i_cell - 1) * (div(4 - 1, 1) + 1) + (i_loc - 1)) + 1)]
        factor = factor_stack[(div(i_ncell - 1, 1) + 1) * (div(4 - 1, 1) + 1) * (div(4 - 1, 1) + 1) + (((i_cell - 1) * (div(4 - 1, 1) + 1) + (i_loc - 1)) + 1)]
        i_node = i_cell_to_node[i_loc, i_cell]
        for i_k = 4:-1:1
            i_k_node = i_cell_to_node[i_k, i_cell]
            auxres2b = auxres2b + res2b[i_k_node]
            auxresb = auxresb + resb[i_k_node]
            factorb = factorb + gamma[i_cell] * auxres2b
            gammab[i_cell] = gammab[i_cell] + factor * auxres2b
            auxres2b = 0.0
            resib = resib + auxresb
            factorb = factorb + beta[i_cell] * auxresb
            betab[i_cell] = betab[i_cell] + factor * auxresb
            auxresb = 0.0
            factor = factor_stack[((i_cell - 1) * ((div(4 - 1, 1) + 1) * (div(4 - 1, 1) + 1)) + (i_loc - 1) * (div(4 - 1, 1) + 1) + (i_k - 1)) + 1]
            CUDA.@atomic dtb[1] += (2dt) * (0.3333333333333333 * (aeresk * factorb))
            aereskb = aereskb + (dt ^ 2 / 3) * factorb
            factorb = 0.0
            aeresk = aeresk_stack[((i_cell - 1) * ((div(4 - 1, 1) + 1) * (div(4 - 1, 1) + 1)) + (i_loc - 1) * (div(4 - 1, 1) + 1) + (i_k - 1)) + 1]
            aerexb = aerexb + skx[i_k, i_cell] * aereskb
            skxb[i_k, i_cell] = skxb[i_k, i_cell] + aerex * aereskb
            aereyb = aereyb + sky[i_k, i_cell] * aereskb
            skyb[i_k, i_cell] = skyb[i_k, i_cell] + aerey * aereskb
            aerezb = aerezb + skz[i_k, i_cell] * aereskb
            skzb[i_k, i_cell] = skzb[i_k, i_cell] + aerez * aereskb
            aereskb = 0.0
        end
        CUDA.@atomic dtb[1] += 0.25 * -(((0.5 - gamma[i_cell]) * vere) * resib)
        gammab[i_cell] = gammab[i_cell] + -((-(dt / 4) * vere) * resib)
        vereb = vereb + (-(dt / 4) * (0.5 - gamma[i_cell])) * resib
        resib = 0.0
        aerez = aerez_stack[((i_cell - 1) * (div(4 - 1, 1) + 1) + (i_loc - 1)) + 1]
        cavgzb = cavgzb + re * aerezb
        reb = reb + cavgz * aerezb
        aerezb = 0.0
        aerey = aerey_stack[((i_cell - 1) * (div(4 - 1, 1) + 1) + (i_loc - 1)) + 1]
        cavgyb = cavgyb + re * aereyb
        reb = reb + cavgy * aereyb
        aereyb = 0.0
        aerex = aerex_stack[((i_cell - 1) * (div(4 - 1, 1) + 1) + (i_loc - 1)) + 1]
        cavgxb = cavgxb + re * aerexb
        reb = reb + cavgx * aerexb
        aerexb = 0.0
        re = re_stack[((i_cell - 1) * (div(4 - 1, 1) + 1) + (i_loc - 1)) + 1]
        vereb = vereb + (1.0 / cell_vol[i_cell]) * reb
        cell_volb[i_cell] = cell_volb[i_cell] + -(vere / cell_vol[i_cell] ^ 2) * reb
        reb = 0.0
        vere = vere_stack[((i_cell - 1) * (div(4 - 1, 1) + 1) + (i_loc - 1)) + 1]
        CUDA.@atomic ub[i_node] += (-(1.0 / 3.0) * (c[1, i_node] * skx[i_loc, i_cell] + c[2, i_node] * sky[i_loc, i_cell] + c[3, i_node] * skz[i_loc, i_cell])) * vereb
        CUDA.@atomic cb[1, i_node] += skx[i_loc, i_cell] * ((-(1.0 / 3.0) * u[i_node]) * vereb)
        skxb[i_loc, i_cell] = skxb[i_loc, i_cell] + c[1, i_node] * ((-(1.0 / 3.0) * u[i_node]) * vereb)
        CUDA.@atomic cb[2, i_node] += sky[i_loc, i_cell] * ((-(1.0 / 3.0) * u[i_node]) * vereb)
        skyb[i_loc, i_cell] = skyb[i_loc, i_cell] + c[2, i_node] * ((-(1.0 / 3.0) * u[i_node]) * vereb)
        CUDA.@atomic cb[3, i_node] += skz[i_loc, i_cell] * ((-(1.0 / 3.0) * u[i_node]) * vereb)
        skzb[i_loc, i_cell] = skzb[i_loc, i_cell] + c[3, i_node] * ((-(1.0 / 3.0) * u[i_node]) * vereb)
        vereb = 0.0
    end
    for i_loc = 4:-1:1
        i_node = i_cell_to_node[i_loc, i_cell]
        cavgz = cavgz_stack[(div(i_ncell - 1, 1) + 1) + (((i_cell - 1) * (div(4 - 1, 1) + 1) + (i_loc - 1)) + 1)]
        CUDA.@atomic cb[3, i_node] += 0.25cavgzb
        cavgy = cavgy_stack[(div(i_ncell - 1, 1) + 1) + (((i_cell - 1) * (div(4 - 1, 1) + 1) + (i_loc - 1)) + 1)]
        CUDA.@atomic cb[2, i_node] += 0.25cavgyb
        cavgx = cavgx_stack[(div(i_ncell - 1, 1) + 1) + (((i_cell - 1) * (div(4 - 1, 1) + 1) + (i_loc - 1)) + 1)]
        CUDA.@atomic cb[1, i_node] += 0.25cavgxb
    end
    cavgz = cavgz_stack[(i_cell - 1) + 1]
    cavgzb = 0.0
    cavgy = cavgy_stack[(i_cell - 1) + 1]
    cavgyb = 0.0
    cavgx = cavgx_stack[(i_cell - 1) + 1]
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

function ttgc_b_cuda(u, ub, uref, urefb, i_cell_to_node, cell_vol, cell_volb, node_vol, node_volb, skx, skxb, sky, skyb, skz, skzb, i_ncell, i_nnode, c, cb, dt, dtb, beta, betab, gamma, gammab, i_njac, res, resb, res2, res2b, up, upb, mup, mupb, npernode_half, resperio, resperiob, i_node_perio, loss, lossb, cavgx_stack, cavgy_stack, cavgz_stack, vere_stack, re_stack, aerex_stack, aerey_stack, aerez_stack, aeresk_stack, factor_stack, mup_stack, auxu_stack, up_stack)
    dtb = CuArray([dtb])
    nthread_per_block = 256
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
    @cuda threads = nthread_per_block blocks = cld(div(i_ncell - 1, 1) + 1, nthread_per_block) cuda_kernel_ttgc_b_1!(aeresk_stack, aerex_stack, aerey_stack, aerez_stack, beta, c, cavgx_stack, cavgy_stack, cavgz_stack, cell_vol, dt, factor_stack, gamma, i_cell_to_node, i_ncell, re_stack, res, res2, skx, sky, skz, u, vere_stack)
    @cuda threads = nthread_per_block blocks = cld(div(npernode_half - 1, 1) + 1, nthread_per_block) cuda_kernel_ttgc_b_2!(i_node_perio, npernode_half, res, resperio)
    @cuda threads = nthread_per_block blocks = cld(div(npernode_half - 1, 1) + 1, nthread_per_block) cuda_kernel_ttgc_b_3!(i_node_perio, npernode_half, res, resperio)
    @cuda threads = nthread_per_block blocks = cld(div(i_nnode - 1, 1) + 1, nthread_per_block) cuda_kernel_ttgc_b_4!(i_nnode, node_vol, res, up)
    for i_seq_ = 1:i_njac
        @cuda threads = nthread_per_block blocks = cld(div(i_nnode - 1, 1) + 1, nthread_per_block) cuda_kernel_ttgc_b_5!(i_nnode, i_seq_, mup, mup_stack)
        @cuda threads = nthread_per_block blocks = cld(div(i_ncell - 1, 1) + 1, nthread_per_block) cuda_kernel_ttgc_b_6!(auxu_stack, cell_vol, i_cell_to_node, i_ncell, i_njac, i_nnode, i_seq_, mup, mup_stack, up)
        @cuda threads = nthread_per_block blocks = cld(div(npernode_half - 1, 1) + 1, nthread_per_block) cuda_kernel_ttgc_b_7!(i_node_perio, mup, npernode_half, resperio)
        @cuda threads = nthread_per_block blocks = cld(div(npernode_half - 1, 1) + 1, nthread_per_block) cuda_kernel_ttgc_b_8!(i_ncell, i_njac, i_nnode, i_node_perio, i_seq_, mup, mup_stack, npernode_half, resperio)
        @cuda threads = nthread_per_block blocks = cld(div(i_nnode - 1, 1) + 1, nthread_per_block) cuda_kernel_ttgc_b_9!(i_nnode, i_seq_, mup, node_vol, res, up, up_stack)
        CUDA.@allowscalar begin
                auxu_stack[(div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1) + ((i_seq_ - 1) + 1)] = auxu
            end
    end
    @cuda threads = nthread_per_block blocks = cld(div(i_ncell - 1, 1) + 1, nthread_per_block) cuda_kernel_ttgc_b_10!(c, dt, i_cell_to_node, i_ncell, res2, skx, sky, u, up, vere_stack)
    @cuda threads = nthread_per_block blocks = cld(div(npernode_half - 1, 1) + 1, nthread_per_block) cuda_kernel_ttgc_b_11!(i_node_perio, npernode_half, res2, resperio)
    @cuda threads = nthread_per_block blocks = cld(div(npernode_half - 1, 1) + 1, nthread_per_block) cuda_kernel_ttgc_b_12!(i_node_perio, npernode_half, res2, resperio)
    @cuda threads = nthread_per_block blocks = cld(div(i_nnode - 1, 1) + 1, nthread_per_block) cuda_kernel_ttgc_b_13!(i_njac, i_nnode, node_vol, res2, up, up_stack)
    for i_seq_ = 1:i_njac
        @cuda threads = nthread_per_block blocks = cld(div(i_nnode - 1, 1) + 1, nthread_per_block) cuda_kernel_ttgc_b_14!(i_ncell, i_njac, i_nnode, i_seq_, mup, mup_stack, npernode_half)
        @cuda threads = nthread_per_block blocks = cld(div(i_ncell - 1, 1) + 1, nthread_per_block) cuda_kernel_ttgc_b_15!(auxu_stack, cell_vol, i_cell_to_node, i_ncell, i_njac, i_nnode, i_seq_, mup, mup_stack, npernode_half, up)
        @cuda threads = nthread_per_block blocks = cld(div(npernode_half - 1, 1) + 1, nthread_per_block) cuda_kernel_ttgc_b_16!(i_node_perio, mup, npernode_half, resperio)
        @cuda threads = nthread_per_block blocks = cld(div(npernode_half - 1, 1) + 1, nthread_per_block) cuda_kernel_ttgc_b_17!(i_ncell, i_njac, i_nnode, i_node_perio, i_seq_, mup, mup_stack, npernode_half, resperio)
        @cuda threads = nthread_per_block blocks = cld(div(i_nnode - 1, 1) + 1, nthread_per_block) cuda_kernel_ttgc_b_18!(i_njac, i_nnode, i_seq_, mup, node_vol, res2, up, up_stack)
        CUDA.@allowscalar begin
                auxu_stack[(((div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1) + (div(i_njac - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + ((i_seq_ - 1) + 1)] = auxu
            end
    end
    CUDA.@allowscalar begin
            loss[1] = loss[1] + mapreduce(((__mr_1, __mr_2, __mr_3)->((__mr_1 + __mr_2) - __mr_3) ^ 2), +, u, up, uref)
            aeresk_stack[(((div(i_ncell - 1, 1) + 1) * (div(4 - 1, 1) + 1) * (div(4 - 1, 1) + 1) + (div(i_ncell - 1, 1) + 1) * (div(4 - 1, 1) + 1)) + (div(i_ncell - 1, 1) + 1)) + 1] = aeresk
            aerex_stack[((div(i_ncell - 1, 1) + 1) * (div(4 - 1, 1) + 1) + (div(i_ncell - 1, 1) + 1)) + 1] = aerex
            aerey_stack[((div(i_ncell - 1, 1) + 1) * (div(4 - 1, 1) + 1) + (div(i_ncell - 1, 1) + 1)) + 1] = aerey
            aerez_stack[((div(i_ncell - 1, 1) + 1) * (div(4 - 1, 1) + 1) + (div(i_ncell - 1, 1) + 1)) + 1] = aerez
            auxu_stack[((((div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1) + (div(i_njac - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1)) + 1] = auxu
            cavgx_stack[(((div(i_ncell - 1, 1) + 1) + (div(i_ncell - 1, 1) + 1) * (div(4 - 1, 1) + 1)) + (div(i_ncell - 1, 1) + 1)) + 1] = cavgx
            cavgy_stack[(((div(i_ncell - 1, 1) + 1) + (div(i_ncell - 1, 1) + 1) * (div(4 - 1, 1) + 1)) + (div(i_ncell - 1, 1) + 1)) + 1] = cavgy
            cavgz_stack[(((div(i_ncell - 1, 1) + 1) + (div(i_ncell - 1, 1) + 1) * (div(4 - 1, 1) + 1)) + (div(i_ncell - 1, 1) + 1)) + 1] = cavgz
            factor_stack[(((div(i_ncell - 1, 1) + 1) * (div(4 - 1, 1) + 1) * (div(4 - 1, 1) + 1) + (div(i_ncell - 1, 1) + 1) * (div(4 - 1, 1) + 1)) + (div(i_ncell - 1, 1) + 1)) + 1] = factor
            re_stack[((div(i_ncell - 1, 1) + 1) * (div(4 - 1, 1) + 1) + (div(i_ncell - 1, 1) + 1)) + 1] = re
            vere_stack[((((div(i_ncell - 1, 1) + 1) * (div(4 - 1, 1) + 1) + (div(i_ncell - 1, 1) + 1)) + (div(i_ncell - 1, 1) + 1) * (div(4 - 1, 1) + 1)) + (div(i_ncell - 1, 1) + 1)) + 1] = vere
            aeresk = aeresk_stack[(((div(i_ncell - 1, 1) + 1) * (div(4 - 1, 1) + 1) * (div(4 - 1, 1) + 1) + (div(i_ncell - 1, 1) + 1) * (div(4 - 1, 1) + 1)) + (div(i_ncell - 1, 1) + 1)) + 1]
            aerex = aerex_stack[((div(i_ncell - 1, 1) + 1) * (div(4 - 1, 1) + 1) + (div(i_ncell - 1, 1) + 1)) + 1]
            aerey = aerey_stack[((div(i_ncell - 1, 1) + 1) * (div(4 - 1, 1) + 1) + (div(i_ncell - 1, 1) + 1)) + 1]
            aerez = aerez_stack[((div(i_ncell - 1, 1) + 1) * (div(4 - 1, 1) + 1) + (div(i_ncell - 1, 1) + 1)) + 1]
            auxu = auxu_stack[((((div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1) + (div(i_njac - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1)) + 1]
            cavgx = cavgx_stack[(((div(i_ncell - 1, 1) + 1) + (div(i_ncell - 1, 1) + 1) * (div(4 - 1, 1) + 1)) + (div(i_ncell - 1, 1) + 1)) + 1]
            cavgy = cavgy_stack[(((div(i_ncell - 1, 1) + 1) + (div(i_ncell - 1, 1) + 1) * (div(4 - 1, 1) + 1)) + (div(i_ncell - 1, 1) + 1)) + 1]
            cavgz = cavgz_stack[(((div(i_ncell - 1, 1) + 1) + (div(i_ncell - 1, 1) + 1) * (div(4 - 1, 1) + 1)) + (div(i_ncell - 1, 1) + 1)) + 1]
            factor = factor_stack[(((div(i_ncell - 1, 1) + 1) * (div(4 - 1, 1) + 1) * (div(4 - 1, 1) + 1) + (div(i_ncell - 1, 1) + 1) * (div(4 - 1, 1) + 1)) + (div(i_ncell - 1, 1) + 1)) + 1]
            re = re_stack[((div(i_ncell - 1, 1) + 1) * (div(4 - 1, 1) + 1) + (div(i_ncell - 1, 1) + 1)) + 1]
            vere = vere_stack[((((div(i_ncell - 1, 1) + 1) * (div(4 - 1, 1) + 1) + (div(i_ncell - 1, 1) + 1)) + (div(i_ncell - 1, 1) + 1) * (div(4 - 1, 1) + 1)) + (div(i_ncell - 1, 1) + 1)) + 1]
        end
    @cuda threads = nthread_per_block blocks = cld(div(1 - i_nnode, -1) + 1, nthread_per_block) cuda_kernel_ttgc_b_19!(i_nnode, lossb, u, ub, up, upb, uref, urefb)
    for i_seq_ = i_njac:-1:1
        CUDA.@allowscalar begin
                auxu = auxu_stack[(((div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1) + (div(i_njac - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + ((i_seq_ - 1) + 1)]
            end
        @cuda threads = nthread_per_block blocks = cld(div(1 - i_nnode, -1) + 1, nthread_per_block) cuda_kernel_ttgc_b_20!(i_njac, i_nnode, i_seq_, mup, mupb, node_vol, node_volb, res2, res2b, up, up_stack, upb)
        @cuda threads = nthread_per_block blocks = cld(div(1 - npernode_half, -1) + 1, nthread_per_block) cuda_kernel_ttgc_b_21!(i_ncell, i_njac, i_nnode, i_node_perio, i_seq_, mup, mup_stack, mupb, npernode_half, resperiob)
        @cuda threads = nthread_per_block blocks = cld(div(npernode_half - 1, 1) + 1, nthread_per_block) cuda_kernel_ttgc_b_22!(i_node_perio, mupb, npernode_half, resperiob)
        @cuda threads = nthread_per_block blocks = cld(div(1 - i_ncell, -1) + 1, nthread_per_block) cuda_kernel_ttgc_b_23!(auxu_stack, cell_vol, cell_volb, i_cell_to_node, i_ncell, i_njac, i_nnode, i_seq_, mup, mup_stack, mupb, npernode_half, up, upb)
        @cuda threads = nthread_per_block blocks = cld(div(1 - i_nnode, -1) + 1, nthread_per_block) cuda_kernel_ttgc_b_24!(i_ncell, i_njac, i_nnode, i_seq_, mup, mup_stack, mupb, npernode_half)
    end
    @cuda threads = nthread_per_block blocks = cld(div(1 - i_nnode, -1) + 1, nthread_per_block) cuda_kernel_ttgc_b_25!(i_njac, i_nnode, node_vol, node_volb, res2, res2b, up, up_stack, upb)
    @cuda threads = nthread_per_block blocks = cld(div(npernode_half - 1, 1) + 1, nthread_per_block) cuda_kernel_ttgc_b_26!(i_node_perio, npernode_half, res2b, resperiob)
    @cuda threads = nthread_per_block blocks = cld(div(npernode_half - 1, 1) + 1, nthread_per_block) cuda_kernel_ttgc_b_27!(i_node_perio, npernode_half, res2b, resperiob)
    @cuda threads = nthread_per_block blocks = cld(div(1 - i_ncell, -1) + 1, nthread_per_block) cuda_kernel_ttgc_b_28!(c, cb, dt, dtb, i_cell_to_node, i_ncell, res2b, skx, skxb, sky, skyb, u, ub, up, upb, vere_stack)
    for i_seq_ = i_njac:-1:1
        CUDA.@allowscalar begin
                auxu = auxu_stack[(div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1) + ((i_seq_ - 1) + 1)]
            end
        @cuda threads = nthread_per_block blocks = cld(div(1 - i_nnode, -1) + 1, nthread_per_block) cuda_kernel_ttgc_b_29!(i_nnode, i_seq_, mup, mupb, node_vol, node_volb, res, resb, up, up_stack, upb)
        @cuda threads = nthread_per_block blocks = cld(div(1 - npernode_half, -1) + 1, nthread_per_block) cuda_kernel_ttgc_b_30!(i_ncell, i_njac, i_nnode, i_node_perio, i_seq_, mup, mup_stack, mupb, npernode_half, resperiob)
        @cuda threads = nthread_per_block blocks = cld(div(npernode_half - 1, 1) + 1, nthread_per_block) cuda_kernel_ttgc_b_31!(i_node_perio, mupb, npernode_half, resperiob)
        @cuda threads = nthread_per_block blocks = cld(div(1 - i_ncell, -1) + 1, nthread_per_block) cuda_kernel_ttgc_b_32!(auxu_stack, cell_vol, cell_volb, i_cell_to_node, i_ncell, i_njac, i_nnode, i_seq_, mup, mup_stack, mupb, up, upb)
        @cuda threads = nthread_per_block blocks = cld(div(1 - i_nnode, -1) + 1, nthread_per_block) cuda_kernel_ttgc_b_33!(i_nnode, i_seq_, mup, mup_stack, mupb)
    end
    @cuda threads = nthread_per_block blocks = cld(div(i_nnode - 1, 1) + 1, nthread_per_block) cuda_kernel_ttgc_b_34!(i_nnode, node_vol, node_volb, res, resb, upb)
    @cuda threads = nthread_per_block blocks = cld(div(npernode_half - 1, 1) + 1, nthread_per_block) cuda_kernel_ttgc_b_35!(i_node_perio, npernode_half, resb, resperiob)
    @cuda threads = nthread_per_block blocks = cld(div(npernode_half - 1, 1) + 1, nthread_per_block) cuda_kernel_ttgc_b_36!(i_node_perio, npernode_half, resb, resperiob)
    @cuda threads = nthread_per_block blocks = cld(div(1 - i_ncell, -1) + 1, nthread_per_block) cuda_kernel_ttgc_b_37!(aeresk_stack, aerex_stack, aerey_stack, aerez_stack, beta, betab, c, cavgx_stack, cavgy_stack, cavgz_stack, cb, cell_vol, cell_volb, dt, dtb, factor_stack, gamma, gammab, i_cell_to_node, i_ncell, re_stack, res2b, resb, skx, skxb, sky, skyb, skz, skzb, u, ub, vere_stack)
    dtb = (Array(dtb))[1]
    return dtb
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
