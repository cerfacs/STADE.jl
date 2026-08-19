import Pkg
haskey(Pkg.project().dependencies, "CUDA") || Pkg.add("CUDA")
using CUDA
using LinearAlgebra
CUDA.allowscalar(false)

function cuda_kernel_ttgc_d_1!(beta, betad, c, cd, cell_vol, cell_vold, dt, dtd, gamma, gammad, i_cell_to_node, i_ncell, res, res2, res2d, resd, skx, skxd, sky, skyd, skz, skzd, u, ud)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(i_ncell - 1, 1) + 1
        return nothing
    end
    i_cell = 1 + (__tid - 1)
    cavgxd = 0.0
    cavgx = 0.0
    cavgyd = 0.0
    cavgy = 0.0
    cavgzd = 0.0
    cavgz = 0.0
    for i_loc = 1:4
        i_noded = 0.0
        i_node = i_cell_to_node[i_loc, i_cell]
        cavgxd = cavgxd + 0.25 * cd[1, i_node]
        cavgx = cavgx + c[1, i_node] / 4
        cavgyd = cavgyd + 0.25 * cd[2, i_node]
        cavgy = cavgy + c[2, i_node] / 4
        cavgzd = cavgzd + 0.25 * cd[3, i_node]
        cavgz = cavgz + c[3, i_node] / 4
    end
    for i_loc = 1:4
        i_noded = 0.0
        i_node = i_cell_to_node[i_loc, i_cell]
        vered = (-(1.0 / 3.0) * (c[1, i_node] * skx[i_loc, i_cell] + c[2, i_node] * sky[i_loc, i_cell] + c[3, i_node] * skz[i_loc, i_cell])) * ud[i_node] + (-(1.0 / 3.0) * u[i_node]) * (((skx[i_loc, i_cell] * cd[1, i_node] + c[1, i_node] * skxd[i_loc, i_cell]) + (sky[i_loc, i_cell] * cd[2, i_node] + c[2, i_node] * skyd[i_loc, i_cell])) + (skz[i_loc, i_cell] * cd[3, i_node] + c[3, i_node] * skzd[i_loc, i_cell]))
        vere = -(1.0 / 3.0) * u[i_node] * (c[1, i_node] * skx[i_loc, i_cell] + c[2, i_node] * sky[i_loc, i_cell] + c[3, i_node] * skz[i_loc, i_cell])
        red = (1.0 / cell_vol[i_cell]) * vered + -(vere / cell_vol[i_cell] ^ 2) * cell_vold[i_cell]
        re = vere / cell_vol[i_cell]
        aerexd = re * cavgxd + cavgx * red
        aerex = cavgx * re
        aereyd = re * cavgyd + cavgy * red
        aerey = cavgy * re
        aerezd = re * cavgzd + cavgz * red
        aerez = cavgz * re
        resid = (((0.5 - gamma[i_cell]) * vere) * -(0.25dtd) + (-(dt / 4) * vere) * -(gammad[i_cell])) + (-(dt / 4) * (0.5 - gamma[i_cell])) * vered
        resi = -(dt / 4) * (0.5 - gamma[i_cell]) * vere
        for i_k = 1:4
            aereskd = ((skx[i_k, i_cell] * aerexd + aerex * skxd[i_k, i_cell]) + (sky[i_k, i_cell] * aereyd + aerey * skyd[i_k, i_cell])) + (skz[i_k, i_cell] * aerezd + aerez * skzd[i_k, i_cell])
            aeresk = aerex * skx[i_k, i_cell] + aerey * sky[i_k, i_cell] + aerez * skz[i_k, i_cell]
            factord = aeresk * (0.3333333333333333 * ((2dt) * dtd)) + (dt ^ 2 / 3) * aereskd
            factor = (dt ^ 2 / 3) * aeresk
            auxresd = resid + (beta[i_cell] * factord + factor * betad[i_cell])
            auxres = resi + factor * beta[i_cell]
            auxres2d = gamma[i_cell] * factord + factor * gammad[i_cell]
            auxres2 = factor * gamma[i_cell]
            i_k_noded = 0.0
            i_k_node = i_cell_to_node[i_k, i_cell]
            CUDA.@atomic resd[i_k_node] += auxresd
            CUDA.@atomic res[i_k_node] += auxres
            CUDA.@atomic res2d[i_k_node] += auxres2d
            CUDA.@atomic res2[i_k_node] += auxres2
        end
    end
    return nothing
end

function cuda_kernel_ttgc_d_2!(i_node_perio, npernode_half, res, resd, resperio, resperiod)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(npernode_half - 1, 1) + 1
        return nothing
    end
    k = 1 + (__tid - 1)
    i1d = 0.0
    i1 = i_node_perio[k, 1]
    i2d = 0.0
    i2 = i_node_perio[k, 2]
    resperiod[k, 1] = resd[i1]
    resperio[k, 1] = res[i1]
    resperiod[k, 2] = resd[i2]
    resperio[k, 2] = res[i2]
    return nothing
end

function cuda_kernel_ttgc_d_3!(i_node_perio, npernode_half, res, resd, resperio, resperiod)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(npernode_half - 1, 1) + 1
        return nothing
    end
    k = 1 + (__tid - 1)
    i1d = 0.0
    i1 = i_node_perio[k, 1]
    i2d = 0.0
    i2 = i_node_perio[k, 2]
    CUDA.@atomic resd[i1] += resperiod[k, 2]
    CUDA.@atomic res[i1] += resperio[k, 2]
    CUDA.@atomic resd[i2] += resperiod[k, 1]
    CUDA.@atomic res[i2] += resperio[k, 1]
    return nothing
end

function cuda_kernel_ttgc_d_4!(i_nnode, node_vol, node_vold, res, resd, up, upd)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(i_nnode - 1, 1) + 1
        return nothing
    end
    i_node = 1 + (__tid - 1)
    upd[i_node] = (1.0 / node_vol[i_node]) * resd[i_node] + -(res[i_node] / node_vol[i_node] ^ 2) * node_vold[i_node]
    up[i_node] = res[i_node] / node_vol[i_node]
    return nothing
end

function cuda_kernel_ttgc_d_5!(i_nnode, mup, mupd)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(i_nnode - 1, 1) + 1
        return nothing
    end
    i_node = 1 + (__tid - 1)
    mupd[i_node] = 0.0
    mup[i_node] = 0.0
    return nothing
end

function cuda_kernel_ttgc_d_6!(cell_vol, cell_vold, i_cell_to_node, i_ncell, mup, mupd, up, upd)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(i_ncell - 1, 1) + 1
        return nothing
    end
    i_cell = 1 + (__tid - 1)
    i_node1d = 0.0
    i_node1 = i_cell_to_node[1, i_cell]
    i_node2d = 0.0
    i_node2 = i_cell_to_node[2, i_cell]
    i_node3d = 0.0
    i_node3 = i_cell_to_node[3, i_cell]
    i_node4d = 0.0
    i_node4 = i_cell_to_node[4, i_cell]
    auxud = ((upd[i_node1] + upd[i_node2]) + upd[i_node3]) + upd[i_node4]
    auxu = up[i_node1] + up[i_node2] + up[i_node3] + up[i_node4]
    CUDA.@atomic mupd[i_node1] += 0.05 * (cell_vol[i_cell] * (auxud + upd[i_node1]) + (auxu + up[i_node1]) * cell_vold[i_cell])
    CUDA.@atomic mup[i_node1] += ((auxu + up[i_node1]) * cell_vol[i_cell]) / 20
    CUDA.@atomic mupd[i_node2] += 0.05 * (cell_vol[i_cell] * (auxud + upd[i_node2]) + (auxu + up[i_node2]) * cell_vold[i_cell])
    CUDA.@atomic mup[i_node2] += ((auxu + up[i_node2]) * cell_vol[i_cell]) / 20
    CUDA.@atomic mupd[i_node3] += 0.05 * (cell_vol[i_cell] * (auxud + upd[i_node3]) + (auxu + up[i_node3]) * cell_vold[i_cell])
    CUDA.@atomic mup[i_node3] += ((auxu + up[i_node3]) * cell_vol[i_cell]) / 20
    CUDA.@atomic mupd[i_node4] += 0.05 * (cell_vol[i_cell] * (auxud + upd[i_node4]) + (auxu + up[i_node4]) * cell_vold[i_cell])
    CUDA.@atomic mup[i_node4] += ((auxu + up[i_node4]) * cell_vol[i_cell]) / 20
    return nothing
end

function cuda_kernel_ttgc_d_7!(i_node_perio, mup, mupd, npernode_half, resperio, resperiod)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(npernode_half - 1, 1) + 1
        return nothing
    end
    k = 1 + (__tid - 1)
    i1d = 0.0
    i1 = i_node_perio[k, 1]
    i2d = 0.0
    i2 = i_node_perio[k, 2]
    resperiod[k, 1] = mupd[i1]
    resperio[k, 1] = mup[i1]
    resperiod[k, 2] = mupd[i2]
    resperio[k, 2] = mup[i2]
    return nothing
end

function cuda_kernel_ttgc_d_8!(i_node_perio, mup, mupd, npernode_half, resperio, resperiod)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(npernode_half - 1, 1) + 1
        return nothing
    end
    k = 1 + (__tid - 1)
    i1d = 0.0
    i1 = i_node_perio[k, 1]
    i2d = 0.0
    i2 = i_node_perio[k, 2]
    CUDA.@atomic mupd[i1] += resperiod[k, 2]
    CUDA.@atomic mup[i1] += resperio[k, 2]
    CUDA.@atomic mupd[i2] += resperiod[k, 1]
    CUDA.@atomic mup[i2] += resperio[k, 1]
    return nothing
end

function cuda_kernel_ttgc_d_9!(i_nnode, mup, mupd, node_vol, node_vold, res, resd, up, upd)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(i_nnode - 1, 1) + 1
        return nothing
    end
    i_node = 1 + (__tid - 1)
    upd[i_node] = upd[i_node] + ((1.0 / node_vol[i_node]) * (resd[i_node] + -(mupd[i_node])) + -((res[i_node] - mup[i_node]) / node_vol[i_node] ^ 2) * node_vold[i_node])
    up[i_node] = up[i_node] + (res[i_node] - mup[i_node]) / node_vol[i_node]
    return nothing
end

function cuda_kernel_ttgc_d_10!(c, cd, dt, dtd, i_cell_to_node, i_ncell, res2, res2d, skx, skxd, sky, skyd, u, ud, up, upd)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(i_ncell - 1, 1) + 1
        return nothing
    end
    i_cell = 1 + (__tid - 1)
    for i_loc = 1:4
        i_noded = 0.0
        i_node = i_cell_to_node[i_loc, i_cell]
        vered = (-(1.0 / 3.0) * (c[1, i_node] * skx[i_loc, i_cell] + c[2, i_node] * sky[i_loc, i_cell])) * (ud[i_node] + upd[i_node]) + (-(1.0 / 3.0) * (u[i_node] + up[i_node])) * ((skx[i_loc, i_cell] * cd[1, i_node] + c[1, i_node] * skxd[i_loc, i_cell]) + (sky[i_loc, i_cell] * cd[2, i_node] + c[2, i_node] * skyd[i_loc, i_cell]))
        vere = -(1.0 / 3.0) * (u[i_node] + up[i_node]) * (c[1, i_node] * skx[i_loc, i_cell] + c[2, i_node] * sky[i_loc, i_cell])
        resid = vere * -(0.25dtd) + -(dt / 4) * vered
        resi = -(dt / 4) * vere
        for i_k = 1:4
            i_k_noded = 0.0
            i_k_node = i_cell_to_node[i_k, i_cell]
            CUDA.@atomic res2d[i_k_node] += resid
            CUDA.@atomic res2[i_k_node] += resi
        end
    end
    return nothing
end

function cuda_kernel_ttgc_d_11!(i_node_perio, npernode_half, res2, res2d, resperio, resperiod)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(npernode_half - 1, 1) + 1
        return nothing
    end
    k = 1 + (__tid - 1)
    i1d = 0.0
    i1 = i_node_perio[k, 1]
    i2d = 0.0
    i2 = i_node_perio[k, 2]
    resperiod[k, 1] = res2d[i1]
    resperio[k, 1] = res2[i1]
    resperiod[k, 2] = res2d[i2]
    resperio[k, 2] = res2[i2]
    return nothing
end

function cuda_kernel_ttgc_d_12!(i_node_perio, npernode_half, res2, res2d, resperio, resperiod)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(npernode_half - 1, 1) + 1
        return nothing
    end
    k = 1 + (__tid - 1)
    i1d = 0.0
    i1 = i_node_perio[k, 1]
    i2d = 0.0
    i2 = i_node_perio[k, 2]
    CUDA.@atomic res2d[i1] += resperiod[k, 2]
    CUDA.@atomic res2[i1] += resperio[k, 2]
    CUDA.@atomic res2d[i2] += resperiod[k, 1]
    CUDA.@atomic res2[i2] += resperio[k, 1]
    return nothing
end

function cuda_kernel_ttgc_d_13!(i_nnode, node_vol, node_vold, res2, res2d, up, upd)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(i_nnode - 1, 1) + 1
        return nothing
    end
    i_node = 1 + (__tid - 1)
    upd[i_node] = (1.0 / node_vol[i_node]) * res2d[i_node] + -(res2[i_node] / node_vol[i_node] ^ 2) * node_vold[i_node]
    up[i_node] = res2[i_node] / node_vol[i_node]
    return nothing
end

function cuda_kernel_ttgc_d_14!(i_nnode, mup, mupd)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(i_nnode - 1, 1) + 1
        return nothing
    end
    i_node = 1 + (__tid - 1)
    mupd[i_node] = 0.0
    mup[i_node] = 0.0
    return nothing
end

function cuda_kernel_ttgc_d_15!(cell_vol, cell_vold, i_cell_to_node, i_ncell, mup, mupd, up, upd)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(i_ncell - 1, 1) + 1
        return nothing
    end
    i_cell = 1 + (__tid - 1)
    i_node1d = 0.0
    i_node1 = i_cell_to_node[1, i_cell]
    i_node2d = 0.0
    i_node2 = i_cell_to_node[2, i_cell]
    i_node3d = 0.0
    i_node3 = i_cell_to_node[3, i_cell]
    i_node4d = 0.0
    i_node4 = i_cell_to_node[4, i_cell]
    auxud = ((upd[i_node1] + upd[i_node2]) + upd[i_node3]) + upd[i_node4]
    auxu = up[i_node1] + up[i_node2] + up[i_node3] + up[i_node4]
    CUDA.@atomic mupd[i_node1] += 0.05 * (cell_vol[i_cell] * (auxud + upd[i_node1]) + (auxu + up[i_node1]) * cell_vold[i_cell])
    CUDA.@atomic mup[i_node1] += ((auxu + up[i_node1]) * cell_vol[i_cell]) / 20
    CUDA.@atomic mupd[i_node2] += 0.05 * (cell_vol[i_cell] * (auxud + upd[i_node2]) + (auxu + up[i_node2]) * cell_vold[i_cell])
    CUDA.@atomic mup[i_node2] += ((auxu + up[i_node2]) * cell_vol[i_cell]) / 20
    CUDA.@atomic mupd[i_node3] += 0.05 * (cell_vol[i_cell] * (auxud + upd[i_node3]) + (auxu + up[i_node3]) * cell_vold[i_cell])
    CUDA.@atomic mup[i_node3] += ((auxu + up[i_node3]) * cell_vol[i_cell]) / 20
    CUDA.@atomic mupd[i_node4] += 0.05 * (cell_vol[i_cell] * (auxud + upd[i_node4]) + (auxu + up[i_node4]) * cell_vold[i_cell])
    CUDA.@atomic mup[i_node4] += ((auxu + up[i_node4]) * cell_vol[i_cell]) / 20
    return nothing
end

function cuda_kernel_ttgc_d_16!(i_node_perio, mup, mupd, npernode_half, resperio, resperiod)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(npernode_half - 1, 1) + 1
        return nothing
    end
    k = 1 + (__tid - 1)
    i1d = 0.0
    i1 = i_node_perio[k, 1]
    i2d = 0.0
    i2 = i_node_perio[k, 2]
    resperiod[k, 1] = mupd[i1]
    resperio[k, 1] = mup[i1]
    resperiod[k, 2] = mupd[i2]
    resperio[k, 2] = mup[i2]
    return nothing
end

function cuda_kernel_ttgc_d_17!(i_node_perio, mup, mupd, npernode_half, resperio, resperiod)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(npernode_half - 1, 1) + 1
        return nothing
    end
    k = 1 + (__tid - 1)
    i1d = 0.0
    i1 = i_node_perio[k, 1]
    i2d = 0.0
    i2 = i_node_perio[k, 2]
    CUDA.@atomic mupd[i1] += resperiod[k, 2]
    CUDA.@atomic mup[i1] += resperio[k, 2]
    CUDA.@atomic mupd[i2] += resperiod[k, 1]
    CUDA.@atomic mup[i2] += resperio[k, 1]
    return nothing
end

function cuda_kernel_ttgc_d_18!(i_nnode, mup, mupd, node_vol, node_vold, res2, res2d, up, upd)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(i_nnode - 1, 1) + 1
        return nothing
    end
    i_node = 1 + (__tid - 1)
    upd[i_node] = upd[i_node] + ((1.0 / node_vol[i_node]) * (res2d[i_node] + -(mupd[i_node])) + -((res2[i_node] - mup[i_node]) / node_vol[i_node] ^ 2) * node_vold[i_node])
    up[i_node] = up[i_node] + (res2[i_node] - mup[i_node]) / node_vol[i_node]
    return nothing
end

function cuda_kernel_ttgc_d_19!(i_nnode, loss, lossd, u, ud, up, upd, uref, urefd)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(i_nnode - 1, 1) + 1
        return nothing
    end
    i_seq_node = 1 + (__tid - 1)
    CUDA.@atomic lossd[1] += (2 * ((u[i_seq_node] + up[i_seq_node]) - uref[i_seq_node])) * ((ud[i_seq_node] + upd[i_seq_node]) + -(urefd[i_seq_node]))
    CUDA.@atomic loss[1] += ((u[i_seq_node] + up[i_seq_node]) - uref[i_seq_node]) ^ 2
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

function cuda_kernel_ttgc_19!(i_nnode, loss, u, up, uref)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(i_nnode - 1, 1) + 1
        return nothing
    end
    i_seq_node = 1 + (__tid - 1)
    CUDA.@atomic loss[1] += ((u[i_seq_node] + up[i_seq_node]) - uref[i_seq_node]) ^ 2
    return nothing
end

function ttgc_d_cuda(u, ud, uref, urefd, i_cell_to_node, cell_vol, cell_vold, node_vol, node_vold, skx, skxd, sky, skyd, skz, skzd, i_ncell, i_nnode, c, cd, dt, dtd, beta, betad, gamma, gammad, i_njac, res, resd, res2, res2d, up, upd, mup, mupd, npernode_half, resperio, resperiod, i_node_perio, loss, lossd)
    nthread_per_block = 256
    @cuda threads = nthread_per_block blocks = cld(div(i_ncell - 1, 1) + 1, nthread_per_block) cuda_kernel_ttgc_d_1!(beta, betad, c, cd, cell_vol, cell_vold, dt, dtd, gamma, gammad, i_cell_to_node, i_ncell, res, res2, res2d, resd, skx, skxd, sky, skyd, skz, skzd, u, ud)
    @cuda threads = nthread_per_block blocks = cld(div(npernode_half - 1, 1) + 1, nthread_per_block) cuda_kernel_ttgc_d_2!(i_node_perio, npernode_half, res, resd, resperio, resperiod)
    @cuda threads = nthread_per_block blocks = cld(div(npernode_half - 1, 1) + 1, nthread_per_block) cuda_kernel_ttgc_d_3!(i_node_perio, npernode_half, res, resd, resperio, resperiod)
    @cuda threads = nthread_per_block blocks = cld(div(i_nnode - 1, 1) + 1, nthread_per_block) cuda_kernel_ttgc_d_4!(i_nnode, node_vol, node_vold, res, resd, up, upd)
    for i_seq_ = 1:i_njac
        @cuda threads = nthread_per_block blocks = cld(div(i_nnode - 1, 1) + 1, nthread_per_block) cuda_kernel_ttgc_d_5!(i_nnode, mup, mupd)
        @cuda threads = nthread_per_block blocks = cld(div(i_ncell - 1, 1) + 1, nthread_per_block) cuda_kernel_ttgc_d_6!(cell_vol, cell_vold, i_cell_to_node, i_ncell, mup, mupd, up, upd)
        @cuda threads = nthread_per_block blocks = cld(div(npernode_half - 1, 1) + 1, nthread_per_block) cuda_kernel_ttgc_d_7!(i_node_perio, mup, mupd, npernode_half, resperio, resperiod)
        @cuda threads = nthread_per_block blocks = cld(div(npernode_half - 1, 1) + 1, nthread_per_block) cuda_kernel_ttgc_d_8!(i_node_perio, mup, mupd, npernode_half, resperio, resperiod)
        @cuda threads = nthread_per_block blocks = cld(div(i_nnode - 1, 1) + 1, nthread_per_block) cuda_kernel_ttgc_d_9!(i_nnode, mup, mupd, node_vol, node_vold, res, resd, up, upd)
    end
    @cuda threads = nthread_per_block blocks = cld(div(i_ncell - 1, 1) + 1, nthread_per_block) cuda_kernel_ttgc_d_10!(c, cd, dt, dtd, i_cell_to_node, i_ncell, res2, res2d, skx, skxd, sky, skyd, u, ud, up, upd)
    @cuda threads = nthread_per_block blocks = cld(div(npernode_half - 1, 1) + 1, nthread_per_block) cuda_kernel_ttgc_d_11!(i_node_perio, npernode_half, res2, res2d, resperio, resperiod)
    @cuda threads = nthread_per_block blocks = cld(div(npernode_half - 1, 1) + 1, nthread_per_block) cuda_kernel_ttgc_d_12!(i_node_perio, npernode_half, res2, res2d, resperio, resperiod)
    @cuda threads = nthread_per_block blocks = cld(div(i_nnode - 1, 1) + 1, nthread_per_block) cuda_kernel_ttgc_d_13!(i_nnode, node_vol, node_vold, res2, res2d, up, upd)
    for i_seq_ = 1:i_njac
        @cuda threads = nthread_per_block blocks = cld(div(i_nnode - 1, 1) + 1, nthread_per_block) cuda_kernel_ttgc_d_14!(i_nnode, mup, mupd)
        @cuda threads = nthread_per_block blocks = cld(div(i_ncell - 1, 1) + 1, nthread_per_block) cuda_kernel_ttgc_d_15!(cell_vol, cell_vold, i_cell_to_node, i_ncell, mup, mupd, up, upd)
        @cuda threads = nthread_per_block blocks = cld(div(npernode_half - 1, 1) + 1, nthread_per_block) cuda_kernel_ttgc_d_16!(i_node_perio, mup, mupd, npernode_half, resperio, resperiod)
        @cuda threads = nthread_per_block blocks = cld(div(npernode_half - 1, 1) + 1, nthread_per_block) cuda_kernel_ttgc_d_17!(i_node_perio, mup, mupd, npernode_half, resperio, resperiod)
        @cuda threads = nthread_per_block blocks = cld(div(i_nnode - 1, 1) + 1, nthread_per_block) cuda_kernel_ttgc_d_18!(i_nnode, mup, mupd, node_vol, node_vold, res2, res2d, up, upd)
    end
    @cuda threads = nthread_per_block blocks = cld(div(i_nnode - 1, 1) + 1, nthread_per_block) cuda_kernel_ttgc_d_19!(i_nnode, loss, lossd, u, ud, up, upd, uref, urefd)
    return nothing
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
    @cuda threads = nthread_per_block blocks = cld(div(i_nnode - 1, 1) + 1, nthread_per_block) cuda_kernel_ttgc_19!(i_nnode, loss, u, up, uref)
    return nothing
end
