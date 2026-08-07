using CUDA
function func(u, i_cell_to_node, cell_vol, node_vol, skx, sky, skz, i_ncell, i_nnode, c, dt, beta, gamma, i_njac, res, res2, up, mup)
    for i_cell = 1:i_ncell
        cavgx = 0.0f0
        cavgy = 0.0f0
        cavgz = 0.0f0
        for i_loc = 1:4
            i_node = i_cell_to_node[i_loc, i_cell]
            cavgx = cavgx + c[1, i_node] / 4
            cavgy = cavgy + c[2, i_node] / 4
            cavgz = cavgz + c[3, i_node] / 4
        end
        for i_loc = 1:4
            i_node = i_cell_to_node[i_loc, i_cell]
            vere = -(1.0f0 / 3.0f0) * u[i_node] * (c[1, i_node] * skx[i_loc, i_cell] + c[2, i_node] * sky[i_loc, i_cell] + c[3, i_node] * skz[i_loc, i_cell])
            re = vere / cell_vol[i_cell]
            aerex = cavgx * re
            aerey = cavgy * re
            aerez = cavgz * re
            resi = -(dt / 4) * (0.5f0 - gamma[i_cell]) * vere
            for i_k = 1:4
                aeresk = aerex * skx[i_k, i_cell] + aerey * sky[i_k, i_cell] + aerez * skz[i_k, i_cell]
                factor = (dt ^ 2 / 3) * aeresk
                auxres = resi + factor * beta[i_cell]
                auxres2 = factor * gamma[i_cell]
                i_k_node = i_cell_to_node[i_k, i_cell]
                res[i_k_node] = res[i_k_node] + auxres
                res2[i_k_node] = res2[i_k_node] + auxres2
            end
        end
    end
    for i_node = 1:i_nnode
        up[i_node] = res[i_node] / node_vol[i_node]
    end
    for i_seq_ = 1:i_njac
        for i_node = 1:i_nnode
            mup[i_node] = 0.0f0
        end
        for i_cell = 1:i_ncell
            i_node1 = i_cell_to_node[1, i_cell]
            i_node2 = i_cell_to_node[2, i_cell]
            i_node3 = i_cell_to_node[3, i_cell]
            i_node4 = i_cell_to_node[4, i_cell]
            auxu = up[i_node1] + up[i_node2] + up[i_node3] + up[i_node4]
            mup[i_node1] = mup[i_node1] + ((auxu + up[i_node1]) * cell_vol[i_cell]) / 20
            mup[i_node2] = mup[i_node2] + ((auxu + up[i_node2]) * cell_vol[i_cell]) / 20
            mup[i_node3] = mup[i_node3] + ((auxu + up[i_node3]) * cell_vol[i_cell]) / 20
            mup[i_node4] = mup[i_node4] + ((auxu + up[i_node4]) * cell_vol[i_cell]) / 20
        end
        for i_node = 1:i_nnode
            up[i_node] = up[i_node] + (res[i_node] - mup[i_node]) / node_vol[i_node]
        end
    end
end
function func_cuda(u, i_cell_to_node, cell_vol, node_vol, skx, sky, skz, i_ncell, i_nnode, c, dt, beta, gamma, i_njac, res, res2, up, mup)
    nthread_per_block = 256
    @cuda threads = nthread_per_block blocks = ceil(Int, ((i_ncell - 1) + 1) / nthread_per_block) func_loop1!(u, i_cell_to_node, cell_vol, node_vol, skx, sky, skz, i_ncell, i_nnode, c, dt, beta, gamma, i_njac, res, res2, up, mup)
    @cuda threads = nthread_per_block blocks = ceil(Int, ((i_nnode - 1) + 1) / nthread_per_block) func_loop2!(u, i_cell_to_node, cell_vol, node_vol, skx, sky, skz, i_ncell, i_nnode, c, dt, beta, gamma, i_njac, res, res2, up, mup)
    for i_seq_ = 1:i_njac
        @cuda threads = nthread_per_block blocks = ceil(Int, ((i_nnode - 1) + 1) / nthread_per_block) func_loop3!(u, i_cell_to_node, cell_vol, node_vol, skx, sky, skz, i_ncell, i_nnode, c, dt, beta, gamma, i_njac, res, res2, up, mup)
        @cuda threads = nthread_per_block blocks = ceil(Int, ((i_ncell - 1) + 1) / nthread_per_block) func_loop4!(u, i_cell_to_node, cell_vol, node_vol, skx, sky, skz, i_ncell, i_nnode, c, dt, beta, gamma, i_njac, res, res2, up, mup)
        @cuda threads = nthread_per_block blocks = ceil(Int, ((i_nnode - 1) + 1) / nthread_per_block) func_loop5!(u, i_cell_to_node, cell_vol, node_vol, skx, sky, skz, i_ncell, i_nnode, c, dt, beta, gamma, i_njac, res, res2, up, mup)
    end
end
function func_loop1!(u, i_cell_to_node, cell_vol, node_vol, skx, sky, skz, i_ncell, i_nnode, c, dt, beta, gamma, i_njac, res, res2, up, mup)
    i_cell = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    i_cell < 1 && return nothing
    i_cell > i_ncell && return nothing
    cavgx = 0.0f0
    cavgy = 0.0f0
    cavgz = 0.0f0
    for i_loc = 1:4
        i_node = i_cell_to_node[i_loc, i_cell]
        cavgx = cavgx + c[1, i_node] / 4
        cavgy = cavgy + c[2, i_node] / 4
        cavgz = cavgz + c[3, i_node] / 4
    end
    for i_loc = 1:4
        i_node = i_cell_to_node[i_loc, i_cell]
        vere = -(1.0f0 / 3.0f0) * u[i_node] * (c[1, i_node] * skx[i_loc, i_cell] + c[2, i_node] * sky[i_loc, i_cell] + c[3, i_node] * skz[i_loc, i_cell])
        re = vere / cell_vol[i_cell]
        aerex = cavgx * re
        aerey = cavgy * re
        aerez = cavgz * re
        resi = -(dt / 4) * (0.5f0 - gamma[i_cell]) * vere
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
function func_loop2!(u, i_cell_to_node, cell_vol, node_vol, skx, sky, skz, i_ncell, i_nnode, c, dt, beta, gamma, i_njac, res, res2, up, mup)
    i_node = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    i_node < 1 && return nothing
    i_node > i_nnode && return nothing
    up[i_node] = res[i_node] / node_vol[i_node]
    return nothing
end
function func_loop3!(u, i_cell_to_node, cell_vol, node_vol, skx, sky, skz, i_ncell, i_nnode, c, dt, beta, gamma, i_njac, res, res2, up, mup)
    i_node = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    i_node < 1 && return nothing
    i_node > i_nnode && return nothing
    mup[i_node] = 0.0f0
    return nothing
end
function func_loop4!(u, i_cell_to_node, cell_vol, node_vol, skx, sky, skz, i_ncell, i_nnode, c, dt, beta, gamma, i_njac, res, res2, up, mup)
    i_cell = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    i_cell < 1 && return nothing
    i_cell > i_ncell && return nothing
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
function func_loop5!(u, i_cell_to_node, cell_vol, node_vol, skx, sky, skz, i_ncell, i_nnode, c, dt, beta, gamma, i_njac, res, res2, up, mup)
    i_node = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    i_node < 1 && return nothing
    i_node > i_nnode && return nothing
    up[i_node] = up[i_node] + (res[i_node] - mup[i_node]) / node_vol[i_node]
    return nothing
end
