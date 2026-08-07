import Pkg
haskey(Pkg.project().dependencies, "CUDA") || Pkg.add("CUDA")
using CUDA
CUDA.allowscalar(false)

function cuda_kernel_func_1!(beta, c, cell_vol, dt, gamma, i_cell_to_node, i_ncell, res, res2, skx, sky, skz, u)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(i_ncell - 1, 1) + 1
        return nothing
    end
    i_cell = 1 + (__tid - 1)
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

function cuda_kernel_func_2!(i_nnode, node_vol, res, up)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(i_nnode - 1, 1) + 1
        return nothing
    end
    i_node = 1 + (__tid - 1)
    up[i_node] = res[i_node] / node_vol[i_node]
    return nothing
end

function cuda_kernel_func_3!(i_nnode, mup)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(i_nnode - 1, 1) + 1
        return nothing
    end
    i_node = 1 + (__tid - 1)
    mup[i_node] = 0.0f0
    return nothing
end

function cuda_kernel_func_4!(cell_vol, i_cell_to_node, i_ncell, mup, up)
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

function cuda_kernel_func_5!(i_nnode, mup, node_vol, res, up)
    __tid = ((blockIdx()).x - 1) * (blockDim()).x + (threadIdx()).x
    if __tid > div(i_nnode - 1, 1) + 1
        return nothing
    end
    i_node = 1 + (__tid - 1)
    up[i_node] = up[i_node] + (res[i_node] - mup[i_node]) / node_vol[i_node]
    return nothing
end

function func_cuda(u, i_cell_to_node, cell_vol, node_vol, skx, sky, skz, i_ncell, i_nnode, c, dt, beta, gamma, i_njac, res, res2, up, mup)
    nthread_per_block = 256
    @cuda threads = nthread_per_block blocks = cld(div(i_ncell - 1, 1) + 1, nthread_per_block) cuda_kernel_func_1!(beta, c, cell_vol, dt, gamma, i_cell_to_node, i_ncell, res, res2, skx, sky, skz, u)
    @cuda threads = nthread_per_block blocks = cld(div(i_nnode - 1, 1) + 1, nthread_per_block) cuda_kernel_func_2!(i_nnode, node_vol, res, up)
    for i_seq_ = 1:i_njac
        @cuda threads = nthread_per_block blocks = cld(div(i_nnode - 1, 1) + 1, nthread_per_block) cuda_kernel_func_3!(i_nnode, mup)
        @cuda threads = nthread_per_block blocks = cld(div(i_ncell - 1, 1) + 1, nthread_per_block) cuda_kernel_func_4!(cell_vol, i_cell_to_node, i_ncell, mup, up)
        @cuda threads = nthread_per_block blocks = cld(div(i_nnode - 1, 1) + 1, nthread_per_block) cuda_kernel_func_5!(i_nnode, mup, node_vol, res, up)
    end
    return nothing
end

function func(u, i_cell_to_node, cell_vol, node_vol, skx, sky, skz, i_ncell, i_nnode, c, dt, beta, gamma, i_njac, res, res2, up, mup)
    #= none:2 =#
    #= none:3 =#
    for i_cell = 1:i_ncell
        #= none:4 =#
        cavgx = 0.0f0
        #= none:5 =#
        cavgy = 0.0f0
        #= none:6 =#
        cavgz = 0.0f0
        #= none:7 =#
        for i_loc = 1:4
            #= none:8 =#
            i_node = i_cell_to_node[i_loc, i_cell]
            #= none:9 =#
            cavgx = cavgx + c[1, i_node] / 4
            #= none:10 =#
            cavgy = cavgy + c[2, i_node] / 4
            #= none:11 =#
            cavgz = cavgz + c[3, i_node] / 4
            #= none:12 =#
        end
        #= none:13 =#
        for i_loc = 1:4
            #= none:14 =#
            i_node = i_cell_to_node[i_loc, i_cell]
            #= none:15 =#
            vere = -(1.0f0 / 3.0f0) * u[i_node] * (c[1, i_node] * skx[i_loc, i_cell] + c[2, i_node] * sky[i_loc, i_cell] + c[3, i_node] * skz[i_loc, i_cell])
            #= none:16 =#
            re = vere / cell_vol[i_cell]
            #= none:17 =#
            aerex = cavgx * re
            #= none:18 =#
            aerey = cavgy * re
            #= none:19 =#
            aerez = cavgz * re
            #= none:20 =#
            resi = -(dt / 4) * (0.5f0 - gamma[i_cell]) * vere
            #= none:21 =#
            for i_k = 1:4
                #= none:22 =#
                aeresk = aerex * skx[i_k, i_cell] + aerey * sky[i_k, i_cell] + aerez * skz[i_k, i_cell]
                #= none:23 =#
                factor = (dt ^ 2 / 3) * aeresk
                #= none:24 =#
                auxres = resi + factor * beta[i_cell]
                #= none:25 =#
                auxres2 = factor * gamma[i_cell]
                #= none:26 =#
                i_k_node = i_cell_to_node[i_k, i_cell]
                #= none:27 =#
                res[i_k_node] = res[i_k_node] + auxres
                #= none:28 =#
                res2[i_k_node] = res2[i_k_node] + auxres2
                #= none:29 =#
            end
            #= none:30 =#
        end
        #= none:31 =#
    end
    #= none:32 =#
    for i_node = 1:i_nnode
        #= none:33 =#
        up[i_node] = res[i_node] / node_vol[i_node]
        #= none:34 =#
    end
    #= none:35 =#
    for i_seq_ = 1:i_njac
        #= none:36 =#
        for i_node = 1:i_nnode
            #= none:37 =#
            mup[i_node] = 0.0f0
            #= none:38 =#
        end
        #= none:39 =#
        for i_cell = 1:i_ncell
            #= none:40 =#
            i_node1 = i_cell_to_node[1, i_cell]
            #= none:41 =#
            i_node2 = i_cell_to_node[2, i_cell]
            #= none:42 =#
            i_node3 = i_cell_to_node[3, i_cell]
            #= none:43 =#
            i_node4 = i_cell_to_node[4, i_cell]
            #= none:44 =#
            auxu = up[i_node1] + up[i_node2] + up[i_node3] + up[i_node4]
            #= none:45 =#
            mup[i_node1] = mup[i_node1] + ((auxu + up[i_node1]) * cell_vol[i_cell]) / 20
            #= none:46 =#
            mup[i_node2] = mup[i_node2] + ((auxu + up[i_node2]) * cell_vol[i_cell]) / 20
            #= none:47 =#
            mup[i_node3] = mup[i_node3] + ((auxu + up[i_node3]) * cell_vol[i_cell]) / 20
            #= none:48 =#
            mup[i_node4] = mup[i_node4] + ((auxu + up[i_node4]) * cell_vol[i_cell]) / 20
            #= none:49 =#
        end
        #= none:50 =#
        for i_node = 1:i_nnode
            #= none:51 =#
            up[i_node] = up[i_node] + (res[i_node] - mup[i_node]) / node_vol[i_node]
            #= none:52 =#
        end
        #= none:53 =#
    end
end
