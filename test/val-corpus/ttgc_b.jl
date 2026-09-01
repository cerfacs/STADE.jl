function initstacks_ttgc_b(i_ncell, i_njac, i_nnode)
    vere_stack = Vector{Float64}(undef, ((max(0, div(i_ncell - 1, 1) + 1) * max(0, div(4 - 1, 1) + 1) + max(0, div(i_ncell - 1, 1) + 1)) + max(0, div(i_ncell - 1, 1) + 1) * max(0, div(4 - 1, 1) + 1)) + max(0, div(i_ncell - 1, 1) + 1))
    mup_stack = Vector{Float64}(undef, ((max(0, div(i_njac - 1, 1) + 1) * max(0, div(i_nnode - 1, 1) + 1) + max(0, div(i_njac - 1, 1) + 1) * max(0, div(i_ncell - 1, 1) + 1) * max(0, div(4 - 1, 1) + 1)) + max(0, div(i_njac - 1, 1) + 1) * max(0, div(i_nnode - 1, 1) + 1)) + max(0, div(i_njac - 1, 1) + 1) * max(0, div(i_ncell - 1, 1) + 1) * max(0, div(4 - 1, 1) + 1))
    up_stack = Vector{Float64}(undef, (max(0, div(i_njac - 1, 1) + 1) * max(0, div(i_nnode - 1, 1) + 1) + max(0, div(i_nnode - 1, 1) + 1)) + max(0, div(i_njac - 1, 1) + 1) * max(0, div(i_nnode - 1, 1) + 1))
    auxu_stack = Vector{Float64}(undef, max(0, div(i_njac - 1, 1) + 1) * max(0, div(i_ncell - 1, 1) + 1) + max(0, div(i_njac - 1, 1) + 1) * max(0, div(i_ncell - 1, 1) + 1))
    return (vere_stack, mup_stack, up_stack, auxu_stack)
end

function ttgc_b(u, ub, uref, urefb, i_cell_to_node, cell_vol, cell_volb, node_vol, node_volb, skx, skxb, sky, skyb, skz, skzb, i_ncell, i_nnode, c, cb, dt, dtb, beta, betab, gamma, gammab, i_njac, res, resb, res2, res2b, up, upb, mup, mupb, loss, lossb, vere_stack, mup_stack, up_stack, auxu_stack)
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
    for i_cell = 1:i_ncell
        aeresk = 0.0
        aerex = 0.0
        aerey = 0.0
        aerez = 0.0
        factor = 0.0
        re = 0.0
        vere = 0.0
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
            aeresk = 0.0
            factor = 0.0
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
                res[i_k_node] = res[i_k_node] + auxres
                res2[i_k_node] = res2[i_k_node] + auxres2
            end
        end
    end
    for i_node = 1:i_nnode
        up[i_node] = res[i_node] / node_vol[i_node]
    end
    for i_ = 1:i_njac
        for i_node = 1:i_nnode
            __idx_mup_stack_0 = ((i_ - 1) * (div(i_nnode - 1, 1) + 1) + (i_node - 1)) + 1
            mup_stack[__idx_mup_stack_0] = mup[i_node]
            mup[i_node] = 0.0
        end
        for i_cell = 1:i_ncell
            auxu = 0.0
            for i_loc = 1:4
                i_lnode = i_cell_to_node[i_loc, i_cell]
                auxu = auxu + up[i_lnode]
            end
            for i_loc = 1:4
                i_lnode = i_cell_to_node[i_loc, i_cell]
                __idx_mup_stack_1 = max(0, div(i_njac - 1, 1) + 1) * max(0, div(i_nnode - 1, 1) + 1) + (((i_ - 1) * ((div(i_ncell - 1, 1) + 1) * (div(4 - 1, 1) + 1)) + (i_cell - 1) * (div(4 - 1, 1) + 1) + (i_loc - 1)) + 1)
                __cse_0 = mup[i_lnode]
                mup_stack[__idx_mup_stack_1] = __cse_0
                mup[i_lnode] = __cse_0 + ((auxu + up[i_lnode]) * cell_vol[i_cell]) / 20.0
            end
            __idx_auxu_stack_3 = ((i_ - 1) * (div(i_ncell - 1, 1) + 1) + (i_cell - 1)) + 1
            auxu_stack[__idx_auxu_stack_3] = auxu
        end
        for i_node = 1:i_nnode
            __idx_up_stack_0 = ((i_ - 1) * (div(i_nnode - 1, 1) + 1) + (i_node - 1)) + 1
            __cse_1 = up[i_node]
            up_stack[__idx_up_stack_0] = __cse_1
            up[i_node] = __cse_1 + (res[i_node] - mup[i_node]) / node_vol[i_node]
        end
    end
    for i_cell = 1:i_ncell
        vere = 0.0
        for i_loc = 1:4
            i_node = i_cell_to_node[i_loc, i_cell]
            __idx_vere_stack_1 = (max(0, div(i_ncell - 1, 1) + 1) * max(0, div(4 - 1, 1) + 1) + max(0, div(i_ncell - 1, 1) + 1)) + (((i_cell - 1) * (div(4 - 1, 1) + 1) + (i_loc - 1)) + 1)
            vere_stack[__idx_vere_stack_1] = vere
            vere = -(1.0 / 3.0) * (u[i_node] + up[i_node]) * (c[1, i_node] * skx[i_loc, i_cell] + c[2, i_node] * sky[i_loc, i_cell])
            resi = -(dt / 4) * vere
            for i_k = 1:4
                i_k_node = i_cell_to_node[i_k, i_cell]
                res2[i_k_node] = res2[i_k_node] + resi
            end
        end
        __idx_vere_stack_2 = ((max(0, div(i_ncell - 1, 1) + 1) * max(0, div(4 - 1, 1) + 1) + max(0, div(i_ncell - 1, 1) + 1)) + max(0, div(i_ncell - 1, 1) + 1) * max(0, div(4 - 1, 1) + 1)) + ((i_cell - 1) + 1)
        vere_stack[__idx_vere_stack_2] = vere
    end
    for i_node = 1:i_nnode
        __idx_up_stack_0 = max(0, div(i_njac - 1, 1) + 1) * max(0, div(i_nnode - 1, 1) + 1) + ((i_node - 1) + 1)
        up_stack[__idx_up_stack_0] = up[i_node]
        up[i_node] = res2[i_node] / node_vol[i_node]
    end
    for i_ = 1:i_njac
        for i_node = 1:i_nnode
            __idx_mup_stack_0 = (max(0, div(i_njac - 1, 1) + 1) * max(0, div(i_nnode - 1, 1) + 1) + max(0, div(i_njac - 1, 1) + 1) * max(0, div(i_ncell - 1, 1) + 1) * max(0, div(4 - 1, 1) + 1)) + (((i_ - 1) * (div(i_nnode - 1, 1) + 1) + (i_node - 1)) + 1)
            mup_stack[__idx_mup_stack_0] = mup[i_node]
            mup[i_node] = 0.0
        end
        for i_cell = 1:i_ncell
            auxu = 0.0
            for i_loc = 1:4
                i_lnode = i_cell_to_node[i_loc, i_cell]
                auxu = auxu + up[i_lnode]
            end
            for i_loc = 1:4
                i_lnode = i_cell_to_node[i_loc, i_cell]
                __idx_mup_stack_1 = ((max(0, div(i_njac - 1, 1) + 1) * max(0, div(i_nnode - 1, 1) + 1) + max(0, div(i_njac - 1, 1) + 1) * max(0, div(i_ncell - 1, 1) + 1) * max(0, div(4 - 1, 1) + 1)) + max(0, div(i_njac - 1, 1) + 1) * max(0, div(i_nnode - 1, 1) + 1)) + (((i_ - 1) * ((div(i_ncell - 1, 1) + 1) * (div(4 - 1, 1) + 1)) + (i_cell - 1) * (div(4 - 1, 1) + 1) + (i_loc - 1)) + 1)
                __cse_2 = mup[i_lnode]
                mup_stack[__idx_mup_stack_1] = __cse_2
                mup[i_lnode] = __cse_2 + ((auxu + up[i_lnode]) * cell_vol[i_cell]) / 20.0
            end
            __idx_auxu_stack_3 = max(0, div(i_njac - 1, 1) + 1) * max(0, div(i_ncell - 1, 1) + 1) + (((i_ - 1) * (div(i_ncell - 1, 1) + 1) + (i_cell - 1)) + 1)
            auxu_stack[__idx_auxu_stack_3] = auxu
        end
        for i_node = 1:i_nnode
            __idx_up_stack_0 = (max(0, div(i_njac - 1, 1) + 1) * max(0, div(i_nnode - 1, 1) + 1) + max(0, div(i_nnode - 1, 1) + 1)) + (((i_ - 1) * (div(i_nnode - 1, 1) + 1) + (i_node - 1)) + 1)
            __cse_3 = up[i_node]
            up_stack[__idx_up_stack_0] = __cse_3
            up[i_node] = __cse_3 + (res2[i_node] - mup[i_node]) / node_vol[i_node]
        end
    end
    for i_node2 = 1:i_nnode
        loss[1] = loss[1] + ((u[i_node2] + up[i_node2]) - uref[i_node2]) ^ 2
    end
    for i_node2 = i_nnode:-1:1
        __cse_4 = (2 * ((u[i_node2] + up[i_node2]) - uref[i_node2])) * lossb[1]
        ub[i_node2] = ub[i_node2] + __cse_4
        upb[i_node2] = upb[i_node2] + __cse_4
        urefb[i_node2] = urefb[i_node2] + -__cse_4
    end
    for i_ = i_njac:-1:1
        for i_node = i_nnode:-1:1
            __idx_up_stack_0 = (max(0, div(i_njac - 1, 1) + 1) * max(0, div(i_nnode - 1, 1) + 1) + max(0, div(i_nnode - 1, 1) + 1)) + (((i_ - 1) * (div(i_nnode - 1, 1) + 1) + (i_node - 1)) + 1)
            up[i_node] = up_stack[__idx_up_stack_0]
            __cse_5 = node_vol[i_node]
            __cse_6 = upb[i_node]
            __cse_7 = (1.0 / __cse_5) * __cse_6
            res2b[i_node] = res2b[i_node] + __cse_7
            mupb[i_node] = mupb[i_node] + -__cse_7
            node_volb[i_node] = node_volb[i_node] + -((res2[i_node] - mup[i_node]) / __cse_5 ^ 2) * __cse_6
        end
        for i_cell = i_ncell:-1:1
            __idx_auxu_stack_0 = max(0, div(i_njac - 1, 1) + 1) * max(0, div(i_ncell - 1, 1) + 1) + (((i_ - 1) * (div(i_ncell - 1, 1) + 1) + (i_cell - 1)) + 1)
            auxu = auxu_stack[__idx_auxu_stack_0]
            for i_loc = 4:-1:1
                i_lnode = i_cell_to_node[i_loc, i_cell]
                __idx_mup_stack_0 = ((max(0, div(i_njac - 1, 1) + 1) * max(0, div(i_nnode - 1, 1) + 1) + max(0, div(i_njac - 1, 1) + 1) * max(0, div(i_ncell - 1, 1) + 1) * max(0, div(4 - 1, 1) + 1)) + max(0, div(i_njac - 1, 1) + 1) * max(0, div(i_nnode - 1, 1) + 1)) + (((i_ - 1) * ((div(i_ncell - 1, 1) + 1) * (div(4 - 1, 1) + 1)) + (i_cell - 1) * (div(4 - 1, 1) + 1) + (i_loc - 1)) + 1)
                mup[i_lnode] = mup_stack[__idx_mup_stack_0]
                __cse_8 = 0.05 * mupb[i_lnode]
                __cse_9 = cell_vol[i_cell] * __cse_8
                auxub = auxub + __cse_9
                upb[i_lnode] = upb[i_lnode] + __cse_9
                cell_volb[i_cell] = cell_volb[i_cell] + (auxu + up[i_lnode]) * __cse_8
            end
            for i_loc = 1:4
                i_lnode = i_cell_to_node[i_loc, i_cell]
                auxu = auxu + up[i_lnode]
                i_lnode = i_cell_to_node[i_loc, i_cell]
                upb[i_lnode] = upb[i_lnode] + auxub
            end
            auxub = 0.0
        end
        for i_node = i_nnode:-1:1
            __idx_mup_stack_0 = (max(0, div(i_njac - 1, 1) + 1) * max(0, div(i_nnode - 1, 1) + 1) + max(0, div(i_njac - 1, 1) + 1) * max(0, div(i_ncell - 1, 1) + 1) * max(0, div(4 - 1, 1) + 1)) + (((i_ - 1) * (div(i_nnode - 1, 1) + 1) + (i_node - 1)) + 1)
            mup[i_node] = mup_stack[__idx_mup_stack_0]
            mupb[i_node] = 0.0
        end
    end
    for i_node = i_nnode:-1:1
        __idx_up_stack_0 = max(0, div(i_njac - 1, 1) + 1) * max(0, div(i_nnode - 1, 1) + 1) + ((i_node - 1) + 1)
        up[i_node] = up_stack[__idx_up_stack_0]
        __cse_10 = node_vol[i_node]
        __cse_11 = upb[i_node]
        res2b[i_node] = res2b[i_node] + (1.0 / __cse_10) * __cse_11
        node_volb[i_node] = node_volb[i_node] + -(res2[i_node] / __cse_10 ^ 2) * __cse_11
        upb[i_node] = 0.0
    end
    for i_cell = i_ncell:-1:1
        __idx_vere_stack_0 = ((max(0, div(i_ncell - 1, 1) + 1) * max(0, div(4 - 1, 1) + 1) + max(0, div(i_ncell - 1, 1) + 1)) + max(0, div(i_ncell - 1, 1) + 1) * max(0, div(4 - 1, 1) + 1)) + ((i_cell - 1) + 1)
        vere = vere_stack[__idx_vere_stack_0]
        for i_loc = 4:-1:1
            i_node = i_cell_to_node[i_loc, i_cell]
            for i_k = 4:-1:1
                i_k_node = i_cell_to_node[i_k, i_cell]
                resib = resib + res2b[i_k_node]
            end
            dtb = dtb + 0.25 * -(vere * resib)
            vereb = vereb + -(dt / 4) * resib
            resib = 0.0
            __idx_vere_stack_0 = (max(0, div(i_ncell - 1, 1) + 1) * max(0, div(4 - 1, 1) + 1) + max(0, div(i_ncell - 1, 1) + 1)) + (((i_cell - 1) * (div(4 - 1, 1) + 1) + (i_loc - 1)) + 1)
            vere = vere_stack[__idx_vere_stack_0]
            __cse_12 = c[1, i_node]
            __cse_13 = skx[i_loc, i_cell]
            __cse_14 = c[2, i_node]
            __cse_15 = sky[i_loc, i_cell]
            __cse_16 = (-(1.0 / 3.0) * (__cse_12 * __cse_13 + __cse_14 * __cse_15)) * vereb
            ub[i_node] = ub[i_node] + __cse_16
            upb[i_node] = upb[i_node] + __cse_16
            __cse_17 = (-(1.0 / 3.0) * (u[i_node] + up[i_node])) * vereb
            cb[1, i_node] = cb[1, i_node] + __cse_13 * __cse_17
            skxb[i_loc, i_cell] = skxb[i_loc, i_cell] + __cse_12 * __cse_17
            cb[2, i_node] = cb[2, i_node] + __cse_15 * __cse_17
            skyb[i_loc, i_cell] = skyb[i_loc, i_cell] + __cse_14 * __cse_17
            vereb = 0.0
        end
        vereb = 0.0
    end
    for i_ = i_njac:-1:1
        for i_node = i_nnode:-1:1
            __idx_up_stack_0 = ((i_ - 1) * (div(i_nnode - 1, 1) + 1) + (i_node - 1)) + 1
            up[i_node] = up_stack[__idx_up_stack_0]
            __cse_18 = node_vol[i_node]
            __cse_19 = upb[i_node]
            __cse_20 = (1.0 / __cse_18) * __cse_19
            resb[i_node] = resb[i_node] + __cse_20
            mupb[i_node] = mupb[i_node] + -__cse_20
            node_volb[i_node] = node_volb[i_node] + -((res[i_node] - mup[i_node]) / __cse_18 ^ 2) * __cse_19
        end
        for i_cell = i_ncell:-1:1
            __idx_auxu_stack_0 = ((i_ - 1) * (div(i_ncell - 1, 1) + 1) + (i_cell - 1)) + 1
            auxu = auxu_stack[__idx_auxu_stack_0]
            for i_loc = 4:-1:1
                i_lnode = i_cell_to_node[i_loc, i_cell]
                __idx_mup_stack_0 = max(0, div(i_njac - 1, 1) + 1) * max(0, div(i_nnode - 1, 1) + 1) + (((i_ - 1) * ((div(i_ncell - 1, 1) + 1) * (div(4 - 1, 1) + 1)) + (i_cell - 1) * (div(4 - 1, 1) + 1) + (i_loc - 1)) + 1)
                mup[i_lnode] = mup_stack[__idx_mup_stack_0]
                __cse_21 = 0.05 * mupb[i_lnode]
                __cse_22 = cell_vol[i_cell] * __cse_21
                auxub = auxub + __cse_22
                upb[i_lnode] = upb[i_lnode] + __cse_22
                cell_volb[i_cell] = cell_volb[i_cell] + (auxu + up[i_lnode]) * __cse_21
            end
            for i_loc = 1:4
                i_lnode = i_cell_to_node[i_loc, i_cell]
                auxu = auxu + up[i_lnode]
                i_lnode = i_cell_to_node[i_loc, i_cell]
                upb[i_lnode] = upb[i_lnode] + auxub
            end
            auxub = 0.0
        end
        for i_node = i_nnode:-1:1
            __idx_mup_stack_0 = ((i_ - 1) * (div(i_nnode - 1, 1) + 1) + (i_node - 1)) + 1
            mup[i_node] = mup_stack[__idx_mup_stack_0]
            mupb[i_node] = 0.0
        end
    end
    for i_node = i_nnode:-1:1
        __cse_23 = node_vol[i_node]
        __cse_24 = upb[i_node]
        resb[i_node] = resb[i_node] + (1.0 / __cse_23) * __cse_24
        node_volb[i_node] = node_volb[i_node] + -(res[i_node] / __cse_23 ^ 2) * __cse_24
        upb[i_node] = 0.0
    end
    for i_cell = 1:i_ncell
        aeresk = 0.0
        aerex = 0.0
        aerey = 0.0
        aerez = 0.0
        factor = 0.0
        re = 0.0
        vere = 0.0
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
            aeresk = 0.0
            factor = 0.0
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
            end
        end
        for i_loc = 1:4
            aeresk = 0.0
            factor = 0.0
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
            end
            i_node = i_cell_to_node[i_loc, i_cell]
            for i_k = 1:4
                __cse_25 = skx[i_k, i_cell]
                __cse_26 = sky[i_k, i_cell]
                __cse_27 = skz[i_k, i_cell]
                aeresk = aerex * __cse_25 + aerey * __cse_26 + aerez * __cse_27
                __cse_28 = dt ^ 2 / 3
                factor = __cse_28 * aeresk
                __cse_29 = beta[i_cell]
                auxres = resi + factor * __cse_29
                __cse_30 = gamma[i_cell]
                auxres2 = factor * __cse_30
                i_k_node = i_cell_to_node[i_k, i_cell]
                i_k_node = i_cell_to_node[i_k, i_cell]
                auxres2b = auxres2b + res2b[i_k_node]
                auxresb = auxresb + resb[i_k_node]
                factorb = factorb + __cse_30 * auxres2b
                gammab[i_cell] = gammab[i_cell] + factor * auxres2b
                auxres2b = 0.0
                resib = resib + auxresb
                factorb = factorb + __cse_29 * auxresb
                betab[i_cell] = betab[i_cell] + factor * auxresb
                auxresb = 0.0
                dtb = dtb + (2dt) * (0.3333333333333333 * (aeresk * factorb))
                aereskb = aereskb + __cse_28 * factorb
                factorb = 0.0
                aerexb = aerexb + __cse_25 * aereskb
                skxb[i_k, i_cell] = skxb[i_k, i_cell] + aerex * aereskb
                aereyb = aereyb + __cse_26 * aereskb
                skyb[i_k, i_cell] = skyb[i_k, i_cell] + aerey * aereskb
                aerezb = aerezb + __cse_27 * aereskb
                skzb[i_k, i_cell] = skzb[i_k, i_cell] + aerez * aereskb
                aereskb = 0.0
            end
            __cse_31 = 0.5 - gamma[i_cell]
            dtb = dtb + 0.25 * -((__cse_31 * vere) * resib)
            __cse_32 = -(dt / 4)
            gammab[i_cell] = gammab[i_cell] + -((__cse_32 * vere) * resib)
            vereb = vereb + (__cse_32 * __cse_31) * resib
            resib = 0.0
            cavgzb = cavgzb + re * aerezb
            reb = reb + cavgz * aerezb
            aerezb = 0.0
            cavgyb = cavgyb + re * aereyb
            reb = reb + cavgy * aereyb
            aereyb = 0.0
            cavgxb = cavgxb + re * aerexb
            reb = reb + cavgx * aerexb
            aerexb = 0.0
            __cse_33 = cell_vol[i_cell]
            vereb = vereb + (1.0 / __cse_33) * reb
            cell_volb[i_cell] = cell_volb[i_cell] + -(vere / __cse_33 ^ 2) * reb
            reb = 0.0
            __cse_34 = c[1, i_node]
            __cse_35 = skx[i_loc, i_cell]
            __cse_36 = c[2, i_node]
            __cse_37 = sky[i_loc, i_cell]
            __cse_38 = c[3, i_node]
            __cse_39 = skz[i_loc, i_cell]
            ub[i_node] = ub[i_node] + (-(1.0 / 3.0) * (__cse_34 * __cse_35 + __cse_36 * __cse_37 + __cse_38 * __cse_39)) * vereb
            __cse_40 = (-(1.0 / 3.0) * u[i_node]) * vereb
            cb[1, i_node] = cb[1, i_node] + __cse_35 * __cse_40
            skxb[i_loc, i_cell] = skxb[i_loc, i_cell] + __cse_34 * __cse_40
            cb[2, i_node] = cb[2, i_node] + __cse_37 * __cse_40
            skyb[i_loc, i_cell] = skyb[i_loc, i_cell] + __cse_36 * __cse_40
            cb[3, i_node] = cb[3, i_node] + __cse_39 * __cse_40
            skzb[i_loc, i_cell] = skzb[i_loc, i_cell] + __cse_38 * __cse_40
            vereb = 0.0
            factorb = 0.0
            aereskb = 0.0
        end
        for i_loc = 1:4
            i_node = i_cell_to_node[i_loc, i_cell]
            cavgx = cavgx + c[1, i_node] / 4
            cavgy = cavgy + c[2, i_node] / 4
            cavgz = cavgz + c[3, i_node] / 4
            i_node = i_cell_to_node[i_loc, i_cell]
            cb[3, i_node] = cb[3, i_node] + 0.25cavgzb
            cb[2, i_node] = cb[2, i_node] + 0.25cavgyb
            cb[1, i_node] = cb[1, i_node] + 0.25cavgxb
        end
        cavgzb = 0.0
        cavgyb = 0.0
        cavgxb = 0.0
        vereb = 0.0
        reb = 0.0
        factorb = 0.0
        aerezb = 0.0
        aereyb = 0.0
        aerexb = 0.0
        aereskb = 0.0
    end
    return dtb
end

function ttgc(u, uref, i_cell_to_node, cell_vol, node_vol, skx, sky, skz, i_ncell, i_nnode, c, dt, beta, gamma, i_njac, res, res2, up, mup, loss)
    for i_cell = 1:i_ncell
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
                res[i_k_node] = res[i_k_node] + auxres
                res2[i_k_node] = res2[i_k_node] + auxres2
            end
        end
    end
    for i_node = 1:i_nnode
        up[i_node] = res[i_node] / node_vol[i_node]
    end
    for i_ = 1:i_njac
        for i_node = 1:i_nnode
            mup[i_node] = 0.0
        end
        for i_cell = 1:i_ncell
            auxu = 0.0
            for i_loc = 1:4
                i_lnode = i_cell_to_node[i_loc, i_cell]
                auxu = auxu + up[i_lnode]
            end
            for i_loc = 1:4
                i_lnode = i_cell_to_node[i_loc, i_cell]
                mup[i_lnode] = mup[i_lnode] + ((auxu + up[i_lnode]) * cell_vol[i_cell]) / 20.0
            end
        end
        for i_node = 1:i_nnode
            up[i_node] = up[i_node] + (res[i_node] - mup[i_node]) / node_vol[i_node]
        end
    end
    for i_cell = 1:i_ncell
        for i_loc = 1:4
            i_node = i_cell_to_node[i_loc, i_cell]
            vere = -(1.0 / 3.0) * (u[i_node] + up[i_node]) * (c[1, i_node] * skx[i_loc, i_cell] + c[2, i_node] * sky[i_loc, i_cell])
            resi = -(dt / 4) * vere
            for i_k = 1:4
                i_k_node = i_cell_to_node[i_k, i_cell]
                res2[i_k_node] = res2[i_k_node] + resi
            end
        end
    end
    for i_node = 1:i_nnode
        up[i_node] = res2[i_node] / node_vol[i_node]
    end
    for i_ = 1:i_njac
        for i_node = 1:i_nnode
            mup[i_node] = 0.0
        end
        for i_cell = 1:i_ncell
            auxu = 0.0
            for i_loc = 1:4
                i_lnode = i_cell_to_node[i_loc, i_cell]
                auxu = auxu + up[i_lnode]
            end
            for i_loc = 1:4
                i_lnode = i_cell_to_node[i_loc, i_cell]
                mup[i_lnode] = mup[i_lnode] + ((auxu + up[i_lnode]) * cell_vol[i_cell]) / 20.0
            end
        end
        for i_node = 1:i_nnode
            up[i_node] = up[i_node] + (res2[i_node] - mup[i_node]) / node_vol[i_node]
        end
    end
    for i_node2 = 1:i_nnode
        loss[1] = loss[1] + ((u[i_node2] + up[i_node2]) - uref[i_node2]) ^ 2
    end
end
