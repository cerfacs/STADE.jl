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
                __icse_0 = div(4 - 1, 1) + 1
                __idx_mup_stack_1 = max(0, div(i_njac - 1, 1) + 1) * max(0, div(i_nnode - 1, 1) + 1) + (((i_ - 1) * ((div(i_ncell - 1, 1) + 1) * __icse_0) + (i_cell - 1) * __icse_0 + (i_loc - 1)) + 1)
                __cse_1 = mup[i_lnode]
                mup_stack[__idx_mup_stack_1] = __cse_1
                mup[i_lnode] = __cse_1 + ((auxu + up[i_lnode]) * cell_vol[i_cell]) / 20.0
            end
            __idx_auxu_stack_3 = ((i_ - 1) * (div(i_ncell - 1, 1) + 1) + (i_cell - 1)) + 1
            auxu_stack[__idx_auxu_stack_3] = auxu
        end
        for i_node = 1:i_nnode
            __idx_up_stack_0 = ((i_ - 1) * (div(i_nnode - 1, 1) + 1) + (i_node - 1)) + 1
            __cse_2 = up[i_node]
            up_stack[__idx_up_stack_0] = __cse_2
            up[i_node] = __cse_2 + (res[i_node] - mup[i_node]) / node_vol[i_node]
        end
    end
    for i_cell = 1:i_ncell
        vere = 0.0
        for i_loc = 1:4
            i_node = i_cell_to_node[i_loc, i_cell]
            __icse_3 = max(0, div(i_ncell - 1, 1) + 1)
            __icse_4 = div(4 - 1, 1) + 1
            __idx_vere_stack_1 = (__icse_3 * max(0, __icse_4) + __icse_3) + (((i_cell - 1) * __icse_4 + (i_loc - 1)) + 1)
            vere_stack[__idx_vere_stack_1] = vere
            vere = -(1.0 / 3.0) * (u[i_node] + up[i_node]) * (c[1, i_node] * skx[i_loc, i_cell] + c[2, i_node] * sky[i_loc, i_cell])
            resi = -(dt / 4) * vere
            for i_k = 1:4
                i_k_node = i_cell_to_node[i_k, i_cell]
                res2[i_k_node] = res2[i_k_node] + resi
            end
        end
        __icse_5 = max(0, div(i_ncell - 1, 1) + 1)
        __icse_6 = __icse_5 * max(0, div(4 - 1, 1) + 1)
        __idx_vere_stack_2 = ((__icse_6 + __icse_5) + __icse_6) + ((i_cell - 1) + 1)
        vere_stack[__idx_vere_stack_2] = vere
    end
    for i_node = 1:i_nnode
        __idx_up_stack_0 = max(0, div(i_njac - 1, 1) + 1) * max(0, div(i_nnode - 1, 1) + 1) + ((i_node - 1) + 1)
        up_stack[__idx_up_stack_0] = up[i_node]
        up[i_node] = res2[i_node] / node_vol[i_node]
    end
    for i_ = 1:i_njac
        for i_node = 1:i_nnode
            __icse_7 = max(0, div(i_njac - 1, 1) + 1)
            __icse_8 = div(i_nnode - 1, 1) + 1
            __idx_mup_stack_0 = (__icse_7 * max(0, __icse_8) + __icse_7 * max(0, div(i_ncell - 1, 1) + 1) * max(0, div(4 - 1, 1) + 1)) + (((i_ - 1) * __icse_8 + (i_node - 1)) + 1)
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
                __icse_9 = max(0, div(i_njac - 1, 1) + 1)
                __icse_10 = __icse_9 * max(0, div(i_nnode - 1, 1) + 1)
                __icse_11 = div(i_ncell - 1, 1) + 1
                __icse_12 = div(4 - 1, 1) + 1
                __idx_mup_stack_1 = ((__icse_10 + __icse_9 * max(0, __icse_11) * max(0, __icse_12)) + __icse_10) + (((i_ - 1) * (__icse_11 * __icse_12) + (i_cell - 1) * __icse_12 + (i_loc - 1)) + 1)
                __cse_13 = mup[i_lnode]
                mup_stack[__idx_mup_stack_1] = __cse_13
                mup[i_lnode] = __cse_13 + ((auxu + up[i_lnode]) * cell_vol[i_cell]) / 20.0
            end
            __icse_14 = div(i_ncell - 1, 1) + 1
            __idx_auxu_stack_3 = max(0, div(i_njac - 1, 1) + 1) * max(0, __icse_14) + (((i_ - 1) * __icse_14 + (i_cell - 1)) + 1)
            auxu_stack[__idx_auxu_stack_3] = auxu
        end
        for i_node = 1:i_nnode
            __icse_15 = div(i_nnode - 1, 1) + 1
            __icse_16 = max(0, __icse_15)
            __idx_up_stack_0 = (max(0, div(i_njac - 1, 1) + 1) * __icse_16 + __icse_16) + (((i_ - 1) * __icse_15 + (i_node - 1)) + 1)
            __cse_17 = up[i_node]
            up_stack[__idx_up_stack_0] = __cse_17
            up[i_node] = __cse_17 + (res2[i_node] - mup[i_node]) / node_vol[i_node]
        end
    end
    for i_node2 = 1:i_nnode
        loss[1] = loss[1] + ((u[i_node2] + up[i_node2]) - uref[i_node2]) ^ 2
    end
    for i_node2 = i_nnode:-1:1
        __cse_18 = (2 * ((u[i_node2] + up[i_node2]) - uref[i_node2])) * lossb[1]
        ub[i_node2] = ub[i_node2] + __cse_18
        upb[i_node2] = upb[i_node2] + __cse_18
        urefb[i_node2] = urefb[i_node2] + -__cse_18
    end
    for i_ = i_njac:-1:1
        for i_node = i_nnode:-1:1
            __icse_19 = div(i_nnode - 1, 1) + 1
            __icse_20 = max(0, __icse_19)
            __idx_up_stack_0 = (max(0, div(i_njac - 1, 1) + 1) * __icse_20 + __icse_20) + (((i_ - 1) * __icse_19 + (i_node - 1)) + 1)
            up[i_node] = up_stack[__idx_up_stack_0]
            __cse_21 = node_vol[i_node]
            __cse_22 = upb[i_node]
            __cse_23 = (1.0 / __cse_21) * __cse_22
            res2b[i_node] = res2b[i_node] + __cse_23
            mupb[i_node] = mupb[i_node] + -__cse_23
            node_volb[i_node] = node_volb[i_node] + -((res2[i_node] - mup[i_node]) / __cse_21 ^ 2) * __cse_22
        end
        for i_cell = i_ncell:-1:1
            __icse_24 = div(i_ncell - 1, 1) + 1
            __idx_auxu_stack_0 = max(0, div(i_njac - 1, 1) + 1) * max(0, __icse_24) + (((i_ - 1) * __icse_24 + (i_cell - 1)) + 1)
            auxu = auxu_stack[__idx_auxu_stack_0]
            for i_loc = 4:-1:1
                i_lnode = i_cell_to_node[i_loc, i_cell]
                __icse_25 = max(0, div(i_njac - 1, 1) + 1)
                __icse_26 = __icse_25 * max(0, div(i_nnode - 1, 1) + 1)
                __icse_27 = div(i_ncell - 1, 1) + 1
                __icse_28 = div(4 - 1, 1) + 1
                __idx_mup_stack_0 = ((__icse_26 + __icse_25 * max(0, __icse_27) * max(0, __icse_28)) + __icse_26) + (((i_ - 1) * (__icse_27 * __icse_28) + (i_cell - 1) * __icse_28 + (i_loc - 1)) + 1)
                mup[i_lnode] = mup_stack[__idx_mup_stack_0]
                __cse_29 = 0.05 * mupb[i_lnode]
                __cse_30 = cell_vol[i_cell] * __cse_29
                auxub = auxub + __cse_30
                upb[i_lnode] = upb[i_lnode] + __cse_30
                cell_volb[i_cell] = cell_volb[i_cell] + (auxu + up[i_lnode]) * __cse_29
            end
            for i_loc = 1:4
                __icse_31 = i_cell_to_node[i_loc, i_cell]
                i_lnode = __icse_31
                auxu = auxu + up[i_lnode]
                i_lnode = __icse_31
                upb[i_lnode] = upb[i_lnode] + auxub
            end
            auxub = 0.0
        end
        for i_node = i_nnode:-1:1
            __icse_32 = max(0, div(i_njac - 1, 1) + 1)
            __icse_33 = div(i_nnode - 1, 1) + 1
            __idx_mup_stack_0 = (__icse_32 * max(0, __icse_33) + __icse_32 * max(0, div(i_ncell - 1, 1) + 1) * max(0, div(4 - 1, 1) + 1)) + (((i_ - 1) * __icse_33 + (i_node - 1)) + 1)
            mup[i_node] = mup_stack[__idx_mup_stack_0]
            mupb[i_node] = 0.0
        end
    end
    for i_node = i_nnode:-1:1
        __idx_up_stack_0 = max(0, div(i_njac - 1, 1) + 1) * max(0, div(i_nnode - 1, 1) + 1) + ((i_node - 1) + 1)
        up[i_node] = up_stack[__idx_up_stack_0]
        __oldb_2 = upb[i_node]
        upb[i_node] = 0.0
        __cse_34 = node_vol[i_node]
        res2b[i_node] = res2b[i_node] + (1.0 / __cse_34) * __oldb_2
        node_volb[i_node] = node_volb[i_node] + -(res2[i_node] / __cse_34 ^ 2) * __oldb_2
    end
    for i_cell = i_ncell:-1:1
        __icse_35 = max(0, div(i_ncell - 1, 1) + 1)
        __icse_36 = __icse_35 * max(0, div(4 - 1, 1) + 1)
        __idx_vere_stack_0 = ((__icse_36 + __icse_35) + __icse_36) + ((i_cell - 1) + 1)
        vere = vere_stack[__idx_vere_stack_0]
        for i_loc = 4:-1:1
            i_node = i_cell_to_node[i_loc, i_cell]
            for i_k = 4:-1:1
                i_k_node = i_cell_to_node[i_k, i_cell]
                resib = resib + res2b[i_k_node]
            end
            __oldb_0 = resib
            resib = 0.0
            dtb = dtb + 0.25 * -(vere * __oldb_0)
            vereb = vereb + -(dt / 4) * __oldb_0
            __icse_37 = max(0, div(i_ncell - 1, 1) + 1)
            __icse_38 = div(4 - 1, 1) + 1
            __idx_vere_stack_0 = (__icse_37 * max(0, __icse_38) + __icse_37) + (((i_cell - 1) * __icse_38 + (i_loc - 1)) + 1)
            vere = vere_stack[__idx_vere_stack_0]
            __oldb_2 = vereb
            vereb = 0.0
            __icse_39 = -(1.0 / 3.0)
            __cse_40 = c[1, i_node]
            __cse_41 = skx[i_loc, i_cell]
            __cse_42 = c[2, i_node]
            __cse_43 = sky[i_loc, i_cell]
            __cse_44 = (__icse_39 * (__cse_40 * __cse_41 + __cse_42 * __cse_43)) * __oldb_2
            ub[i_node] = ub[i_node] + __cse_44
            upb[i_node] = upb[i_node] + __cse_44
            __cse_45 = (__icse_39 * (u[i_node] + up[i_node])) * __oldb_2
            cb[1, i_node] = cb[1, i_node] + __cse_41 * __cse_45
            skxb[i_loc, i_cell] = skxb[i_loc, i_cell] + __cse_40 * __cse_45
            cb[2, i_node] = cb[2, i_node] + __cse_43 * __cse_45
            skyb[i_loc, i_cell] = skyb[i_loc, i_cell] + __cse_42 * __cse_45
        end
        vereb = 0.0
    end
    for i_ = i_njac:-1:1
        for i_node = i_nnode:-1:1
            __idx_up_stack_0 = ((i_ - 1) * (div(i_nnode - 1, 1) + 1) + (i_node - 1)) + 1
            up[i_node] = up_stack[__idx_up_stack_0]
            __cse_46 = node_vol[i_node]
            __cse_47 = upb[i_node]
            __cse_48 = (1.0 / __cse_46) * __cse_47
            resb[i_node] = resb[i_node] + __cse_48
            mupb[i_node] = mupb[i_node] + -__cse_48
            node_volb[i_node] = node_volb[i_node] + -((res[i_node] - mup[i_node]) / __cse_46 ^ 2) * __cse_47
        end
        for i_cell = i_ncell:-1:1
            __idx_auxu_stack_0 = ((i_ - 1) * (div(i_ncell - 1, 1) + 1) + (i_cell - 1)) + 1
            auxu = auxu_stack[__idx_auxu_stack_0]
            for i_loc = 4:-1:1
                i_lnode = i_cell_to_node[i_loc, i_cell]
                __icse_49 = div(4 - 1, 1) + 1
                __idx_mup_stack_0 = max(0, div(i_njac - 1, 1) + 1) * max(0, div(i_nnode - 1, 1) + 1) + (((i_ - 1) * ((div(i_ncell - 1, 1) + 1) * __icse_49) + (i_cell - 1) * __icse_49 + (i_loc - 1)) + 1)
                mup[i_lnode] = mup_stack[__idx_mup_stack_0]
                __cse_50 = 0.05 * mupb[i_lnode]
                __cse_51 = cell_vol[i_cell] * __cse_50
                auxub = auxub + __cse_51
                upb[i_lnode] = upb[i_lnode] + __cse_51
                cell_volb[i_cell] = cell_volb[i_cell] + (auxu + up[i_lnode]) * __cse_50
            end
            for i_loc = 1:4
                __icse_52 = i_cell_to_node[i_loc, i_cell]
                i_lnode = __icse_52
                auxu = auxu + up[i_lnode]
                i_lnode = __icse_52
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
        __oldb_0 = upb[i_node]
        upb[i_node] = 0.0
        __cse_53 = node_vol[i_node]
        resb[i_node] = resb[i_node] + (1.0 / __cse_53) * __oldb_0
        node_volb[i_node] = node_volb[i_node] + -(res[i_node] / __cse_53 ^ 2) * __oldb_0
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
                __cse_54 = skx[i_k, i_cell]
                __cse_55 = sky[i_k, i_cell]
                __cse_56 = skz[i_k, i_cell]
                aeresk = aerex * __cse_54 + aerey * __cse_55 + aerez * __cse_56
                __cse_57 = dt ^ 2 / 3
                factor = __cse_57 * aeresk
                __cse_58 = beta[i_cell]
                auxres = resi + factor * __cse_58
                __cse_59 = gamma[i_cell]
                auxres2 = factor * __cse_59
                __icse_60 = i_cell_to_node[i_k, i_cell]
                i_k_node = __icse_60
                i_k_node = __icse_60
                auxres2b = auxres2b + res2b[i_k_node]
                auxresb = auxresb + resb[i_k_node]
                __oldb_0 = auxres2b
                auxres2b = 0.0
                factorb = factorb + __cse_59 * __oldb_0
                gammab[i_cell] = gammab[i_cell] + factor * __oldb_0
                __oldb_0 = auxresb
                auxresb = 0.0
                resib = resib + __oldb_0
                factorb = factorb + __cse_58 * __oldb_0
                betab[i_cell] = betab[i_cell] + factor * __oldb_0
                __oldb_0 = factorb
                factorb = 0.0
                dtb = dtb + (2dt) * (0.3333333333333333 * (aeresk * __oldb_0))
                aereskb = aereskb + __cse_57 * __oldb_0
                __oldb_0 = aereskb
                aereskb = 0.0
                aerexb = aerexb + __cse_54 * __oldb_0
                skxb[i_k, i_cell] = skxb[i_k, i_cell] + aerex * __oldb_0
                aereyb = aereyb + __cse_55 * __oldb_0
                skyb[i_k, i_cell] = skyb[i_k, i_cell] + aerey * __oldb_0
                aerezb = aerezb + __cse_56 * __oldb_0
                skzb[i_k, i_cell] = skzb[i_k, i_cell] + aerez * __oldb_0
            end
            __oldb_0 = resib
            resib = 0.0
            __cse_61 = 0.5 - gamma[i_cell]
            dtb = dtb + 0.25 * -((__cse_61 * vere) * __oldb_0)
            __cse_62 = -(dt / 4)
            gammab[i_cell] = gammab[i_cell] + -((__cse_62 * vere) * __oldb_0)
            vereb = vereb + (__cse_62 * __cse_61) * __oldb_0
            __oldb_0 = aerezb
            aerezb = 0.0
            cavgzb = cavgzb + re * __oldb_0
            reb = reb + cavgz * __oldb_0
            __oldb_0 = aereyb
            aereyb = 0.0
            cavgyb = cavgyb + re * __oldb_0
            reb = reb + cavgy * __oldb_0
            __oldb_0 = aerexb
            aerexb = 0.0
            cavgxb = cavgxb + re * __oldb_0
            reb = reb + cavgx * __oldb_0
            __oldb_0 = reb
            reb = 0.0
            __cse_63 = cell_vol[i_cell]
            vereb = vereb + (1.0 / __cse_63) * __oldb_0
            cell_volb[i_cell] = cell_volb[i_cell] + -(vere / __cse_63 ^ 2) * __oldb_0
            __oldb_0 = vereb
            vereb = 0.0
            __icse_64 = -(1.0 / 3.0)
            __cse_65 = c[1, i_node]
            __cse_66 = skx[i_loc, i_cell]
            __cse_67 = c[2, i_node]
            __cse_68 = sky[i_loc, i_cell]
            __cse_69 = c[3, i_node]
            __cse_70 = skz[i_loc, i_cell]
            ub[i_node] = ub[i_node] + (__icse_64 * (__cse_65 * __cse_66 + __cse_67 * __cse_68 + __cse_69 * __cse_70)) * __oldb_0
            __cse_71 = (__icse_64 * u[i_node]) * __oldb_0
            cb[1, i_node] = cb[1, i_node] + __cse_66 * __cse_71
            skxb[i_loc, i_cell] = skxb[i_loc, i_cell] + __cse_65 * __cse_71
            cb[2, i_node] = cb[2, i_node] + __cse_68 * __cse_71
            skyb[i_loc, i_cell] = skyb[i_loc, i_cell] + __cse_67 * __cse_71
            cb[3, i_node] = cb[3, i_node] + __cse_70 * __cse_71
            skzb[i_loc, i_cell] = skzb[i_loc, i_cell] + __cse_69 * __cse_71
            factorb = 0.0
            aereskb = 0.0
        end
        for i_loc = 1:4
            __icse_72 = i_cell_to_node[i_loc, i_cell]
            i_node = __icse_72
            cavgx = cavgx + c[1, i_node] / 4
            cavgy = cavgy + c[2, i_node] / 4
            cavgz = cavgz + c[3, i_node] / 4
            i_node = __icse_72
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
