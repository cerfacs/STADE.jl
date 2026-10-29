function initstacks_ttgc_b(i_ncell, i_njac, i_nnode, mup)
    __len_mup = length(mup)
    vere_stack = Vector{Float64}(undef, ((max(0, div(i_ncell - 1, 1) + 1) * max(0, div(4 - 1, 1) + 1) + max(0, div(i_ncell - 1, 1) + 1)) + max(0, div(i_ncell - 1, 1) + 1) * max(0, div(4 - 1, 1) + 1)) + max(0, div(i_ncell - 1, 1) + 1))
    up_stack = Vector{Float64}(undef, (max(0, div(i_njac - 1, 1) + 1) * max(0, div(i_nnode - 1, 1) + 1) + max(0, div(i_nnode - 1, 1) + 1)) + max(0, div(i_njac - 1, 1) + 1) * max(0, div(i_nnode - 1, 1) + 1))
    auxu_stack = Vector{Float64}(undef, max(0, div(i_njac - 1, 1) + 1) * max(0, div(i_ncell - 1, 1) + 1) + max(0, div(i_njac - 1, 1) + 1) * max(0, div(i_ncell - 1, 1) + 1))
    mup_snap_stack = Vector{Float64}(undef, (max(0, div(i_njac - 1, 1) + 1) + max(0, div(i_njac - 1, 1) + 1)) * __len_mup)
    return (vere_stack, up_stack, auxu_stack, mup_snap_stack, __len_mup)
end

function ttgc_b(u, ub, uref, urefb, i_cell_to_node, cell_vol, cell_volb, node_vol, node_volb, skx, skxb, sky, skyb, skz, skzb, i_ncell, i_nnode, c, cb, dt, dtb, beta, betab, gamma, gammab, i_njac, res, resb, res2, res2b, up, upb, mup, mupb, loss, lossb, vere_stack, up_stack, auxu_stack, mup_snap_stack, __len_mup)
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
        __idx_mup_snap_stack_0 = (i_ - 1) + 1
        __sbase_mup = (__idx_mup_snap_stack_0 - 1) * __len_mup
        for __sk_mup = 1:__len_mup
            mup_snap_stack[__sbase_mup + __sk_mup] = mup[__sk_mup]
        end
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
            __idx_auxu_stack_3 = ((i_cell - 1) + (i_ - 1) * (div(i_ncell - 1, 1) + 1)) + 1
            auxu_stack[__idx_auxu_stack_3] = auxu
        end
        for i_node = 1:i_nnode
            __idx_up_stack_0 = ((i_node - 1) + (i_ - 1) * (div(i_nnode - 1, 1) + 1)) + 1
            __cse_0 = up[i_node]
            up_stack[__idx_up_stack_0] = __cse_0
            up[i_node] = __cse_0 + (res[i_node] - mup[i_node]) / node_vol[i_node]
        end
    end
    for i_cell = 1:i_ncell
        vere = 0.0
        for i_loc = 1:4
            i_node = i_cell_to_node[i_loc, i_cell]
            __icse_1 = div(i_ncell - 1, 1) + 1
            __icse_2 = max(0, __icse_1)
            __idx_vere_stack_1 = (__icse_2 * max(0, div(4 - 1, 1) + 1) + __icse_2) + (((i_cell - 1) + (i_loc - 1) * __icse_1) + 1)
            vere_stack[__idx_vere_stack_1] = vere
            vere = -(1.0 / 3.0) * (u[i_node] + up[i_node]) * (c[1, i_node] * skx[i_loc, i_cell] + c[2, i_node] * sky[i_loc, i_cell])
            resi = -(dt / 4) * vere
            for i_k = 1:4
                i_k_node = i_cell_to_node[i_k, i_cell]
                res2[i_k_node] = res2[i_k_node] + resi
            end
        end
        __icse_3 = max(0, div(i_ncell - 1, 1) + 1)
        __icse_4 = __icse_3 * max(0, div(4 - 1, 1) + 1)
        __idx_vere_stack_2 = ((__icse_4 + __icse_3) + __icse_4) + ((i_cell - 1) + 1)
        vere_stack[__idx_vere_stack_2] = vere
    end
    for i_node = 1:i_nnode
        __idx_up_stack_0 = max(0, div(i_njac - 1, 1) + 1) * max(0, div(i_nnode - 1, 1) + 1) + ((i_node - 1) + 1)
        up_stack[__idx_up_stack_0] = up[i_node]
        up[i_node] = res2[i_node] / node_vol[i_node]
    end
    for i_ = 1:i_njac
        __idx_mup_snap_stack_0 = max(0, div(i_njac - 1, 1) + 1) + ((i_ - 1) + 1)
        __sbase_mup = (__idx_mup_snap_stack_0 - 1) * __len_mup
        for __sk_mup = 1:__len_mup
            mup_snap_stack[__sbase_mup + __sk_mup] = mup[__sk_mup]
        end
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
            __icse_5 = div(i_ncell - 1, 1) + 1
            __idx_auxu_stack_3 = max(0, div(i_njac - 1, 1) + 1) * max(0, __icse_5) + (((i_cell - 1) + (i_ - 1) * __icse_5) + 1)
            auxu_stack[__idx_auxu_stack_3] = auxu
        end
        for i_node = 1:i_nnode
            __icse_6 = div(i_nnode - 1, 1) + 1
            __icse_7 = max(0, __icse_6)
            __idx_up_stack_0 = (max(0, div(i_njac - 1, 1) + 1) * __icse_7 + __icse_7) + (((i_node - 1) + (i_ - 1) * __icse_6) + 1)
            __cse_8 = up[i_node]
            up_stack[__idx_up_stack_0] = __cse_8
            up[i_node] = __cse_8 + (res2[i_node] - mup[i_node]) / node_vol[i_node]
        end
    end
    for i_node2 = 1:i_nnode
        loss[1] = loss[1] + ((u[i_node2] + up[i_node2]) - uref[i_node2]) ^ 2
    end
    for i_node2 = i_nnode:-1:1
        __cse_9 = (2 * ((u[i_node2] + up[i_node2]) - uref[i_node2])) * lossb[1]
        ub[i_node2] = ub[i_node2] + __cse_9
        upb[i_node2] = upb[i_node2] + __cse_9
        urefb[i_node2] = urefb[i_node2] + -__cse_9
    end
    for i_ = i_njac:-1:1
        for i_node = i_nnode:-1:1
            __icse_10 = div(i_nnode - 1, 1) + 1
            __icse_11 = max(0, __icse_10)
            __idx_up_stack_0 = (max(0, div(i_njac - 1, 1) + 1) * __icse_11 + __icse_11) + (((i_node - 1) + (i_ - 1) * __icse_10) + 1)
            up[i_node] = up_stack[__idx_up_stack_0]
            __cse_12 = node_vol[i_node]
            __cse_13 = upb[i_node]
            __cse_14 = (1.0 / __cse_12) * __cse_13
            res2b[i_node] = res2b[i_node] + __cse_14
            mupb[i_node] = mupb[i_node] + -__cse_14
            node_volb[i_node] = node_volb[i_node] + -((res2[i_node] - mup[i_node]) / __cse_12 ^ 2) * __cse_13
        end
        for i_cell = i_ncell:-1:1
            __icse_15 = div(i_ncell - 1, 1) + 1
            __idx_auxu_stack_0 = max(0, div(i_njac - 1, 1) + 1) * max(0, __icse_15) + (((i_cell - 1) + (i_ - 1) * __icse_15) + 1)
            auxu = auxu_stack[__idx_auxu_stack_0]
            for i_loc = 4:-1:1
                i_lnode = i_cell_to_node[i_loc, i_cell]
                __cse_16 = 0.05 * mupb[i_lnode]
                __cse_17 = cell_vol[i_cell] * __cse_16
                auxub = auxub + __cse_17
                upb[i_lnode] = upb[i_lnode] + __cse_17
                cell_volb[i_cell] = cell_volb[i_cell] + (auxu + up[i_lnode]) * __cse_16
            end
            for i_loc = 1:4
                __icse_18 = i_cell_to_node[i_loc, i_cell]
                i_lnode = __icse_18
                auxu = auxu + up[i_lnode]
                i_lnode = __icse_18
                upb[i_lnode] = upb[i_lnode] + auxub
            end
            auxub = 0.0
        end
        for i_node = i_nnode:-1:1
            mupb[i_node] = 0.0
        end
        __idx_mup_snap_stack_0 = max(0, div(i_njac - 1, 1) + 1) + ((i_ - 1) + 1)
        __sbase_mup = (__idx_mup_snap_stack_0 - 1) * __len_mup
        for __sk_mup = 1:__len_mup
            mup[__sk_mup] = mup_snap_stack[__sbase_mup + __sk_mup]
        end
    end
    for i_node = i_nnode:-1:1
        __idx_up_stack_0 = max(0, div(i_njac - 1, 1) + 1) * max(0, div(i_nnode - 1, 1) + 1) + ((i_node - 1) + 1)
        up[i_node] = up_stack[__idx_up_stack_0]
        __oldb_2 = upb[i_node]
        upb[i_node] = 0.0
        __cse_19 = node_vol[i_node]
        res2b[i_node] = res2b[i_node] + (1.0 / __cse_19) * __oldb_2
        node_volb[i_node] = node_volb[i_node] + -(res2[i_node] / __cse_19 ^ 2) * __oldb_2
    end
    for i_cell = i_ncell:-1:1
        __icse_20 = max(0, div(i_ncell - 1, 1) + 1)
        __icse_21 = __icse_20 * max(0, div(4 - 1, 1) + 1)
        __idx_vere_stack_0 = ((__icse_21 + __icse_20) + __icse_21) + ((i_cell - 1) + 1)
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
            __icse_22 = div(i_ncell - 1, 1) + 1
            __icse_23 = max(0, __icse_22)
            __idx_vere_stack_0 = (__icse_23 * max(0, div(4 - 1, 1) + 1) + __icse_23) + (((i_cell - 1) + (i_loc - 1) * __icse_22) + 1)
            vere = vere_stack[__idx_vere_stack_0]
            __oldb_2 = vereb
            vereb = 0.0
            __icse_24 = -(1.0 / 3.0)
            __cse_25 = c[1, i_node]
            __cse_26 = skx[i_loc, i_cell]
            __cse_27 = c[2, i_node]
            __cse_28 = sky[i_loc, i_cell]
            __cse_29 = (__icse_24 * (__cse_25 * __cse_26 + __cse_27 * __cse_28)) * __oldb_2
            ub[i_node] = ub[i_node] + __cse_29
            upb[i_node] = upb[i_node] + __cse_29
            __cse_30 = (__icse_24 * (u[i_node] + up[i_node])) * __oldb_2
            cb[1, i_node] = cb[1, i_node] + __cse_26 * __cse_30
            skxb[i_loc, i_cell] = skxb[i_loc, i_cell] + __cse_25 * __cse_30
            cb[2, i_node] = cb[2, i_node] + __cse_28 * __cse_30
            skyb[i_loc, i_cell] = skyb[i_loc, i_cell] + __cse_27 * __cse_30
        end
        vereb = 0.0
    end
    for i_ = i_njac:-1:1
        for i_node = i_nnode:-1:1
            __idx_up_stack_0 = ((i_node - 1) + (i_ - 1) * (div(i_nnode - 1, 1) + 1)) + 1
            up[i_node] = up_stack[__idx_up_stack_0]
            __cse_31 = node_vol[i_node]
            __cse_32 = upb[i_node]
            __cse_33 = (1.0 / __cse_31) * __cse_32
            resb[i_node] = resb[i_node] + __cse_33
            mupb[i_node] = mupb[i_node] + -__cse_33
            node_volb[i_node] = node_volb[i_node] + -((res[i_node] - mup[i_node]) / __cse_31 ^ 2) * __cse_32
        end
        for i_cell = i_ncell:-1:1
            __idx_auxu_stack_0 = ((i_cell - 1) + (i_ - 1) * (div(i_ncell - 1, 1) + 1)) + 1
            auxu = auxu_stack[__idx_auxu_stack_0]
            for i_loc = 4:-1:1
                i_lnode = i_cell_to_node[i_loc, i_cell]
                __cse_34 = 0.05 * mupb[i_lnode]
                __cse_35 = cell_vol[i_cell] * __cse_34
                auxub = auxub + __cse_35
                upb[i_lnode] = upb[i_lnode] + __cse_35
                cell_volb[i_cell] = cell_volb[i_cell] + (auxu + up[i_lnode]) * __cse_34
            end
            for i_loc = 1:4
                __icse_36 = i_cell_to_node[i_loc, i_cell]
                i_lnode = __icse_36
                auxu = auxu + up[i_lnode]
                i_lnode = __icse_36
                upb[i_lnode] = upb[i_lnode] + auxub
            end
            auxub = 0.0
        end
        for i_node = i_nnode:-1:1
            mupb[i_node] = 0.0
        end
        __idx_mup_snap_stack_0 = (i_ - 1) + 1
        __sbase_mup = (__idx_mup_snap_stack_0 - 1) * __len_mup
        for __sk_mup = 1:__len_mup
            mup[__sk_mup] = mup_snap_stack[__sbase_mup + __sk_mup]
        end
    end
    for i_node = i_nnode:-1:1
        __oldb_0 = upb[i_node]
        upb[i_node] = 0.0
        __cse_37 = node_vol[i_node]
        resb[i_node] = resb[i_node] + (1.0 / __cse_37) * __oldb_0
        node_volb[i_node] = node_volb[i_node] + -(res[i_node] / __cse_37 ^ 2) * __oldb_0
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
                __cse_38 = skx[i_k, i_cell]
                __cse_39 = sky[i_k, i_cell]
                __cse_40 = skz[i_k, i_cell]
                aeresk = aerex * __cse_38 + aerey * __cse_39 + aerez * __cse_40
                __cse_41 = dt ^ 2 / 3
                factor = __cse_41 * aeresk
                __cse_42 = beta[i_cell]
                auxres = resi + factor * __cse_42
                __cse_43 = gamma[i_cell]
                auxres2 = factor * __cse_43
                __icse_44 = i_cell_to_node[i_k, i_cell]
                i_k_node = __icse_44
                i_k_node = __icse_44
                auxres2b = auxres2b + res2b[i_k_node]
                auxresb = auxresb + resb[i_k_node]
                __oldb_0 = auxres2b
                auxres2b = 0.0
                factorb = factorb + __cse_43 * __oldb_0
                gammab[i_cell] = gammab[i_cell] + factor * __oldb_0
                __oldb_0 = auxresb
                auxresb = 0.0
                resib = resib + __oldb_0
                factorb = factorb + __cse_42 * __oldb_0
                betab[i_cell] = betab[i_cell] + factor * __oldb_0
                __oldb_0 = factorb
                factorb = 0.0
                dtb = dtb + (2dt) * (0.3333333333333333 * (aeresk * __oldb_0))
                aereskb = aereskb + __cse_41 * __oldb_0
                __oldb_0 = aereskb
                aereskb = 0.0
                aerexb = aerexb + __cse_38 * __oldb_0
                skxb[i_k, i_cell] = skxb[i_k, i_cell] + aerex * __oldb_0
                aereyb = aereyb + __cse_39 * __oldb_0
                skyb[i_k, i_cell] = skyb[i_k, i_cell] + aerey * __oldb_0
                aerezb = aerezb + __cse_40 * __oldb_0
                skzb[i_k, i_cell] = skzb[i_k, i_cell] + aerez * __oldb_0
            end
            __oldb_0 = resib
            resib = 0.0
            __cse_45 = 0.5 - gamma[i_cell]
            dtb = dtb + 0.25 * -((__cse_45 * vere) * __oldb_0)
            __cse_46 = -(dt / 4)
            gammab[i_cell] = gammab[i_cell] + -((__cse_46 * vere) * __oldb_0)
            vereb = vereb + (__cse_46 * __cse_45) * __oldb_0
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
            __cse_47 = cell_vol[i_cell]
            vereb = vereb + (1.0 / __cse_47) * __oldb_0
            cell_volb[i_cell] = cell_volb[i_cell] + -(vere / __cse_47 ^ 2) * __oldb_0
            __oldb_0 = vereb
            vereb = 0.0
            __icse_48 = -(1.0 / 3.0)
            __cse_49 = c[1, i_node]
            __cse_50 = skx[i_loc, i_cell]
            __cse_51 = c[2, i_node]
            __cse_52 = sky[i_loc, i_cell]
            __cse_53 = c[3, i_node]
            __cse_54 = skz[i_loc, i_cell]
            ub[i_node] = ub[i_node] + (__icse_48 * (__cse_49 * __cse_50 + __cse_51 * __cse_52 + __cse_53 * __cse_54)) * __oldb_0
            __cse_55 = (__icse_48 * u[i_node]) * __oldb_0
            cb[1, i_node] = cb[1, i_node] + __cse_50 * __cse_55
            skxb[i_loc, i_cell] = skxb[i_loc, i_cell] + __cse_49 * __cse_55
            cb[2, i_node] = cb[2, i_node] + __cse_52 * __cse_55
            skyb[i_loc, i_cell] = skyb[i_loc, i_cell] + __cse_51 * __cse_55
            cb[3, i_node] = cb[3, i_node] + __cse_54 * __cse_55
            skzb[i_loc, i_cell] = skzb[i_loc, i_cell] + __cse_53 * __cse_55
            factorb = 0.0
            aereskb = 0.0
        end
        for i_loc = 1:4
            __icse_56 = i_cell_to_node[i_loc, i_cell]
            i_node = __icse_56
            cavgx = cavgx + c[1, i_node] / 4
            cavgy = cavgy + c[2, i_node] / 4
            cavgz = cavgz + c[3, i_node] / 4
            i_node = __icse_56
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
