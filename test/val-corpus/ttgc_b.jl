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
                mup_stack[__idx_mup_stack_1] = mup[i_lnode]
                mup[i_lnode] = mup[i_lnode] + ((auxu + up[i_lnode]) * cell_vol[i_cell]) / 20.0
            end
            __idx_auxu_stack_3 = ((i_ - 1) * (div(i_ncell - 1, 1) + 1) + (i_cell - 1)) + 1
            auxu_stack[__idx_auxu_stack_3] = auxu
        end
        for i_node = 1:i_nnode
            __idx_up_stack_0 = ((i_ - 1) * (div(i_nnode - 1, 1) + 1) + (i_node - 1)) + 1
            up_stack[__idx_up_stack_0] = up[i_node]
            up[i_node] = up[i_node] + (res[i_node] - mup[i_node]) / node_vol[i_node]
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
                mup_stack[__idx_mup_stack_1] = mup[i_lnode]
                mup[i_lnode] = mup[i_lnode] + ((auxu + up[i_lnode]) * cell_vol[i_cell]) / 20.0
            end
            __idx_auxu_stack_3 = max(0, div(i_njac - 1, 1) + 1) * max(0, div(i_ncell - 1, 1) + 1) + (((i_ - 1) * (div(i_ncell - 1, 1) + 1) + (i_cell - 1)) + 1)
            auxu_stack[__idx_auxu_stack_3] = auxu
        end
        for i_node = 1:i_nnode
            __idx_up_stack_0 = (max(0, div(i_njac - 1, 1) + 1) * max(0, div(i_nnode - 1, 1) + 1) + max(0, div(i_nnode - 1, 1) + 1)) + (((i_ - 1) * (div(i_nnode - 1, 1) + 1) + (i_node - 1)) + 1)
            up_stack[__idx_up_stack_0] = up[i_node]
            up[i_node] = up[i_node] + (res2[i_node] - mup[i_node]) / node_vol[i_node]
        end
    end
    for i_node2 = 1:i_nnode
        loss[1] = loss[1] + ((u[i_node2] + up[i_node2]) - uref[i_node2]) ^ 2
    end
    for i_node2 = i_nnode:-1:1
        ub[i_node2] = ub[i_node2] + (2 * ((u[i_node2] + up[i_node2]) - uref[i_node2])) * lossb[1]
        upb[i_node2] = upb[i_node2] + (2 * ((u[i_node2] + up[i_node2]) - uref[i_node2])) * lossb[1]
        urefb[i_node2] = urefb[i_node2] + -((2 * ((u[i_node2] + up[i_node2]) - uref[i_node2])) * lossb[1])
    end
    for i_ = i_njac:-1:1
        for i_node = i_nnode:-1:1
            __idx_up_stack_0 = (max(0, div(i_njac - 1, 1) + 1) * max(0, div(i_nnode - 1, 1) + 1) + max(0, div(i_nnode - 1, 1) + 1)) + (((i_ - 1) * (div(i_nnode - 1, 1) + 1) + (i_node - 1)) + 1)
            up[i_node] = up_stack[__idx_up_stack_0]
            res2b[i_node] = res2b[i_node] + (1.0 / node_vol[i_node]) * upb[i_node]
            mupb[i_node] = mupb[i_node] + -((1.0 / node_vol[i_node]) * upb[i_node])
            node_volb[i_node] = node_volb[i_node] + -((res2[i_node] - mup[i_node]) / node_vol[i_node] ^ 2) * upb[i_node]
        end
        for i_cell = i_ncell:-1:1
            __idx_auxu_stack_0 = max(0, div(i_njac - 1, 1) + 1) * max(0, div(i_ncell - 1, 1) + 1) + (((i_ - 1) * (div(i_ncell - 1, 1) + 1) + (i_cell - 1)) + 1)
            auxu = auxu_stack[__idx_auxu_stack_0]
            for i_loc = 4:-1:1
                i_lnode = i_cell_to_node[i_loc, i_cell]
                __idx_mup_stack_0 = ((max(0, div(i_njac - 1, 1) + 1) * max(0, div(i_nnode - 1, 1) + 1) + max(0, div(i_njac - 1, 1) + 1) * max(0, div(i_ncell - 1, 1) + 1) * max(0, div(4 - 1, 1) + 1)) + max(0, div(i_njac - 1, 1) + 1) * max(0, div(i_nnode - 1, 1) + 1)) + (((i_ - 1) * ((div(i_ncell - 1, 1) + 1) * (div(4 - 1, 1) + 1)) + (i_cell - 1) * (div(4 - 1, 1) + 1) + (i_loc - 1)) + 1)
                mup[i_lnode] = mup_stack[__idx_mup_stack_0]
                auxub = auxub + cell_vol[i_cell] * (0.05 * mupb[i_lnode])
                upb[i_lnode] = upb[i_lnode] + cell_vol[i_cell] * (0.05 * mupb[i_lnode])
                cell_volb[i_cell] = cell_volb[i_cell] + (auxu + up[i_lnode]) * (0.05 * mupb[i_lnode])
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
        res2b[i_node] = res2b[i_node] + (1.0 / node_vol[i_node]) * upb[i_node]
        node_volb[i_node] = node_volb[i_node] + -(res2[i_node] / node_vol[i_node] ^ 2) * upb[i_node]
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
            ub[i_node] = ub[i_node] + (-(1.0 / 3.0) * (c[1, i_node] * skx[i_loc, i_cell] + c[2, i_node] * sky[i_loc, i_cell])) * vereb
            upb[i_node] = upb[i_node] + (-(1.0 / 3.0) * (c[1, i_node] * skx[i_loc, i_cell] + c[2, i_node] * sky[i_loc, i_cell])) * vereb
            cb[1, i_node] = cb[1, i_node] + skx[i_loc, i_cell] * ((-(1.0 / 3.0) * (u[i_node] + up[i_node])) * vereb)
            skxb[i_loc, i_cell] = skxb[i_loc, i_cell] + c[1, i_node] * ((-(1.0 / 3.0) * (u[i_node] + up[i_node])) * vereb)
            cb[2, i_node] = cb[2, i_node] + sky[i_loc, i_cell] * ((-(1.0 / 3.0) * (u[i_node] + up[i_node])) * vereb)
            skyb[i_loc, i_cell] = skyb[i_loc, i_cell] + c[2, i_node] * ((-(1.0 / 3.0) * (u[i_node] + up[i_node])) * vereb)
            vereb = 0.0
        end
        vereb = 0.0
    end
    for i_ = i_njac:-1:1
        for i_node = i_nnode:-1:1
            __idx_up_stack_0 = ((i_ - 1) * (div(i_nnode - 1, 1) + 1) + (i_node - 1)) + 1
            up[i_node] = up_stack[__idx_up_stack_0]
            resb[i_node] = resb[i_node] + (1.0 / node_vol[i_node]) * upb[i_node]
            mupb[i_node] = mupb[i_node] + -((1.0 / node_vol[i_node]) * upb[i_node])
            node_volb[i_node] = node_volb[i_node] + -((res[i_node] - mup[i_node]) / node_vol[i_node] ^ 2) * upb[i_node]
        end
        for i_cell = i_ncell:-1:1
            __idx_auxu_stack_0 = ((i_ - 1) * (div(i_ncell - 1, 1) + 1) + (i_cell - 1)) + 1
            auxu = auxu_stack[__idx_auxu_stack_0]
            for i_loc = 4:-1:1
                i_lnode = i_cell_to_node[i_loc, i_cell]
                __idx_mup_stack_0 = max(0, div(i_njac - 1, 1) + 1) * max(0, div(i_nnode - 1, 1) + 1) + (((i_ - 1) * ((div(i_ncell - 1, 1) + 1) * (div(4 - 1, 1) + 1)) + (i_cell - 1) * (div(4 - 1, 1) + 1) + (i_loc - 1)) + 1)
                mup[i_lnode] = mup_stack[__idx_mup_stack_0]
                auxub = auxub + cell_vol[i_cell] * (0.05 * mupb[i_lnode])
                upb[i_lnode] = upb[i_lnode] + cell_vol[i_cell] * (0.05 * mupb[i_lnode])
                cell_volb[i_cell] = cell_volb[i_cell] + (auxu + up[i_lnode]) * (0.05 * mupb[i_lnode])
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
        resb[i_node] = resb[i_node] + (1.0 / node_vol[i_node]) * upb[i_node]
        node_volb[i_node] = node_volb[i_node] + -(res[i_node] / node_vol[i_node] ^ 2) * upb[i_node]
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
                aeresk = aerex * skx[i_k, i_cell] + aerey * sky[i_k, i_cell] + aerez * skz[i_k, i_cell]
                factor = (dt ^ 2 / 3) * aeresk
                auxres = resi + factor * beta[i_cell]
                auxres2 = factor * gamma[i_cell]
                i_k_node = i_cell_to_node[i_k, i_cell]
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
                dtb = dtb + (2dt) * (0.3333333333333333 * (aeresk * factorb))
                aereskb = aereskb + (dt ^ 2 / 3) * factorb
                factorb = 0.0
                aerexb = aerexb + skx[i_k, i_cell] * aereskb
                skxb[i_k, i_cell] = skxb[i_k, i_cell] + aerex * aereskb
                aereyb = aereyb + sky[i_k, i_cell] * aereskb
                skyb[i_k, i_cell] = skyb[i_k, i_cell] + aerey * aereskb
                aerezb = aerezb + skz[i_k, i_cell] * aereskb
                skzb[i_k, i_cell] = skzb[i_k, i_cell] + aerez * aereskb
                aereskb = 0.0
            end
            dtb = dtb + 0.25 * -(((0.5 - gamma[i_cell]) * vere) * resib)
            gammab[i_cell] = gammab[i_cell] + -((-(dt / 4) * vere) * resib)
            vereb = vereb + (-(dt / 4) * (0.5 - gamma[i_cell])) * resib
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
            vereb = vereb + (1.0 / cell_vol[i_cell]) * reb
            cell_volb[i_cell] = cell_volb[i_cell] + -(vere / cell_vol[i_cell] ^ 2) * reb
            reb = 0.0
            ub[i_node] = ub[i_node] + (-(1.0 / 3.0) * (c[1, i_node] * skx[i_loc, i_cell] + c[2, i_node] * sky[i_loc, i_cell] + c[3, i_node] * skz[i_loc, i_cell])) * vereb
            cb[1, i_node] = cb[1, i_node] + skx[i_loc, i_cell] * ((-(1.0 / 3.0) * u[i_node]) * vereb)
            skxb[i_loc, i_cell] = skxb[i_loc, i_cell] + c[1, i_node] * ((-(1.0 / 3.0) * u[i_node]) * vereb)
            cb[2, i_node] = cb[2, i_node] + sky[i_loc, i_cell] * ((-(1.0 / 3.0) * u[i_node]) * vereb)
            skyb[i_loc, i_cell] = skyb[i_loc, i_cell] + c[2, i_node] * ((-(1.0 / 3.0) * u[i_node]) * vereb)
            cb[3, i_node] = cb[3, i_node] + skz[i_loc, i_cell] * ((-(1.0 / 3.0) * u[i_node]) * vereb)
            skzb[i_loc, i_cell] = skzb[i_loc, i_cell] + c[3, i_node] * ((-(1.0 / 3.0) * u[i_node]) * vereb)
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
