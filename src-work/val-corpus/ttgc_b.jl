function initstacks_ttgc_b()
    cavgx_stack = Vector{Float64}()
    cavgy_stack = Vector{Float64}()
    cavgz_stack = Vector{Float64}()
    vere_stack = Vector{Float64}()
    re_stack = Vector{Float64}()
    aerex_stack = Vector{Float64}()
    aerey_stack = Vector{Float64}()
    aerez_stack = Vector{Float64}()
    aeresk_stack = Vector{Float64}()
    factor_stack = Vector{Float64}()
    mup_stack = Vector{Float64}()
    auxu_stack = Vector{Float64}()
    up_stack = Vector{Float64}()
    return (cavgx_stack, cavgy_stack, cavgz_stack, vere_stack, re_stack, aerex_stack, aerey_stack, aerez_stack, aeresk_stack, factor_stack, mup_stack, auxu_stack, up_stack)
end

function ttgc_b(u, ub, uref, urefb, i_cell_to_node, cell_vol, cell_volb, node_vol, node_volb, skx, skxb, sky, skyb, skz, skzb, i_ncell, i_nnode, c, cb, dt, dtb, beta, betab, gamma, gammab, i_njac, res, resb, res2, res2b, up, upb, mup, mupb, loss, lossb, cavgx_stack, cavgy_stack, cavgz_stack, vere_stack, re_stack, aerex_stack, aerey_stack, aerez_stack, aeresk_stack, factor_stack, mup_stack, auxu_stack, up_stack)
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
        push!(cavgx_stack, cavgx)
        cavgx = 0.0
        push!(cavgy_stack, cavgy)
        cavgy = 0.0
        push!(cavgz_stack, cavgz)
        cavgz = 0.0
        for i_seq_loc = 1:4
            i_node = i_cell_to_node[i_seq_loc, i_cell]
            cavgx = cavgx + c[1, i_node] / 4
            cavgy = cavgy + c[2, i_node] / 4
            cavgz = cavgz + c[3, i_node] / 4
        end
        for i_seq_loc = 1:4
            i_node = i_cell_to_node[i_seq_loc, i_cell]
            push!(vere_stack, vere)
            vere = -(1.0 / 3.0) * u[i_node] * (c[1, i_node] * skx[i_seq_loc, i_cell] + c[2, i_node] * sky[i_seq_loc, i_cell] + c[3, i_node] * skz[i_seq_loc, i_cell])
            push!(re_stack, re)
            re = vere / cell_vol[i_cell]
            push!(aerex_stack, aerex)
            aerex = cavgx * re
            push!(aerey_stack, aerey)
            aerey = cavgy * re
            push!(aerez_stack, aerez)
            aerez = cavgz * re
            resi = -(dt / 4) * (0.5 - gamma[i_cell]) * vere
            for i_seq_k = 1:4
                push!(aeresk_stack, aeresk)
                aeresk = aerex * skx[i_seq_k, i_cell] + aerey * sky[i_seq_k, i_cell] + aerez * skz[i_seq_k, i_cell]
                push!(factor_stack, factor)
                factor = (dt ^ 2 / 3) * aeresk
                auxres = resi + factor * beta[i_cell]
                auxres2 = factor * gamma[i_cell]
                i_seq_k_node = i_cell_to_node[i_seq_k, i_cell]
                res[i_seq_k_node] = res[i_seq_k_node] + auxres
                res2[i_seq_k_node] = res2[i_seq_k_node] + auxres2
            end
            push!(aeresk_stack, aeresk)
            push!(factor_stack, factor)
        end
        push!(aeresk_stack, aeresk)
        push!(aerex_stack, aerex)
        push!(aerey_stack, aerey)
        push!(aerez_stack, aerez)
        push!(cavgx_stack, cavgx)
        push!(cavgy_stack, cavgy)
        push!(cavgz_stack, cavgz)
        push!(factor_stack, factor)
        push!(re_stack, re)
        push!(vere_stack, vere)
    end
    for i_node = 1:i_nnode
        up[i_node] = res[i_node] / node_vol[i_node]
    end
    for i_seq_ = 1:i_njac
        for i_node = 1:i_nnode
            push!(mup_stack, mup[i_node])
            mup[i_node] = 0.0
        end
        for i_cell = 1:i_ncell
            i_node1 = i_cell_to_node[1, i_cell]
            i_node2 = i_cell_to_node[2, i_cell]
            i_node3 = i_cell_to_node[3, i_cell]
            i_node4 = i_cell_to_node[4, i_cell]
            push!(auxu_stack, auxu)
            auxu = up[i_node1] + up[i_node2] + up[i_node3] + up[i_node4]
            push!(mup_stack, mup[i_node1])
            mup[i_node1] = mup[i_node1] + ((auxu + up[i_node1]) * cell_vol[i_cell]) / 20.0
            push!(mup_stack, mup[i_node2])
            mup[i_node2] = mup[i_node2] + ((auxu + up[i_node2]) * cell_vol[i_cell]) / 20.0
            push!(mup_stack, mup[i_node3])
            mup[i_node3] = mup[i_node3] + ((auxu + up[i_node3]) * cell_vol[i_cell]) / 20.0
            push!(mup_stack, mup[i_node4])
            mup[i_node4] = mup[i_node4] + ((auxu + up[i_node4]) * cell_vol[i_cell]) / 20.0
        end
        for i_node = 1:i_nnode
            push!(up_stack, up[i_node])
            up[i_node] = up[i_node] + (res[i_node] - mup[i_node]) / node_vol[i_node]
        end
        push!(auxu_stack, auxu)
    end
    for i_cell = 1:i_ncell
        for i_seq_loc = 1:4
            i_node = i_cell_to_node[i_seq_loc, i_cell]
            push!(vere_stack, vere)
            vere = -(1.0 / 3.0) * (u[i_node] + up[i_node]) * (c[1, i_node] * skx[i_seq_loc, i_cell] + c[2, i_node] * sky[i_seq_loc, i_cell])
            resi = -(dt / 4) * vere
            for i_seq_k = 1:4
                i_seq_k_node = i_cell_to_node[i_seq_k, i_cell]
                res2[i_seq_k_node] = res2[i_seq_k_node] + resi
            end
        end
        push!(vere_stack, vere)
    end
    for i_node = 1:i_nnode
        push!(up_stack, up[i_node])
        up[i_node] = res2[i_node] / node_vol[i_node]
    end
    for i_seq_ = 1:i_njac
        for i_node = 1:i_nnode
            push!(mup_stack, mup[i_node])
            mup[i_node] = 0.0
        end
        for i_cell = 1:i_ncell
            i_node1 = i_cell_to_node[1, i_cell]
            i_node2 = i_cell_to_node[2, i_cell]
            i_node3 = i_cell_to_node[3, i_cell]
            i_node4 = i_cell_to_node[4, i_cell]
            push!(auxu_stack, auxu)
            auxu = up[i_node1] + up[i_node2] + up[i_node3] + up[i_node4]
            push!(mup_stack, mup[i_node1])
            mup[i_node1] = mup[i_node1] + ((auxu + up[i_node1]) * cell_vol[i_cell]) / 20.0
            push!(mup_stack, mup[i_node2])
            mup[i_node2] = mup[i_node2] + ((auxu + up[i_node2]) * cell_vol[i_cell]) / 20.0
            push!(mup_stack, mup[i_node3])
            mup[i_node3] = mup[i_node3] + ((auxu + up[i_node3]) * cell_vol[i_cell]) / 20.0
            push!(mup_stack, mup[i_node4])
            mup[i_node4] = mup[i_node4] + ((auxu + up[i_node4]) * cell_vol[i_cell]) / 20.0
        end
        for i_node = 1:i_nnode
            push!(up_stack, up[i_node])
            up[i_node] = up[i_node] + (res2[i_node] - mup[i_node]) / node_vol[i_node]
        end
        push!(auxu_stack, auxu)
    end
    for i_seq_node = 1:i_nnode
        loss[1] = loss[1] + ((u[i_seq_node] + up[i_seq_node]) - uref[i_seq_node]) ^ 2
    end
    push!(aeresk_stack, aeresk)
    push!(aerex_stack, aerex)
    push!(aerey_stack, aerey)
    push!(aerez_stack, aerez)
    push!(auxu_stack, auxu)
    push!(cavgx_stack, cavgx)
    push!(cavgy_stack, cavgy)
    push!(cavgz_stack, cavgz)
    push!(factor_stack, factor)
    push!(re_stack, re)
    push!(vere_stack, vere)
    aeresk = pop!(aeresk_stack)
    aerex = pop!(aerex_stack)
    aerey = pop!(aerey_stack)
    aerez = pop!(aerez_stack)
    auxu = pop!(auxu_stack)
    cavgx = pop!(cavgx_stack)
    cavgy = pop!(cavgy_stack)
    cavgz = pop!(cavgz_stack)
    factor = pop!(factor_stack)
    re = pop!(re_stack)
    vere = pop!(vere_stack)
    for i_seq_node = i_nnode:-1:1
        ub[i_seq_node] = ub[i_seq_node] + (2 * ((u[i_seq_node] + up[i_seq_node]) - uref[i_seq_node])) * lossb[1]
        upb[i_seq_node] = upb[i_seq_node] + (2 * ((u[i_seq_node] + up[i_seq_node]) - uref[i_seq_node])) * lossb[1]
        urefb[i_seq_node] = urefb[i_seq_node] + -((2 * ((u[i_seq_node] + up[i_seq_node]) - uref[i_seq_node])) * lossb[1])
    end
    for i_seq_ = i_njac:-1:1
        auxu = pop!(auxu_stack)
        for i_node = i_nnode:-1:1
            up[i_node] = pop!(up_stack)
            res2b[i_node] = res2b[i_node] + (1.0 / node_vol[i_node]) * upb[i_node]
            mupb[i_node] = mupb[i_node] + -((1.0 / node_vol[i_node]) * upb[i_node])
            node_volb[i_node] = node_volb[i_node] + -((res2[i_node] - mup[i_node]) / node_vol[i_node] ^ 2) * upb[i_node]
        end
        for i_cell = i_ncell:-1:1
            i_node1 = i_cell_to_node[1, i_cell]
            i_node2 = i_cell_to_node[2, i_cell]
            i_node3 = i_cell_to_node[3, i_cell]
            i_node4 = i_cell_to_node[4, i_cell]
            mup[i_node4] = pop!(mup_stack)
            auxub = auxub + cell_vol[i_cell] * (0.05 * mupb[i_node4])
            upb[i_node4] = upb[i_node4] + cell_vol[i_cell] * (0.05 * mupb[i_node4])
            cell_volb[i_cell] = cell_volb[i_cell] + (auxu + up[i_node4]) * (0.05 * mupb[i_node4])
            mup[i_node3] = pop!(mup_stack)
            auxub = auxub + cell_vol[i_cell] * (0.05 * mupb[i_node3])
            upb[i_node3] = upb[i_node3] + cell_vol[i_cell] * (0.05 * mupb[i_node3])
            cell_volb[i_cell] = cell_volb[i_cell] + (auxu + up[i_node3]) * (0.05 * mupb[i_node3])
            mup[i_node2] = pop!(mup_stack)
            auxub = auxub + cell_vol[i_cell] * (0.05 * mupb[i_node2])
            upb[i_node2] = upb[i_node2] + cell_vol[i_cell] * (0.05 * mupb[i_node2])
            cell_volb[i_cell] = cell_volb[i_cell] + (auxu + up[i_node2]) * (0.05 * mupb[i_node2])
            mup[i_node1] = pop!(mup_stack)
            auxub = auxub + cell_vol[i_cell] * (0.05 * mupb[i_node1])
            upb[i_node1] = upb[i_node1] + cell_vol[i_cell] * (0.05 * mupb[i_node1])
            cell_volb[i_cell] = cell_volb[i_cell] + (auxu + up[i_node1]) * (0.05 * mupb[i_node1])
            auxu = pop!(auxu_stack)
            upb[i_node1] = upb[i_node1] + auxub
            upb[i_node2] = upb[i_node2] + auxub
            upb[i_node3] = upb[i_node3] + auxub
            upb[i_node4] = upb[i_node4] + auxub
            auxub = 0.0
        end
        for i_node = i_nnode:-1:1
            mup[i_node] = pop!(mup_stack)
            mupb[i_node] = 0.0
        end
    end
    for i_node = i_nnode:-1:1
        up[i_node] = pop!(up_stack)
        res2b[i_node] = res2b[i_node] + (1.0 / node_vol[i_node]) * upb[i_node]
        node_volb[i_node] = node_volb[i_node] + -(res2[i_node] / node_vol[i_node] ^ 2) * upb[i_node]
        upb[i_node] = 0.0
    end
    for i_cell = i_ncell:-1:1
        vere = pop!(vere_stack)
        for i_seq_loc = 4:-1:1
            i_node = i_cell_to_node[i_seq_loc, i_cell]
            for i_seq_k = 4:-1:1
                i_seq_k_node = i_cell_to_node[i_seq_k, i_cell]
                resib = resib + res2b[i_seq_k_node]
            end
            dtb = dtb + 0.25 * -(vere * resib)
            vereb = vereb + -(dt / 4) * resib
            resib = 0.0
            vere = pop!(vere_stack)
            ub[i_node] = ub[i_node] + (-(1.0 / 3.0) * (c[1, i_node] * skx[i_seq_loc, i_cell] + c[2, i_node] * sky[i_seq_loc, i_cell])) * vereb
            upb[i_node] = upb[i_node] + (-(1.0 / 3.0) * (c[1, i_node] * skx[i_seq_loc, i_cell] + c[2, i_node] * sky[i_seq_loc, i_cell])) * vereb
            cb[1, i_node] = cb[1, i_node] + skx[i_seq_loc, i_cell] * ((-(1.0 / 3.0) * (u[i_node] + up[i_node])) * vereb)
            skxb[i_seq_loc, i_cell] = skxb[i_seq_loc, i_cell] + c[1, i_node] * ((-(1.0 / 3.0) * (u[i_node] + up[i_node])) * vereb)
            cb[2, i_node] = cb[2, i_node] + sky[i_seq_loc, i_cell] * ((-(1.0 / 3.0) * (u[i_node] + up[i_node])) * vereb)
            skyb[i_seq_loc, i_cell] = skyb[i_seq_loc, i_cell] + c[2, i_node] * ((-(1.0 / 3.0) * (u[i_node] + up[i_node])) * vereb)
            vereb = 0.0
        end
    end
    for i_seq_ = i_njac:-1:1
        auxu = pop!(auxu_stack)
        for i_node = i_nnode:-1:1
            up[i_node] = pop!(up_stack)
            resb[i_node] = resb[i_node] + (1.0 / node_vol[i_node]) * upb[i_node]
            mupb[i_node] = mupb[i_node] + -((1.0 / node_vol[i_node]) * upb[i_node])
            node_volb[i_node] = node_volb[i_node] + -((res[i_node] - mup[i_node]) / node_vol[i_node] ^ 2) * upb[i_node]
        end
        for i_cell = i_ncell:-1:1
            i_node1 = i_cell_to_node[1, i_cell]
            i_node2 = i_cell_to_node[2, i_cell]
            i_node3 = i_cell_to_node[3, i_cell]
            i_node4 = i_cell_to_node[4, i_cell]
            mup[i_node4] = pop!(mup_stack)
            auxub = auxub + cell_vol[i_cell] * (0.05 * mupb[i_node4])
            upb[i_node4] = upb[i_node4] + cell_vol[i_cell] * (0.05 * mupb[i_node4])
            cell_volb[i_cell] = cell_volb[i_cell] + (auxu + up[i_node4]) * (0.05 * mupb[i_node4])
            mup[i_node3] = pop!(mup_stack)
            auxub = auxub + cell_vol[i_cell] * (0.05 * mupb[i_node3])
            upb[i_node3] = upb[i_node3] + cell_vol[i_cell] * (0.05 * mupb[i_node3])
            cell_volb[i_cell] = cell_volb[i_cell] + (auxu + up[i_node3]) * (0.05 * mupb[i_node3])
            mup[i_node2] = pop!(mup_stack)
            auxub = auxub + cell_vol[i_cell] * (0.05 * mupb[i_node2])
            upb[i_node2] = upb[i_node2] + cell_vol[i_cell] * (0.05 * mupb[i_node2])
            cell_volb[i_cell] = cell_volb[i_cell] + (auxu + up[i_node2]) * (0.05 * mupb[i_node2])
            mup[i_node1] = pop!(mup_stack)
            auxub = auxub + cell_vol[i_cell] * (0.05 * mupb[i_node1])
            upb[i_node1] = upb[i_node1] + cell_vol[i_cell] * (0.05 * mupb[i_node1])
            cell_volb[i_cell] = cell_volb[i_cell] + (auxu + up[i_node1]) * (0.05 * mupb[i_node1])
            auxu = pop!(auxu_stack)
            upb[i_node1] = upb[i_node1] + auxub
            upb[i_node2] = upb[i_node2] + auxub
            upb[i_node3] = upb[i_node3] + auxub
            upb[i_node4] = upb[i_node4] + auxub
            auxub = 0.0
        end
        for i_node = i_nnode:-1:1
            mup[i_node] = pop!(mup_stack)
            mupb[i_node] = 0.0
        end
    end
    for i_node = 1:i_nnode
        resb[i_node] = resb[i_node] + (1.0 / node_vol[i_node]) * upb[i_node]
        node_volb[i_node] = node_volb[i_node] + -(res[i_node] / node_vol[i_node] ^ 2) * upb[i_node]
        upb[i_node] = 0.0
    end
    for i_cell = i_ncell:-1:1
        aeresk = pop!(aeresk_stack)
        aerex = pop!(aerex_stack)
        aerey = pop!(aerey_stack)
        aerez = pop!(aerez_stack)
        cavgx = pop!(cavgx_stack)
        cavgy = pop!(cavgy_stack)
        cavgz = pop!(cavgz_stack)
        factor = pop!(factor_stack)
        re = pop!(re_stack)
        vere = pop!(vere_stack)
        for i_seq_loc = 4:-1:1
            aeresk = pop!(aeresk_stack)
            factor = pop!(factor_stack)
            i_node = i_cell_to_node[i_seq_loc, i_cell]
            for i_seq_k = 4:-1:1
                i_seq_k_node = i_cell_to_node[i_seq_k, i_cell]
                auxres2b = auxres2b + res2b[i_seq_k_node]
                auxresb = auxresb + resb[i_seq_k_node]
                factorb = factorb + gamma[i_cell] * auxres2b
                gammab[i_cell] = gammab[i_cell] + factor * auxres2b
                auxres2b = 0.0
                resib = resib + auxresb
                factorb = factorb + beta[i_cell] * auxresb
                betab[i_cell] = betab[i_cell] + factor * auxresb
                auxresb = 0.0
                factor = pop!(factor_stack)
                dtb = dtb + (2dt) * (0.3333333333333333 * (aeresk * factorb))
                aereskb = aereskb + (dt ^ 2 / 3) * factorb
                factorb = 0.0
                aeresk = pop!(aeresk_stack)
                aerexb = aerexb + skx[i_seq_k, i_cell] * aereskb
                skxb[i_seq_k, i_cell] = skxb[i_seq_k, i_cell] + aerex * aereskb
                aereyb = aereyb + sky[i_seq_k, i_cell] * aereskb
                skyb[i_seq_k, i_cell] = skyb[i_seq_k, i_cell] + aerey * aereskb
                aerezb = aerezb + skz[i_seq_k, i_cell] * aereskb
                skzb[i_seq_k, i_cell] = skzb[i_seq_k, i_cell] + aerez * aereskb
                aereskb = 0.0
            end
            dtb = dtb + 0.25 * -(((0.5 - gamma[i_cell]) * vere) * resib)
            gammab[i_cell] = gammab[i_cell] + -((-(dt / 4) * vere) * resib)
            vereb = vereb + (-(dt / 4) * (0.5 - gamma[i_cell])) * resib
            resib = 0.0
            aerez = pop!(aerez_stack)
            cavgzb = cavgzb + re * aerezb
            reb = reb + cavgz * aerezb
            aerezb = 0.0
            aerey = pop!(aerey_stack)
            cavgyb = cavgyb + re * aereyb
            reb = reb + cavgy * aereyb
            aereyb = 0.0
            aerex = pop!(aerex_stack)
            cavgxb = cavgxb + re * aerexb
            reb = reb + cavgx * aerexb
            aerexb = 0.0
            re = pop!(re_stack)
            vereb = vereb + (1.0 / cell_vol[i_cell]) * reb
            cell_volb[i_cell] = cell_volb[i_cell] + -(vere / cell_vol[i_cell] ^ 2) * reb
            reb = 0.0
            vere = pop!(vere_stack)
            ub[i_node] = ub[i_node] + (-(1.0 / 3.0) * (c[1, i_node] * skx[i_seq_loc, i_cell] + c[2, i_node] * sky[i_seq_loc, i_cell] + c[3, i_node] * skz[i_seq_loc, i_cell])) * vereb
            cb[1, i_node] = cb[1, i_node] + skx[i_seq_loc, i_cell] * ((-(1.0 / 3.0) * u[i_node]) * vereb)
            skxb[i_seq_loc, i_cell] = skxb[i_seq_loc, i_cell] + c[1, i_node] * ((-(1.0 / 3.0) * u[i_node]) * vereb)
            cb[2, i_node] = cb[2, i_node] + sky[i_seq_loc, i_cell] * ((-(1.0 / 3.0) * u[i_node]) * vereb)
            skyb[i_seq_loc, i_cell] = skyb[i_seq_loc, i_cell] + c[2, i_node] * ((-(1.0 / 3.0) * u[i_node]) * vereb)
            cb[3, i_node] = cb[3, i_node] + skz[i_seq_loc, i_cell] * ((-(1.0 / 3.0) * u[i_node]) * vereb)
            skzb[i_seq_loc, i_cell] = skzb[i_seq_loc, i_cell] + c[3, i_node] * ((-(1.0 / 3.0) * u[i_node]) * vereb)
            vereb = 0.0
        end
        for i_seq_loc = 1:4
            i_node = i_cell_to_node[i_seq_loc, i_cell]
            cavgx = cavgx + c[1, i_node] / 4
            cavgy = cavgy + c[2, i_node] / 4
            cavgz = cavgz + c[3, i_node] / 4
            i_node = i_cell_to_node[i_seq_loc, i_cell]
            cb[3, i_node] = cb[3, i_node] + 0.25cavgzb
            cb[2, i_node] = cb[2, i_node] + 0.25cavgyb
            cb[1, i_node] = cb[1, i_node] + 0.25cavgxb
        end
        cavgz = pop!(cavgz_stack)
        cavgzb = 0.0
        cavgy = pop!(cavgy_stack)
        cavgyb = 0.0
        cavgx = pop!(cavgx_stack)
        cavgxb = 0.0
    end
    return dtb
end

function ttgc(u, uref, i_cell_to_node, cell_vol, node_vol, skx, sky, skz, i_ncell, i_nnode, c, dt, beta, gamma, i_njac, res, res2, up, mup, loss)
    for i_cell = 1:i_ncell
        cavgx = 0.0
        cavgy = 0.0
        cavgz = 0.0
        for i_seq_loc = 1:4
            i_node = i_cell_to_node[i_seq_loc, i_cell]
            cavgx = cavgx + c[1, i_node] / 4
            cavgy = cavgy + c[2, i_node] / 4
            cavgz = cavgz + c[3, i_node] / 4
        end
        for i_seq_loc = 1:4
            i_node = i_cell_to_node[i_seq_loc, i_cell]
            vere = -(1.0 / 3.0) * u[i_node] * (c[1, i_node] * skx[i_seq_loc, i_cell] + c[2, i_node] * sky[i_seq_loc, i_cell] + c[3, i_node] * skz[i_seq_loc, i_cell])
            re = vere / cell_vol[i_cell]
            aerex = cavgx * re
            aerey = cavgy * re
            aerez = cavgz * re
            resi = -(dt / 4) * (0.5 - gamma[i_cell]) * vere
            for i_seq_k = 1:4
                aeresk = aerex * skx[i_seq_k, i_cell] + aerey * sky[i_seq_k, i_cell] + aerez * skz[i_seq_k, i_cell]
                factor = (dt ^ 2 / 3) * aeresk
                auxres = resi + factor * beta[i_cell]
                auxres2 = factor * gamma[i_cell]
                i_seq_k_node = i_cell_to_node[i_seq_k, i_cell]
                res[i_seq_k_node] = res[i_seq_k_node] + auxres
                res2[i_seq_k_node] = res2[i_seq_k_node] + auxres2
            end
        end
    end
    for i_node = 1:i_nnode
        up[i_node] = res[i_node] / node_vol[i_node]
    end
    for i_seq_ = 1:i_njac
        for i_node = 1:i_nnode
            mup[i_node] = 0.0
        end
        for i_cell = 1:i_ncell
            i_node1 = i_cell_to_node[1, i_cell]
            i_node2 = i_cell_to_node[2, i_cell]
            i_node3 = i_cell_to_node[3, i_cell]
            i_node4 = i_cell_to_node[4, i_cell]
            auxu = up[i_node1] + up[i_node2] + up[i_node3] + up[i_node4]
            mup[i_node1] = mup[i_node1] + ((auxu + up[i_node1]) * cell_vol[i_cell]) / 20.0
            mup[i_node2] = mup[i_node2] + ((auxu + up[i_node2]) * cell_vol[i_cell]) / 20.0
            mup[i_node3] = mup[i_node3] + ((auxu + up[i_node3]) * cell_vol[i_cell]) / 20.0
            mup[i_node4] = mup[i_node4] + ((auxu + up[i_node4]) * cell_vol[i_cell]) / 20.0
        end
        for i_node = 1:i_nnode
            up[i_node] = up[i_node] + (res[i_node] - mup[i_node]) / node_vol[i_node]
        end
    end
    for i_cell = 1:i_ncell
        for i_seq_loc = 1:4
            i_node = i_cell_to_node[i_seq_loc, i_cell]
            vere = -(1.0 / 3.0) * (u[i_node] + up[i_node]) * (c[1, i_node] * skx[i_seq_loc, i_cell] + c[2, i_node] * sky[i_seq_loc, i_cell])
            resi = -(dt / 4) * vere
            for i_seq_k = 1:4
                i_seq_k_node = i_cell_to_node[i_seq_k, i_cell]
                res2[i_seq_k_node] = res2[i_seq_k_node] + resi
            end
        end
    end
    for i_node = 1:i_nnode
        up[i_node] = res2[i_node] / node_vol[i_node]
    end
    for i_seq_ = 1:i_njac
        for i_node = 1:i_nnode
            mup[i_node] = 0.0
        end
        for i_cell = 1:i_ncell
            i_node1 = i_cell_to_node[1, i_cell]
            i_node2 = i_cell_to_node[2, i_cell]
            i_node3 = i_cell_to_node[3, i_cell]
            i_node4 = i_cell_to_node[4, i_cell]
            auxu = up[i_node1] + up[i_node2] + up[i_node3] + up[i_node4]
            mup[i_node1] = mup[i_node1] + ((auxu + up[i_node1]) * cell_vol[i_cell]) / 20.0
            mup[i_node2] = mup[i_node2] + ((auxu + up[i_node2]) * cell_vol[i_cell]) / 20.0
            mup[i_node3] = mup[i_node3] + ((auxu + up[i_node3]) * cell_vol[i_cell]) / 20.0
            mup[i_node4] = mup[i_node4] + ((auxu + up[i_node4]) * cell_vol[i_cell]) / 20.0
        end
        for i_node = 1:i_nnode
            up[i_node] = up[i_node] + (res2[i_node] - mup[i_node]) / node_vol[i_node]
        end
    end
    for i_seq_node = 1:i_nnode
        loss[1] = loss[1] + ((u[i_seq_node] + up[i_seq_node]) - uref[i_seq_node]) ^ 2
    end
end
