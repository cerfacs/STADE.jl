function initstacks_ttgc_b()
    cavgx_stack = Vector{Float64}()
    vere_stack = Vector{Float64}()
    re_stack = Vector{Float64}()
    aerex_stack = Vector{Float64}()
    aeresk_stack = Vector{Float64}()
    factor_stack = Vector{Float64}()
    return (cavgx_stack, vere_stack, re_stack, aerex_stack, aeresk_stack, factor_stack)
end

function ttgc_hv(u, ub, i_cell_to_node, cell_vol, cell_volb, skx, skxb, sky, skyb, skz, skzb, i_ncell, i_nnode, c, cb, dt, dtb, beta, betab, gamma, gammab, res, resb, up, upb, loss, lossb, ud, ubd, cell_vold, cell_volbd, skxd, skxbd, skyd, skybd, skzd, skzbd, cd, cbd, dtd, dtbd, betad, betabd, gammad, gammabd, resd, resbd, upd, upbd, lossd, lossbd, cavgx_stack, vere_stack, re_stack, aerex_stack, aeresk_stack, factor_stack)
    cavgx_stack_d = Vector{Float64}()
    vere_stack_d = Vector{Float64}()
    re_stack_d = Vector{Float64}()
    aerex_stack_d = Vector{Float64}()
    aeresk_stack_d = Vector{Float64}()
    factor_stack_d = Vector{Float64}()
    aeresk = 0.0
    aerex = 0.0
    auxres = 0.0
    cavgx = 0.0
    factor = 0.0
    re = 0.0
    resi = 0.0
    vere = 0.0
    aereskb = 0.0
    aerexb = 0.0
    auxresb = 0.0
    cavgxb = 0.0
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
    cavgxd = 0.0
    cavgxbd = 0.0
    factord = 0.0
    factorbd = 0.0
    red = 0.0
    rebd = 0.0
    resid = 0.0
    resibd = 0.0
    vered = 0.0
    verebd = 0.0
    for i_cell = 1:i_ncell
        push!(cavgx_stack_d, cavgxd)
        push!(cavgx_stack, cavgx)
        cavgxd = 0.0
        cavgx = 0.0
        for i_loc = 1:4
            i_node = i_cell_to_node[i_loc, i_cell]
            push!(cavgx_stack_d, cavgxd)
            push!(cavgx_stack, cavgx)
            cavgxd = cavgxd + 0.25 * cd[1, i_node]
            cavgx = cavgx + c[1, i_node] / 4
        end
        for i_loc = 1:4
            i_node = i_cell_to_node[i_loc, i_cell]
            push!(vere_stack_d, vered)
            push!(vere_stack, vere)
            vered = (-(1.0 / 3.0) * (c[1, i_node] * skx[i_loc, i_cell] + c[2, i_node] * sky[i_loc, i_cell] + c[3, i_node] * skz[i_loc, i_cell])) * ud[i_node] + (-(1.0 / 3.0) * u[i_node]) * (((skx[i_loc, i_cell] * cd[1, i_node] + c[1, i_node] * skxd[i_loc, i_cell]) + (sky[i_loc, i_cell] * cd[2, i_node] + c[2, i_node] * skyd[i_loc, i_cell])) + (skz[i_loc, i_cell] * cd[3, i_node] + c[3, i_node] * skzd[i_loc, i_cell]))
            vere = -(1.0 / 3.0) * u[i_node] * (c[1, i_node] * skx[i_loc, i_cell] + c[2, i_node] * sky[i_loc, i_cell] + c[3, i_node] * skz[i_loc, i_cell])
            push!(re_stack_d, red)
            push!(re_stack, re)
            red = (1.0 / cell_vol[i_cell]) * vered + -(vere / cell_vol[i_cell] ^ 2) * cell_vold[i_cell]
            re = vere / cell_vol[i_cell]
            push!(aerex_stack_d, aerexd)
            push!(aerex_stack, aerex)
            aerexd = re * cavgxd + cavgx * red
            aerex = cavgx * re
            resid = (((0.5 - gamma[i_cell]) * vere) * -(0.25dtd) + (-(dt / 4) * vere) * -(gammad[i_cell])) + (-(dt / 4) * (0.5 - gamma[i_cell])) * vered
            resi = -(dt / 4) * (0.5 - gamma[i_cell]) * vere
            for i_k = 1:4
                push!(aeresk_stack_d, aereskd)
                push!(aeresk_stack, aeresk)
                aereskd = ((skx[i_k, i_cell] * aerexd + aerex * skxd[i_k, i_cell]) + (sky[i_k, i_cell] * aereyd + aerey * skyd[i_k, i_cell])) + (skz[i_k, i_cell] * aerezd + aerez * skzd[i_k, i_cell])
                aeresk = aerex * skx[i_k, i_cell] + aerey * sky[i_k, i_cell] + aerez * skz[i_k, i_cell]
                push!(factor_stack_d, factord)
                push!(factor_stack, factor)
                factord = dt ^ 2 * aereskd + aeresk * ((2dt) * dtd)
                factor = aeresk * dt ^ 2
                auxresd = resid + (beta[i_cell] * factord + factor * betad[i_cell])
                auxres = resi + factor * beta[i_cell]
                i_k_node = i_cell_to_node[i_k, i_cell]
                resd[i_k_node] = resd[i_k_node] + auxresd
                res[i_k_node] = res[i_k_node] + auxres
            end
            push!(aeresk_stack_d, aereskd)
            push!(aeresk_stack, aeresk)
            push!(factor_stack_d, factord)
            push!(factor_stack, factor)
        end
        push!(aeresk_stack_d, aereskd)
        push!(aeresk_stack, aeresk)
        push!(aerex_stack_d, aerexd)
        push!(aerex_stack, aerex)
        push!(cavgx_stack_d, cavgxd)
        push!(cavgx_stack, cavgx)
        push!(factor_stack_d, factord)
        push!(factor_stack, factor)
        push!(re_stack_d, red)
        push!(re_stack, re)
        push!(vere_stack_d, vered)
        push!(vere_stack, vere)
    end
    for i_seq_node = 1:i_nnode
        lossd[1] = lossd[1] + (2 * (u[i_seq_node] + up[i_seq_node])) * (ud[i_seq_node] + upd[i_seq_node])
        loss[1] = loss[1] + (u[i_seq_node] + up[i_seq_node]) ^ 2
    end
    push!(aeresk_stack_d, aereskd)
    push!(aeresk_stack, aeresk)
    push!(aerex_stack_d, aerexd)
    push!(aerex_stack, aerex)
    push!(cavgx_stack_d, cavgxd)
    push!(cavgx_stack, cavgx)
    push!(factor_stack_d, factord)
    push!(factor_stack, factor)
    push!(re_stack_d, red)
    push!(re_stack, re)
    push!(vere_stack_d, vered)
    push!(vere_stack, vere)
    aereskd = pop!(aeresk_stack_d)
    aeresk = pop!(aeresk_stack)
    aerexd = pop!(aerex_stack_d)
    aerex = pop!(aerex_stack)
    cavgxd = pop!(cavgx_stack_d)
    cavgx = pop!(cavgx_stack)
    factord = pop!(factor_stack_d)
    factor = pop!(factor_stack)
    red = pop!(re_stack_d)
    re = pop!(re_stack)
    vered = pop!(vere_stack_d)
    vere = pop!(vere_stack)
    for i_seq_node = i_nnode:-1:1
        ubd[i_seq_node] = ubd[i_seq_node] + (lossb[1] * (2 * (ud[i_seq_node] + upd[i_seq_node])) + (2 * (u[i_seq_node] + up[i_seq_node])) * lossbd[1])
        ub[i_seq_node] = ub[i_seq_node] + (2 * (u[i_seq_node] + up[i_seq_node])) * lossb[1]
        upbd[i_seq_node] = upbd[i_seq_node] + (lossb[1] * (2 * (ud[i_seq_node] + upd[i_seq_node])) + (2 * (u[i_seq_node] + up[i_seq_node])) * lossbd[1])
        upb[i_seq_node] = upb[i_seq_node] + (2 * (u[i_seq_node] + up[i_seq_node])) * lossb[1]
    end
    for i_cell = i_ncell:-1:1
        aereskd = pop!(aeresk_stack_d)
        aeresk = pop!(aeresk_stack)
        aerexd = pop!(aerex_stack_d)
        aerex = pop!(aerex_stack)
        cavgxd = pop!(cavgx_stack_d)
        cavgx = pop!(cavgx_stack)
        factord = pop!(factor_stack_d)
        factor = pop!(factor_stack)
        red = pop!(re_stack_d)
        re = pop!(re_stack)
        vered = pop!(vere_stack_d)
        vere = pop!(vere_stack)
        for i_loc = 4:-1:1
            aereskd = pop!(aeresk_stack_d)
            aeresk = pop!(aeresk_stack)
            factord = pop!(factor_stack_d)
            factor = pop!(factor_stack)
            i_node = i_cell_to_node[i_loc, i_cell]
            for i_k = 4:-1:1
                i_k_node = i_cell_to_node[i_k, i_cell]
                auxresbd = auxresbd + resbd[i_k_node]
                auxresb = auxresb + resb[i_k_node]
                resibd = resibd + auxresbd
                resib = resib + auxresb
                factorbd = factorbd + (auxresb * betad[i_cell] + beta[i_cell] * auxresbd)
                factorb = factorb + beta[i_cell] * auxresb
                betabd[i_cell] = betabd[i_cell] + (auxresb * factord + factor * auxresbd)
                betab[i_cell] = betab[i_cell] + factor * auxresb
                auxresbd = 0.0
                auxresb = 0.0
                factord = pop!(factor_stack_d)
                factor = pop!(factor_stack)
                aereskbd = aereskbd + (factorb * ((2dt) * dtd) + dt ^ 2 * factorbd)
                aereskb = aereskb + dt ^ 2 * factorb
                dtbd = dtbd + ((aeresk * factorb) * (2dtd) + (2dt) * (factorb * aereskd + aeresk * factorbd))
                dtb = dtb + (2dt) * (aeresk * factorb)
                factorbd = 0.0
                factorb = 0.0
                aereskd = pop!(aeresk_stack_d)
                aeresk = pop!(aeresk_stack)
                aerexbd = aerexbd + (aereskb * skxd[i_k, i_cell] + skx[i_k, i_cell] * aereskbd)
                aerexb = aerexb + skx[i_k, i_cell] * aereskb
                skxbd[i_k, i_cell] = skxbd[i_k, i_cell] + (aereskb * aerexd + aerex * aereskbd)
                skxb[i_k, i_cell] = skxb[i_k, i_cell] + aerex * aereskb
                skybd[i_k, i_cell] = skybd[i_k, i_cell] + (aereskb * aereyd + aerey * aereskbd)
                skyb[i_k, i_cell] = skyb[i_k, i_cell] + aerey * aereskb
                skzbd[i_k, i_cell] = skzbd[i_k, i_cell] + (aereskb * aerezd + aerez * aereskbd)
                skzb[i_k, i_cell] = skzb[i_k, i_cell] + aerez * aereskb
                aereskbd = 0.0
                aereskb = 0.0
            end
            dtbd = dtbd + 0.25 * -((resib * (vere * -(gammad[i_cell]) + (0.5 - gamma[i_cell]) * vered) + ((0.5 - gamma[i_cell]) * vere) * resibd))
            dtb = dtb + 0.25 * -(((0.5 - gamma[i_cell]) * vere) * resib)
            gammabd[i_cell] = gammabd[i_cell] + -((resib * (vere * -(0.25dtd) + -(dt / 4) * vered) + (-(dt / 4) * vere) * resibd))
            gammab[i_cell] = gammab[i_cell] + -((-(dt / 4) * vere) * resib)
            verebd = verebd + (resib * ((0.5 - gamma[i_cell]) * -(0.25dtd) + -(dt / 4) * -(gammad[i_cell])) + (-(dt / 4) * (0.5 - gamma[i_cell])) * resibd)
            vereb = vereb + (-(dt / 4) * (0.5 - gamma[i_cell])) * resib
            resibd = 0.0
            resib = 0.0
            aerexd = pop!(aerex_stack_d)
            aerex = pop!(aerex_stack)
            cavgxbd = cavgxbd + (aerexb * red + re * aerexbd)
            cavgxb = cavgxb + re * aerexb
            rebd = rebd + (aerexb * cavgxd + cavgx * aerexbd)
            reb = reb + cavgx * aerexb
            aerexbd = 0.0
            aerexb = 0.0
            red = pop!(re_stack_d)
            re = pop!(re_stack)
            verebd = verebd + (reb * (-(1.0 / cell_vol[i_cell] ^ 2) * cell_vold[i_cell]) + (1.0 / cell_vol[i_cell]) * rebd)
            vereb = vereb + (1.0 / cell_vol[i_cell]) * reb
            cell_volbd[i_cell] = cell_volbd[i_cell] + (reb * -(((1.0 / cell_vol[i_cell] ^ 2) * vered + -(vere / (cell_vol[i_cell] ^ 2) ^ 2) * ((2 * cell_vol[i_cell]) * cell_vold[i_cell]))) + -(vere / cell_vol[i_cell] ^ 2) * rebd)
            cell_volb[i_cell] = cell_volb[i_cell] + -(vere / cell_vol[i_cell] ^ 2) * reb
            rebd = 0.0
            reb = 0.0
            vered = pop!(vere_stack_d)
            vere = pop!(vere_stack)
            ubd[i_node] = ubd[i_node] + (vereb * (-(1.0 / 3.0) * (((skx[i_loc, i_cell] * cd[1, i_node] + c[1, i_node] * skxd[i_loc, i_cell]) + (sky[i_loc, i_cell] * cd[2, i_node] + c[2, i_node] * skyd[i_loc, i_cell])) + (skz[i_loc, i_cell] * cd[3, i_node] + c[3, i_node] * skzd[i_loc, i_cell]))) + (-(1.0 / 3.0) * (c[1, i_node] * skx[i_loc, i_cell] + c[2, i_node] * sky[i_loc, i_cell] + c[3, i_node] * skz[i_loc, i_cell])) * verebd)
            ub[i_node] = ub[i_node] + (-(1.0 / 3.0) * (c[1, i_node] * skx[i_loc, i_cell] + c[2, i_node] * sky[i_loc, i_cell] + c[3, i_node] * skz[i_loc, i_cell])) * vereb
            cbd[1, i_node] = cbd[1, i_node] + (((-(1.0 / 3.0) * u[i_node]) * vereb) * skxd[i_loc, i_cell] + skx[i_loc, i_cell] * (vereb * (-(1.0 / 3.0) * ud[i_node]) + (-(1.0 / 3.0) * u[i_node]) * verebd))
            cb[1, i_node] = cb[1, i_node] + skx[i_loc, i_cell] * ((-(1.0 / 3.0) * u[i_node]) * vereb)
            skxbd[i_loc, i_cell] = skxbd[i_loc, i_cell] + (((-(1.0 / 3.0) * u[i_node]) * vereb) * cd[1, i_node] + c[1, i_node] * (vereb * (-(1.0 / 3.0) * ud[i_node]) + (-(1.0 / 3.0) * u[i_node]) * verebd))
            skxb[i_loc, i_cell] = skxb[i_loc, i_cell] + c[1, i_node] * ((-(1.0 / 3.0) * u[i_node]) * vereb)
            cbd[2, i_node] = cbd[2, i_node] + (((-(1.0 / 3.0) * u[i_node]) * vereb) * skyd[i_loc, i_cell] + sky[i_loc, i_cell] * (vereb * (-(1.0 / 3.0) * ud[i_node]) + (-(1.0 / 3.0) * u[i_node]) * verebd))
            cb[2, i_node] = cb[2, i_node] + sky[i_loc, i_cell] * ((-(1.0 / 3.0) * u[i_node]) * vereb)
            skybd[i_loc, i_cell] = skybd[i_loc, i_cell] + (((-(1.0 / 3.0) * u[i_node]) * vereb) * cd[2, i_node] + c[2, i_node] * (vereb * (-(1.0 / 3.0) * ud[i_node]) + (-(1.0 / 3.0) * u[i_node]) * verebd))
            skyb[i_loc, i_cell] = skyb[i_loc, i_cell] + c[2, i_node] * ((-(1.0 / 3.0) * u[i_node]) * vereb)
            cbd[3, i_node] = cbd[3, i_node] + (((-(1.0 / 3.0) * u[i_node]) * vereb) * skzd[i_loc, i_cell] + skz[i_loc, i_cell] * (vereb * (-(1.0 / 3.0) * ud[i_node]) + (-(1.0 / 3.0) * u[i_node]) * verebd))
            cb[3, i_node] = cb[3, i_node] + skz[i_loc, i_cell] * ((-(1.0 / 3.0) * u[i_node]) * vereb)
            skzbd[i_loc, i_cell] = skzbd[i_loc, i_cell] + (((-(1.0 / 3.0) * u[i_node]) * vereb) * cd[3, i_node] + c[3, i_node] * (vereb * (-(1.0 / 3.0) * ud[i_node]) + (-(1.0 / 3.0) * u[i_node]) * verebd))
            skzb[i_loc, i_cell] = skzb[i_loc, i_cell] + c[3, i_node] * ((-(1.0 / 3.0) * u[i_node]) * vereb)
            verebd = 0.0
            vereb = 0.0
        end
        for i_loc = 4:-1:1
            i_node = i_cell_to_node[i_loc, i_cell]
            cavgxd = pop!(cavgx_stack_d)
            cavgx = pop!(cavgx_stack)
            cbd[1, i_node] = cbd[1, i_node] + 0.25cavgxbd
            cb[1, i_node] = cb[1, i_node] + 0.25cavgxb
        end
        cavgxd = pop!(cavgx_stack_d)
        cavgx = pop!(cavgx_stack)
        cavgxbd = 0.0
        cavgxb = 0.0
    end
    return (dtb, dtbd)
end

function ttgc(u, i_cell_to_node, cell_vol, skx, sky, skz, i_ncell, i_nnode, c, dt, beta, gamma, res, up, loss)
    for i_cell = 1:i_ncell
        cavgx = 0.0
        for i_loc = 1:4
            i_node = i_cell_to_node[i_loc, i_cell]
            cavgx = cavgx + c[1, i_node] / 4
        end
        for i_loc = 1:4
            i_node = i_cell_to_node[i_loc, i_cell]
            vere = -(1.0 / 3.0) * u[i_node] * (c[1, i_node] * skx[i_loc, i_cell] + c[2, i_node] * sky[i_loc, i_cell] + c[3, i_node] * skz[i_loc, i_cell])
            re = vere / cell_vol[i_cell]
            aerex = cavgx * re
            resi = -(dt / 4) * (0.5 - gamma[i_cell]) * vere
            for i_k = 1:4
                aeresk = aerex * skx[i_k, i_cell] + aerey * sky[i_k, i_cell] + aerez * skz[i_k, i_cell]
                factor = aeresk * dt ^ 2
                auxres = resi + factor * beta[i_cell]
                i_k_node = i_cell_to_node[i_k, i_cell]
                res[i_k_node] = res[i_k_node] + auxres
            end
        end
    end
    for i_seq_node = 1:i_nnode
        loss[1] = loss[1] + (u[i_seq_node] + up[i_seq_node]) ^ 2
    end
end
