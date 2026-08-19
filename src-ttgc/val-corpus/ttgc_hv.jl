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
    return (cavgx_stack, cavgy_stack, cavgz_stack, vere_stack, re_stack, aerex_stack, aerey_stack, aerez_stack, aeresk_stack, factor_stack)
end

function ttgc_hv(u, ub, uref, urefb, i_cell_to_node, cell_vol, cell_volb, node_vol, node_volb, skx, skxb, sky, skyb, skz, skzb, i_ncell, i_nnode, c, cb, dt, dtb, beta, betab, gamma, gammab, i_njac, i_njacb, res, resb, res2, res2b, up, upb, mup, mupb, npernode_half, npernode_halfb, resperio, resperiob, i_node_perio, i_node_periob, loss, lossb, ud, ubd, urefd, urefbd, cell_vold, cell_volbd, node_vold, node_volbd, skxd, skxbd, skyd, skybd, skzd, skzbd, cd, cbd, dtd, dtbd, betad, betabd, gammad, gammabd, i_njacd, i_njacbd, resd, resbd, res2d, res2bd, upd, upbd, mupd, mupbd, npernode_halfd, npernode_halfbd, resperiod, resperiobd, i_node_period, i_node_periobd, lossd, lossbd, cavgx_stack, cavgy_stack, cavgz_stack, vere_stack, re_stack, aerex_stack, aerey_stack, aerez_stack, aeresk_stack, factor_stack)
    cavgx_stack_d = Vector{Float64}()
    cavgy_stack_d = Vector{Float64}()
    cavgz_stack_d = Vector{Float64}()
    vere_stack_d = Vector{Float64}()
    re_stack_d = Vector{Float64}()
    aerex_stack_d = Vector{Float64}()
    aerey_stack_d = Vector{Float64}()
    aerez_stack_d = Vector{Float64}()
    aeresk_stack_d = Vector{Float64}()
    factor_stack_d = Vector{Float64}()
    aeresk = 0.0
    aerex = 0.0
    aerey = 0.0
    aerez = 0.0
    auxres = 0.0
    auxres2 = 0.0
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
    cavgxb = 0.0
    cavgyb = 0.0
    cavgzb = 0.0
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
    auxres2d = 0.0
    auxres2bd = 0.0
    cavgxd = 0.0
    cavgxbd = 0.0
    cavgyd = 0.0
    cavgybd = 0.0
    cavgzd = 0.0
    cavgzbd = 0.0
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
        push!(cavgy_stack_d, cavgyd)
        push!(cavgy_stack, cavgy)
        cavgyd = 0.0
        cavgy = 0.0
        push!(cavgz_stack_d, cavgzd)
        push!(cavgz_stack, cavgz)
        cavgzd = 0.0
        cavgz = 0.0
        for i_loc = 1:4
            i_node = i_cell_to_node[i_loc, i_cell]
            push!(cavgx_stack_d, cavgxd)
            push!(cavgx_stack, cavgx)
            cavgxd = cavgxd + 0.25 * cd[1, i_node]
            cavgx = cavgx + c[1, i_node] / 4
            push!(cavgy_stack_d, cavgyd)
            push!(cavgy_stack, cavgy)
            cavgyd = cavgyd + 0.25 * cd[2, i_node]
            cavgy = cavgy + c[2, i_node] / 4
            push!(cavgz_stack_d, cavgzd)
            push!(cavgz_stack, cavgz)
            cavgzd = cavgzd + 0.25 * cd[3, i_node]
            cavgz = cavgz + c[3, i_node] / 4
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
            push!(aerey_stack_d, aereyd)
            push!(aerey_stack, aerey)
            aereyd = re * cavgyd + cavgy * red
            aerey = cavgy * re
            push!(aerez_stack_d, aerezd)
            push!(aerez_stack, aerez)
            aerezd = re * cavgzd + cavgz * red
            aerez = cavgz * re
            resid = (((0.5 - gamma[i_cell]) * vere) * -(0.25dtd) + (-(dt / 4) * vere) * -(gammad[i_cell])) + (-(dt / 4) * (0.5 - gamma[i_cell])) * vered
            resi = -(dt / 4) * (0.5 - gamma[i_cell]) * vere
            for i_k = 1:4
                push!(aeresk_stack_d, aereskd)
                push!(aeresk_stack, aeresk)
                aereskd = ((skx[i_k, i_cell] * aerexd + aerex * skxd[i_k, i_cell]) + (sky[i_k, i_cell] * aereyd + aerey * skyd[i_k, i_cell])) + (skz[i_k, i_cell] * aerezd + aerez * skzd[i_k, i_cell])
                aeresk = aerex * skx[i_k, i_cell] + aerey * sky[i_k, i_cell] + aerez * skz[i_k, i_cell]
                push!(factor_stack_d, factord)
                push!(factor_stack, factor)
                factord = aeresk * (0.3333333333333333 * ((2dt) * dtd)) + (dt ^ 2 / 3) * aereskd
                factor = (dt ^ 2 / 3) * aeresk
                auxresd = resid + (beta[i_cell] * factord + factor * betad[i_cell])
                auxres = resi + factor * beta[i_cell]
                auxres2d = gamma[i_cell] * factord + factor * gammad[i_cell]
                auxres2 = factor * gamma[i_cell]
                i_k_node = i_cell_to_node[i_k, i_cell]
                resd[i_k_node] = resd[i_k_node] + auxresd
                res[i_k_node] = res[i_k_node] + auxres
                res2d[i_k_node] = res2d[i_k_node] + auxres2d
                res2[i_k_node] = res2[i_k_node] + auxres2
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
        push!(aerey_stack_d, aereyd)
        push!(aerey_stack, aerey)
        push!(aerez_stack_d, aerezd)
        push!(aerez_stack, aerez)
        push!(cavgx_stack_d, cavgxd)
        push!(cavgx_stack, cavgx)
        push!(cavgy_stack_d, cavgyd)
        push!(cavgy_stack, cavgy)
        push!(cavgz_stack_d, cavgzd)
        push!(cavgz_stack, cavgz)
        push!(factor_stack_d, factord)
        push!(factor_stack, factor)
        push!(re_stack_d, red)
        push!(re_stack, re)
        push!(vere_stack_d, vered)
        push!(vere_stack, vere)
    end
    for i_seq_node = 1:i_nnode
        lossd[1] = lossd[1] + (2 * ((u[i_seq_node] + up[i_seq_node]) - uref[i_seq_node])) * ((ud[i_seq_node] + upd[i_seq_node]) + -(urefd[i_seq_node]))
        loss[1] = loss[1] + ((u[i_seq_node] + up[i_seq_node]) - uref[i_seq_node]) ^ 2
    end
    push!(aeresk_stack_d, aereskd)
    push!(aeresk_stack, aeresk)
    push!(aerex_stack_d, aerexd)
    push!(aerex_stack, aerex)
    push!(aerey_stack_d, aereyd)
    push!(aerey_stack, aerey)
    push!(aerez_stack_d, aerezd)
    push!(aerez_stack, aerez)
    push!(cavgx_stack_d, cavgxd)
    push!(cavgx_stack, cavgx)
    push!(cavgy_stack_d, cavgyd)
    push!(cavgy_stack, cavgy)
    push!(cavgz_stack_d, cavgzd)
    push!(cavgz_stack, cavgz)
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
    aereyd = pop!(aerey_stack_d)
    aerey = pop!(aerey_stack)
    aerezd = pop!(aerez_stack_d)
    aerez = pop!(aerez_stack)
    cavgxd = pop!(cavgx_stack_d)
    cavgx = pop!(cavgx_stack)
    cavgyd = pop!(cavgy_stack_d)
    cavgy = pop!(cavgy_stack)
    cavgzd = pop!(cavgz_stack_d)
    cavgz = pop!(cavgz_stack)
    factord = pop!(factor_stack_d)
    factor = pop!(factor_stack)
    red = pop!(re_stack_d)
    re = pop!(re_stack)
    vered = pop!(vere_stack_d)
    vere = pop!(vere_stack)
    for i_seq_node = i_nnode:-1:1
        ubd[i_seq_node] = ubd[i_seq_node] + (lossb[1] * (2 * ((ud[i_seq_node] + upd[i_seq_node]) + -(urefd[i_seq_node]))) + (2 * ((u[i_seq_node] + up[i_seq_node]) - uref[i_seq_node])) * lossbd[1])
        ub[i_seq_node] = ub[i_seq_node] + (2 * ((u[i_seq_node] + up[i_seq_node]) - uref[i_seq_node])) * lossb[1]
        upbd[i_seq_node] = upbd[i_seq_node] + (lossb[1] * (2 * ((ud[i_seq_node] + upd[i_seq_node]) + -(urefd[i_seq_node]))) + (2 * ((u[i_seq_node] + up[i_seq_node]) - uref[i_seq_node])) * lossbd[1])
        upb[i_seq_node] = upb[i_seq_node] + (2 * ((u[i_seq_node] + up[i_seq_node]) - uref[i_seq_node])) * lossb[1]
        urefbd[i_seq_node] = urefbd[i_seq_node] + -((lossb[1] * (2 * ((ud[i_seq_node] + upd[i_seq_node]) + -(urefd[i_seq_node]))) + (2 * ((u[i_seq_node] + up[i_seq_node]) - uref[i_seq_node])) * lossbd[1]))
        urefb[i_seq_node] = urefb[i_seq_node] + -((2 * ((u[i_seq_node] + up[i_seq_node]) - uref[i_seq_node])) * lossb[1])
    end
    for i_cell = i_ncell:-1:1
        aereskd = pop!(aeresk_stack_d)
        aeresk = pop!(aeresk_stack)
        aerexd = pop!(aerex_stack_d)
        aerex = pop!(aerex_stack)
        aereyd = pop!(aerey_stack_d)
        aerey = pop!(aerey_stack)
        aerezd = pop!(aerez_stack_d)
        aerez = pop!(aerez_stack)
        cavgxd = pop!(cavgx_stack_d)
        cavgx = pop!(cavgx_stack)
        cavgyd = pop!(cavgy_stack_d)
        cavgy = pop!(cavgy_stack)
        cavgzd = pop!(cavgz_stack_d)
        cavgz = pop!(cavgz_stack)
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
                auxres2bd = auxres2bd + res2bd[i_k_node]
                auxres2b = auxres2b + res2b[i_k_node]
                auxresbd = auxresbd + resbd[i_k_node]
                auxresb = auxresb + resb[i_k_node]
                factorbd = factorbd + (auxres2b * gammad[i_cell] + gamma[i_cell] * auxres2bd)
                factorb = factorb + gamma[i_cell] * auxres2b
                gammabd[i_cell] = gammabd[i_cell] + (auxres2b * factord + factor * auxres2bd)
                gammab[i_cell] = gammab[i_cell] + factor * auxres2b
                auxres2bd = 0.0
                auxres2b = 0.0
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
                dtbd = dtbd + ((0.3333333333333333 * (aeresk * factorb)) * (2dtd) + (2dt) * (0.3333333333333333 * (factorb * aereskd + aeresk * factorbd)))
                dtb = dtb + (2dt) * (0.3333333333333333 * (aeresk * factorb))
                aereskbd = aereskbd + (factorb * (0.3333333333333333 * ((2dt) * dtd)) + (dt ^ 2 / 3) * factorbd)
                aereskb = aereskb + (dt ^ 2 / 3) * factorb
                factorbd = 0.0
                factorb = 0.0
                aereskd = pop!(aeresk_stack_d)
                aeresk = pop!(aeresk_stack)
                aerexbd = aerexbd + (aereskb * skxd[i_k, i_cell] + skx[i_k, i_cell] * aereskbd)
                aerexb = aerexb + skx[i_k, i_cell] * aereskb
                skxbd[i_k, i_cell] = skxbd[i_k, i_cell] + (aereskb * aerexd + aerex * aereskbd)
                skxb[i_k, i_cell] = skxb[i_k, i_cell] + aerex * aereskb
                aereybd = aereybd + (aereskb * skyd[i_k, i_cell] + sky[i_k, i_cell] * aereskbd)
                aereyb = aereyb + sky[i_k, i_cell] * aereskb
                skybd[i_k, i_cell] = skybd[i_k, i_cell] + (aereskb * aereyd + aerey * aereskbd)
                skyb[i_k, i_cell] = skyb[i_k, i_cell] + aerey * aereskb
                aerezbd = aerezbd + (aereskb * skzd[i_k, i_cell] + skz[i_k, i_cell] * aereskbd)
                aerezb = aerezb + skz[i_k, i_cell] * aereskb
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
            aerezd = pop!(aerez_stack_d)
            aerez = pop!(aerez_stack)
            cavgzbd = cavgzbd + (aerezb * red + re * aerezbd)
            cavgzb = cavgzb + re * aerezb
            rebd = rebd + (aerezb * cavgzd + cavgz * aerezbd)
            reb = reb + cavgz * aerezb
            aerezbd = 0.0
            aerezb = 0.0
            aereyd = pop!(aerey_stack_d)
            aerey = pop!(aerey_stack)
            cavgybd = cavgybd + (aereyb * red + re * aereybd)
            cavgyb = cavgyb + re * aereyb
            rebd = rebd + (aereyb * cavgyd + cavgy * aereybd)
            reb = reb + cavgy * aereyb
            aereybd = 0.0
            aereyb = 0.0
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
            cavgzd = pop!(cavgz_stack_d)
            cavgz = pop!(cavgz_stack)
            cbd[3, i_node] = cbd[3, i_node] + 0.25cavgzbd
            cb[3, i_node] = cb[3, i_node] + 0.25cavgzb
            cavgyd = pop!(cavgy_stack_d)
            cavgy = pop!(cavgy_stack)
            cbd[2, i_node] = cbd[2, i_node] + 0.25cavgybd
            cb[2, i_node] = cb[2, i_node] + 0.25cavgyb
            cavgxd = pop!(cavgx_stack_d)
            cavgx = pop!(cavgx_stack)
            cbd[1, i_node] = cbd[1, i_node] + 0.25cavgxbd
            cb[1, i_node] = cb[1, i_node] + 0.25cavgxb
        end
        cavgzd = pop!(cavgz_stack_d)
        cavgz = pop!(cavgz_stack)
        cavgzbd = 0.0
        cavgzb = 0.0
        cavgyd = pop!(cavgy_stack_d)
        cavgy = pop!(cavgy_stack)
        cavgybd = 0.0
        cavgyb = 0.0
        cavgxd = pop!(cavgx_stack_d)
        cavgx = pop!(cavgx_stack)
        cavgxbd = 0.0
        cavgxb = 0.0
    end
    return (node_volb, node_volbd, dtb, dtbd, i_njacb, i_njacbd, mupb, mupbd, npernode_halfb, npernode_halfbd, resperiob, resperiobd, i_node_periob, i_node_periobd)
end

function ttgc(u, uref, i_cell_to_node, cell_vol, node_vol, skx, sky, skz, i_ncell, i_nnode, c, dt, beta, gamma, i_njac, res, res2, up, mup, npernode_half, resperio, i_node_perio, loss)
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
    for i_seq_node = 1:i_nnode
        loss[1] = loss[1] + ((u[i_seq_node] + up[i_seq_node]) - uref[i_seq_node]) ^ 2
    end
end
