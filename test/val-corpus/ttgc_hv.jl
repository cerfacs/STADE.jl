function initstacks_ttgc_b(i_ncell, i_njac, i_nnode)
    vere_stack = Vector{Float64}(undef, ((((div(i_ncell - 1, 1) + 1) * (div(4 - 1, 1) + 1) + (div(i_ncell - 1, 1) + 1)) + (div(i_ncell - 1, 1) + 1) * (div(4 - 1, 1) + 1)) + (div(i_ncell - 1, 1) + 1)) + 1)
    mup_stack = Vector{Float64}(undef, (((div(i_njac - 1, 1) + 1) * (div(i_nnode - 1, 1) + 1) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1) * (div(4 - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_nnode - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1) * (div(4 - 1, 1) + 1))
    auxu_stack = Vector{Float64}(undef, ((((((((div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1) * (div(4 - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1) * (div(4 - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1)) + 1)
    up_stack = Vector{Float64}(undef, ((div(i_njac - 1, 1) + 1) * (div(i_nnode - 1, 1) + 1) + (div(i_nnode - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_nnode - 1, 1) + 1))
    return (vere_stack, mup_stack, auxu_stack, up_stack)
end

function ttgc_hv(u, ub, uref, urefb, i_cell_to_node, cell_vol, cell_volb, node_vol, node_volb, skx, skxb, sky, skyb, skz, skzb, i_ncell, i_nnode, c, cb, dt, dtb, beta, betab, gamma, gammab, i_njac, res, resb, res2, res2b, up, upb, mup, mupb, loss, lossb, ud, ubd, urefd, urefbd, cell_vold, cell_volbd, node_vold, node_volbd, skxd, skxbd, skyd, skybd, skzd, skzbd, cd, cbd, dtd, dtbd, betad, betabd, gammad, gammabd, resd, resbd, res2d, res2bd, upd, upbd, mupd, mupbd, lossd, lossbd, vere_stack, mup_stack, auxu_stack, up_stack)
    vere_stack_d = Vector{Float64}(undef, length(vere_stack))
    mup_stack_d = Vector{Float64}(undef, length(mup_stack))
    auxu_stack_d = Vector{Float64}(undef, length(auxu_stack))
    up_stack_d = Vector{Float64}(undef, length(up_stack))
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
    auxud = 0.0
    auxubd = 0.0
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
        cavgxd = 0.0
        cavgx = 0.0
        cavgyd = 0.0
        cavgy = 0.0
        cavgzd = 0.0
        cavgz = 0.0
        for i_loc = 1:4
            i_node = i_cell_to_node[i_loc, i_cell]
            cavgxd = cavgxd + 0.25 * cd[1, i_node]
            cavgx = cavgx + c[1, i_node] / 4
            cavgyd = cavgyd + 0.25 * cd[2, i_node]
            cavgy = cavgy + c[2, i_node] / 4
            cavgzd = cavgzd + 0.25 * cd[3, i_node]
            cavgz = cavgz + c[3, i_node] / 4
        end
        for i_loc = 1:4
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
                i_k_node = i_cell_to_node[i_k, i_cell]
                resd[i_k_node] = resd[i_k_node] + auxresd
                res[i_k_node] = res[i_k_node] + auxres
                res2d[i_k_node] = res2d[i_k_node] + auxres2d
                res2[i_k_node] = res2[i_k_node] + auxres2
            end
        end
    end
    for i_node = 1:i_nnode
        upd[i_node] = (1.0 / node_vol[i_node]) * resd[i_node] + -(res[i_node] / node_vol[i_node] ^ 2) * node_vold[i_node]
        up[i_node] = res[i_node] / node_vol[i_node]
    end
    for i_ = 1:i_njac
        for i_node = 1:i_nnode
            mup_stack_d[((i_ - 1) * (div(i_nnode - 1, 1) + 1) + (i_node - 1)) + 1] = mupd[i_node]
            mup_stack[((i_ - 1) * (div(i_nnode - 1, 1) + 1) + (i_node - 1)) + 1] = mup[i_node]
            mupd[i_node] = 0.0
            mup[i_node] = 0.0
        end
        for i_cell = 1:i_ncell
            auxu_stack_d[((i_ - 1) * (div(i_ncell - 1, 1) + 1) + (i_cell - 1)) + 1] = auxud
            auxu_stack[((i_ - 1) * (div(i_ncell - 1, 1) + 1) + (i_cell - 1)) + 1] = auxu
            auxud = 0.0
            auxu = 0.0
            for i_loc = 1:4
                i_lnode = i_cell_to_node[i_loc, i_cell]
                auxud = auxud + upd[i_lnode]
                auxu = auxu + up[i_lnode]
            end
            for i_loc = 1:4
                i_lnode = i_cell_to_node[i_loc, i_cell]
                mup_stack_d[(div(i_njac - 1, 1) + 1) * (div(i_nnode - 1, 1) + 1) + (((i_ - 1) * ((div(i_ncell - 1, 1) + 1) * (div(4 - 1, 1) + 1)) + (i_cell - 1) * (div(4 - 1, 1) + 1) + (i_loc - 1)) + 1)] = mupd[i_lnode]
                mup_stack[(div(i_njac - 1, 1) + 1) * (div(i_nnode - 1, 1) + 1) + (((i_ - 1) * ((div(i_ncell - 1, 1) + 1) * (div(4 - 1, 1) + 1)) + (i_cell - 1) * (div(4 - 1, 1) + 1) + (i_loc - 1)) + 1)] = mup[i_lnode]
                mupd[i_lnode] = mupd[i_lnode] + 0.05 * (cell_vol[i_cell] * (auxud + upd[i_lnode]) + (auxu + up[i_lnode]) * cell_vold[i_cell])
                mup[i_lnode] = mup[i_lnode] + ((auxu + up[i_lnode]) * cell_vol[i_cell]) / 20.0
            end
            auxu_stack_d[((div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1) * (div(4 - 1, 1) + 1)) + (((i_ - 1) * (div(i_ncell - 1, 1) + 1) + (i_cell - 1)) + 1)] = auxud
            auxu_stack[((div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1) * (div(4 - 1, 1) + 1)) + (((i_ - 1) * (div(i_ncell - 1, 1) + 1) + (i_cell - 1)) + 1)] = auxu
        end
        for i_node = 1:i_nnode
            up_stack_d[((i_ - 1) * (div(i_nnode - 1, 1) + 1) + (i_node - 1)) + 1] = upd[i_node]
            up_stack[((i_ - 1) * (div(i_nnode - 1, 1) + 1) + (i_node - 1)) + 1] = up[i_node]
            upd[i_node] = upd[i_node] + ((1.0 / node_vol[i_node]) * (resd[i_node] + -(mupd[i_node])) + -((res[i_node] - mup[i_node]) / node_vol[i_node] ^ 2) * node_vold[i_node])
            up[i_node] = up[i_node] + (res[i_node] - mup[i_node]) / node_vol[i_node]
        end
        auxu_stack_d[(((div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1) * (div(4 - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + ((i_ - 1) + 1)] = auxud
        auxu_stack[(((div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1) * (div(4 - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + ((i_ - 1) + 1)] = auxu
    end
    for i_cell = 1:i_ncell
        for i_loc = 1:4
            i_node = i_cell_to_node[i_loc, i_cell]
            vere_stack_d[((div(i_ncell - 1, 1) + 1) * (div(4 - 1, 1) + 1) + (div(i_ncell - 1, 1) + 1)) + (((i_cell - 1) * (div(4 - 1, 1) + 1) + (i_loc - 1)) + 1)] = vered
            vere_stack[((div(i_ncell - 1, 1) + 1) * (div(4 - 1, 1) + 1) + (div(i_ncell - 1, 1) + 1)) + (((i_cell - 1) * (div(4 - 1, 1) + 1) + (i_loc - 1)) + 1)] = vere
            vered = (-(1.0 / 3.0) * (c[1, i_node] * skx[i_loc, i_cell] + c[2, i_node] * sky[i_loc, i_cell])) * (ud[i_node] + upd[i_node]) + (-(1.0 / 3.0) * (u[i_node] + up[i_node])) * ((skx[i_loc, i_cell] * cd[1, i_node] + c[1, i_node] * skxd[i_loc, i_cell]) + (sky[i_loc, i_cell] * cd[2, i_node] + c[2, i_node] * skyd[i_loc, i_cell]))
            vere = -(1.0 / 3.0) * (u[i_node] + up[i_node]) * (c[1, i_node] * skx[i_loc, i_cell] + c[2, i_node] * sky[i_loc, i_cell])
            resid = vere * -(0.25dtd) + -(dt / 4) * vered
            resi = -(dt / 4) * vere
            for i_k = 1:4
                i_k_node = i_cell_to_node[i_k, i_cell]
                res2d[i_k_node] = res2d[i_k_node] + resid
                res2[i_k_node] = res2[i_k_node] + resi
            end
        end
        vere_stack_d[(((div(i_ncell - 1, 1) + 1) * (div(4 - 1, 1) + 1) + (div(i_ncell - 1, 1) + 1)) + (div(i_ncell - 1, 1) + 1) * (div(4 - 1, 1) + 1)) + ((i_cell - 1) + 1)] = vered
        vere_stack[(((div(i_ncell - 1, 1) + 1) * (div(4 - 1, 1) + 1) + (div(i_ncell - 1, 1) + 1)) + (div(i_ncell - 1, 1) + 1) * (div(4 - 1, 1) + 1)) + ((i_cell - 1) + 1)] = vere
    end
    for i_node = 1:i_nnode
        up_stack_d[(div(i_njac - 1, 1) + 1) * (div(i_nnode - 1, 1) + 1) + ((i_node - 1) + 1)] = upd[i_node]
        up_stack[(div(i_njac - 1, 1) + 1) * (div(i_nnode - 1, 1) + 1) + ((i_node - 1) + 1)] = up[i_node]
        upd[i_node] = (1.0 / node_vol[i_node]) * res2d[i_node] + -(res2[i_node] / node_vol[i_node] ^ 2) * node_vold[i_node]
        up[i_node] = res2[i_node] / node_vol[i_node]
    end
    for i_ = 1:i_njac
        for i_node = 1:i_nnode
            mup_stack_d[((div(i_njac - 1, 1) + 1) * (div(i_nnode - 1, 1) + 1) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1) * (div(4 - 1, 1) + 1)) + (((i_ - 1) * (div(i_nnode - 1, 1) + 1) + (i_node - 1)) + 1)] = mupd[i_node]
            mup_stack[((div(i_njac - 1, 1) + 1) * (div(i_nnode - 1, 1) + 1) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1) * (div(4 - 1, 1) + 1)) + (((i_ - 1) * (div(i_nnode - 1, 1) + 1) + (i_node - 1)) + 1)] = mup[i_node]
            mupd[i_node] = 0.0
            mup[i_node] = 0.0
        end
        for i_cell = 1:i_ncell
            auxu_stack_d[((((div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1) * (div(4 - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1)) + (((i_ - 1) * (div(i_ncell - 1, 1) + 1) + (i_cell - 1)) + 1)] = auxud
            auxu_stack[((((div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1) * (div(4 - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1)) + (((i_ - 1) * (div(i_ncell - 1, 1) + 1) + (i_cell - 1)) + 1)] = auxu
            auxud = 0.0
            auxu = 0.0
            for i_loc = 1:4
                i_lnode = i_cell_to_node[i_loc, i_cell]
                auxud = auxud + upd[i_lnode]
                auxu = auxu + up[i_lnode]
            end
            for i_loc = 1:4
                i_lnode = i_cell_to_node[i_loc, i_cell]
                mup_stack_d[(((div(i_njac - 1, 1) + 1) * (div(i_nnode - 1, 1) + 1) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1) * (div(4 - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_nnode - 1, 1) + 1)) + (((i_ - 1) * ((div(i_ncell - 1, 1) + 1) * (div(4 - 1, 1) + 1)) + (i_cell - 1) * (div(4 - 1, 1) + 1) + (i_loc - 1)) + 1)] = mupd[i_lnode]
                mup_stack[(((div(i_njac - 1, 1) + 1) * (div(i_nnode - 1, 1) + 1) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1) * (div(4 - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_nnode - 1, 1) + 1)) + (((i_ - 1) * ((div(i_ncell - 1, 1) + 1) * (div(4 - 1, 1) + 1)) + (i_cell - 1) * (div(4 - 1, 1) + 1) + (i_loc - 1)) + 1)] = mup[i_lnode]
                mupd[i_lnode] = mupd[i_lnode] + 0.05 * (cell_vol[i_cell] * (auxud + upd[i_lnode]) + (auxu + up[i_lnode]) * cell_vold[i_cell])
                mup[i_lnode] = mup[i_lnode] + ((auxu + up[i_lnode]) * cell_vol[i_cell]) / 20.0
            end
            auxu_stack_d[((((((div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1) * (div(4 - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1) * (div(4 - 1, 1) + 1)) + (((i_ - 1) * (div(i_ncell - 1, 1) + 1) + (i_cell - 1)) + 1)] = auxud
            auxu_stack[((((((div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1) * (div(4 - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1) * (div(4 - 1, 1) + 1)) + (((i_ - 1) * (div(i_ncell - 1, 1) + 1) + (i_cell - 1)) + 1)] = auxu
        end
        for i_node = 1:i_nnode
            up_stack_d[((div(i_njac - 1, 1) + 1) * (div(i_nnode - 1, 1) + 1) + (div(i_nnode - 1, 1) + 1)) + (((i_ - 1) * (div(i_nnode - 1, 1) + 1) + (i_node - 1)) + 1)] = upd[i_node]
            up_stack[((div(i_njac - 1, 1) + 1) * (div(i_nnode - 1, 1) + 1) + (div(i_nnode - 1, 1) + 1)) + (((i_ - 1) * (div(i_nnode - 1, 1) + 1) + (i_node - 1)) + 1)] = up[i_node]
            upd[i_node] = upd[i_node] + ((1.0 / node_vol[i_node]) * (res2d[i_node] + -(mupd[i_node])) + -((res2[i_node] - mup[i_node]) / node_vol[i_node] ^ 2) * node_vold[i_node])
            up[i_node] = up[i_node] + (res2[i_node] - mup[i_node]) / node_vol[i_node]
        end
        auxu_stack_d[(((((((div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1) * (div(4 - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1) * (div(4 - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + ((i_ - 1) + 1)] = auxud
        auxu_stack[(((((((div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1) * (div(4 - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1) * (div(4 - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + ((i_ - 1) + 1)] = auxu
    end
    for i_node2 = 1:i_nnode
        lossd[1] = lossd[1] + (2 * ((u[i_node2] + up[i_node2]) - uref[i_node2])) * ((ud[i_node2] + upd[i_node2]) + -(urefd[i_node2]))
        loss[1] = loss[1] + ((u[i_node2] + up[i_node2]) - uref[i_node2]) ^ 2
    end
    auxu_stack_d[((((((((div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1) * (div(4 - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1) * (div(4 - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1)) + 1] = auxud
    auxu_stack[((((((((div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1) * (div(4 - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1) * (div(4 - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1)) + 1] = auxu
    vere_stack_d[((((div(i_ncell - 1, 1) + 1) * (div(4 - 1, 1) + 1) + (div(i_ncell - 1, 1) + 1)) + (div(i_ncell - 1, 1) + 1) * (div(4 - 1, 1) + 1)) + (div(i_ncell - 1, 1) + 1)) + 1] = vered
    vere_stack[((((div(i_ncell - 1, 1) + 1) * (div(4 - 1, 1) + 1) + (div(i_ncell - 1, 1) + 1)) + (div(i_ncell - 1, 1) + 1) * (div(4 - 1, 1) + 1)) + (div(i_ncell - 1, 1) + 1)) + 1] = vere
    auxud = auxu_stack_d[((((((((div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1) * (div(4 - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1) * (div(4 - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1)) + 1]
    auxu = auxu_stack[((((((((div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1) * (div(4 - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1) * (div(4 - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1)) + 1]
    vered = vere_stack_d[((((div(i_ncell - 1, 1) + 1) * (div(4 - 1, 1) + 1) + (div(i_ncell - 1, 1) + 1)) + (div(i_ncell - 1, 1) + 1) * (div(4 - 1, 1) + 1)) + (div(i_ncell - 1, 1) + 1)) + 1]
    vere = vere_stack[((((div(i_ncell - 1, 1) + 1) * (div(4 - 1, 1) + 1) + (div(i_ncell - 1, 1) + 1)) + (div(i_ncell - 1, 1) + 1) * (div(4 - 1, 1) + 1)) + (div(i_ncell - 1, 1) + 1)) + 1]
    for i_node2 = i_nnode:-1:1
        ubd[i_node2] = ubd[i_node2] + (lossb[1] * (2 * ((ud[i_node2] + upd[i_node2]) + -(urefd[i_node2]))) + (2 * ((u[i_node2] + up[i_node2]) - uref[i_node2])) * lossbd[1])
        ub[i_node2] = ub[i_node2] + (2 * ((u[i_node2] + up[i_node2]) - uref[i_node2])) * lossb[1]
        upbd[i_node2] = upbd[i_node2] + (lossb[1] * (2 * ((ud[i_node2] + upd[i_node2]) + -(urefd[i_node2]))) + (2 * ((u[i_node2] + up[i_node2]) - uref[i_node2])) * lossbd[1])
        upb[i_node2] = upb[i_node2] + (2 * ((u[i_node2] + up[i_node2]) - uref[i_node2])) * lossb[1]
        urefbd[i_node2] = urefbd[i_node2] + -((lossb[1] * (2 * ((ud[i_node2] + upd[i_node2]) + -(urefd[i_node2]))) + (2 * ((u[i_node2] + up[i_node2]) - uref[i_node2])) * lossbd[1]))
        urefb[i_node2] = urefb[i_node2] + -((2 * ((u[i_node2] + up[i_node2]) - uref[i_node2])) * lossb[1])
    end
    for i_ = i_njac:-1:1
        auxud = auxu_stack_d[(((((((div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1) * (div(4 - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1) * (div(4 - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + ((i_ - 1) + 1)]
        auxu = auxu_stack[(((((((div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1) * (div(4 - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1) * (div(4 - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + ((i_ - 1) + 1)]
        for i_node = i_nnode:-1:1
            upd[i_node] = up_stack_d[((div(i_njac - 1, 1) + 1) * (div(i_nnode - 1, 1) + 1) + (div(i_nnode - 1, 1) + 1)) + (((i_ - 1) * (div(i_nnode - 1, 1) + 1) + (i_node - 1)) + 1)]
            up[i_node] = up_stack[((div(i_njac - 1, 1) + 1) * (div(i_nnode - 1, 1) + 1) + (div(i_nnode - 1, 1) + 1)) + (((i_ - 1) * (div(i_nnode - 1, 1) + 1) + (i_node - 1)) + 1)]
            res2bd[i_node] = res2bd[i_node] + (upb[i_node] * (-(1.0 / node_vol[i_node] ^ 2) * node_vold[i_node]) + (1.0 / node_vol[i_node]) * upbd[i_node])
            res2b[i_node] = res2b[i_node] + (1.0 / node_vol[i_node]) * upb[i_node]
            mupbd[i_node] = mupbd[i_node] + -((upb[i_node] * (-(1.0 / node_vol[i_node] ^ 2) * node_vold[i_node]) + (1.0 / node_vol[i_node]) * upbd[i_node]))
            mupb[i_node] = mupb[i_node] + -((1.0 / node_vol[i_node]) * upb[i_node])
            node_volbd[i_node] = node_volbd[i_node] + (upb[i_node] * -(((1.0 / node_vol[i_node] ^ 2) * (res2d[i_node] + -(mupd[i_node])) + -((res2[i_node] - mup[i_node]) / (node_vol[i_node] ^ 2) ^ 2) * ((2 * node_vol[i_node]) * node_vold[i_node]))) + -((res2[i_node] - mup[i_node]) / node_vol[i_node] ^ 2) * upbd[i_node])
            node_volb[i_node] = node_volb[i_node] + -((res2[i_node] - mup[i_node]) / node_vol[i_node] ^ 2) * upb[i_node]
        end
        for i_cell = i_ncell:-1:1
            auxud = auxu_stack_d[((((((div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1) * (div(4 - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1) * (div(4 - 1, 1) + 1)) + (((i_ - 1) * (div(i_ncell - 1, 1) + 1) + (i_cell - 1)) + 1)]
            auxu = auxu_stack[((((((div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1) * (div(4 - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1) * (div(4 - 1, 1) + 1)) + (((i_ - 1) * (div(i_ncell - 1, 1) + 1) + (i_cell - 1)) + 1)]
            for i_loc = 4:-1:1
                i_lnode = i_cell_to_node[i_loc, i_cell]
                mupd[i_lnode] = mup_stack_d[(((div(i_njac - 1, 1) + 1) * (div(i_nnode - 1, 1) + 1) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1) * (div(4 - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_nnode - 1, 1) + 1)) + (((i_ - 1) * ((div(i_ncell - 1, 1) + 1) * (div(4 - 1, 1) + 1)) + (i_cell - 1) * (div(4 - 1, 1) + 1) + (i_loc - 1)) + 1)]
                mup[i_lnode] = mup_stack[(((div(i_njac - 1, 1) + 1) * (div(i_nnode - 1, 1) + 1) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1) * (div(4 - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_nnode - 1, 1) + 1)) + (((i_ - 1) * ((div(i_ncell - 1, 1) + 1) * (div(4 - 1, 1) + 1)) + (i_cell - 1) * (div(4 - 1, 1) + 1) + (i_loc - 1)) + 1)]
                auxubd = auxubd + ((0.05 * mupb[i_lnode]) * cell_vold[i_cell] + cell_vol[i_cell] * (0.05 * mupbd[i_lnode]))
                auxub = auxub + cell_vol[i_cell] * (0.05 * mupb[i_lnode])
                upbd[i_lnode] = upbd[i_lnode] + ((0.05 * mupb[i_lnode]) * cell_vold[i_cell] + cell_vol[i_cell] * (0.05 * mupbd[i_lnode]))
                upb[i_lnode] = upb[i_lnode] + cell_vol[i_cell] * (0.05 * mupb[i_lnode])
                cell_volbd[i_cell] = cell_volbd[i_cell] + ((0.05 * mupb[i_lnode]) * (auxud + upd[i_lnode]) + (auxu + up[i_lnode]) * (0.05 * mupbd[i_lnode]))
                cell_volb[i_cell] = cell_volb[i_cell] + (auxu + up[i_lnode]) * (0.05 * mupb[i_lnode])
            end
            for i_loc = 1:4
                i_lnode = i_cell_to_node[i_loc, i_cell]
                auxud = auxud + upd[i_lnode]
                auxu = auxu + up[i_lnode]
                i_lnode = i_cell_to_node[i_loc, i_cell]
                upbd[i_lnode] = upbd[i_lnode] + auxubd
                upb[i_lnode] = upb[i_lnode] + auxub
            end
            auxud = auxu_stack_d[((((div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1) * (div(4 - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1)) + (((i_ - 1) * (div(i_ncell - 1, 1) + 1) + (i_cell - 1)) + 1)]
            auxu = auxu_stack[((((div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1) * (div(4 - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1)) + (((i_ - 1) * (div(i_ncell - 1, 1) + 1) + (i_cell - 1)) + 1)]
            auxubd = 0.0
            auxub = 0.0
        end
        for i_node = i_nnode:-1:1
            mupd[i_node] = mup_stack_d[((div(i_njac - 1, 1) + 1) * (div(i_nnode - 1, 1) + 1) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1) * (div(4 - 1, 1) + 1)) + (((i_ - 1) * (div(i_nnode - 1, 1) + 1) + (i_node - 1)) + 1)]
            mup[i_node] = mup_stack[((div(i_njac - 1, 1) + 1) * (div(i_nnode - 1, 1) + 1) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1) * (div(4 - 1, 1) + 1)) + (((i_ - 1) * (div(i_nnode - 1, 1) + 1) + (i_node - 1)) + 1)]
            mupbd[i_node] = 0.0
            mupb[i_node] = 0.0
        end
    end
    for i_node = i_nnode:-1:1
        upd[i_node] = up_stack_d[(div(i_njac - 1, 1) + 1) * (div(i_nnode - 1, 1) + 1) + ((i_node - 1) + 1)]
        up[i_node] = up_stack[(div(i_njac - 1, 1) + 1) * (div(i_nnode - 1, 1) + 1) + ((i_node - 1) + 1)]
        res2bd[i_node] = res2bd[i_node] + (upb[i_node] * (-(1.0 / node_vol[i_node] ^ 2) * node_vold[i_node]) + (1.0 / node_vol[i_node]) * upbd[i_node])
        res2b[i_node] = res2b[i_node] + (1.0 / node_vol[i_node]) * upb[i_node]
        node_volbd[i_node] = node_volbd[i_node] + (upb[i_node] * -(((1.0 / node_vol[i_node] ^ 2) * res2d[i_node] + -(res2[i_node] / (node_vol[i_node] ^ 2) ^ 2) * ((2 * node_vol[i_node]) * node_vold[i_node]))) + -(res2[i_node] / node_vol[i_node] ^ 2) * upbd[i_node])
        node_volb[i_node] = node_volb[i_node] + -(res2[i_node] / node_vol[i_node] ^ 2) * upb[i_node]
        upbd[i_node] = 0.0
        upb[i_node] = 0.0
    end
    for i_cell = i_ncell:-1:1
        vered = vere_stack_d[(((div(i_ncell - 1, 1) + 1) * (div(4 - 1, 1) + 1) + (div(i_ncell - 1, 1) + 1)) + (div(i_ncell - 1, 1) + 1) * (div(4 - 1, 1) + 1)) + ((i_cell - 1) + 1)]
        vere = vere_stack[(((div(i_ncell - 1, 1) + 1) * (div(4 - 1, 1) + 1) + (div(i_ncell - 1, 1) + 1)) + (div(i_ncell - 1, 1) + 1) * (div(4 - 1, 1) + 1)) + ((i_cell - 1) + 1)]
        for i_loc = 4:-1:1
            i_node = i_cell_to_node[i_loc, i_cell]
            for i_k = 4:-1:1
                i_k_node = i_cell_to_node[i_k, i_cell]
                resibd = resibd + res2bd[i_k_node]
                resib = resib + res2b[i_k_node]
            end
            dtbd = dtbd + 0.25 * -((resib * vered + vere * resibd))
            dtb = dtb + 0.25 * -(vere * resib)
            verebd = verebd + (resib * -(0.25dtd) + -(dt / 4) * resibd)
            vereb = vereb + -(dt / 4) * resib
            resibd = 0.0
            resib = 0.0
            vered = vere_stack_d[((div(i_ncell - 1, 1) + 1) * (div(4 - 1, 1) + 1) + (div(i_ncell - 1, 1) + 1)) + (((i_cell - 1) * (div(4 - 1, 1) + 1) + (i_loc - 1)) + 1)]
            vere = vere_stack[((div(i_ncell - 1, 1) + 1) * (div(4 - 1, 1) + 1) + (div(i_ncell - 1, 1) + 1)) + (((i_cell - 1) * (div(4 - 1, 1) + 1) + (i_loc - 1)) + 1)]
            ubd[i_node] = ubd[i_node] + (vereb * (-(1.0 / 3.0) * ((skx[i_loc, i_cell] * cd[1, i_node] + c[1, i_node] * skxd[i_loc, i_cell]) + (sky[i_loc, i_cell] * cd[2, i_node] + c[2, i_node] * skyd[i_loc, i_cell]))) + (-(1.0 / 3.0) * (c[1, i_node] * skx[i_loc, i_cell] + c[2, i_node] * sky[i_loc, i_cell])) * verebd)
            ub[i_node] = ub[i_node] + (-(1.0 / 3.0) * (c[1, i_node] * skx[i_loc, i_cell] + c[2, i_node] * sky[i_loc, i_cell])) * vereb
            upbd[i_node] = upbd[i_node] + (vereb * (-(1.0 / 3.0) * ((skx[i_loc, i_cell] * cd[1, i_node] + c[1, i_node] * skxd[i_loc, i_cell]) + (sky[i_loc, i_cell] * cd[2, i_node] + c[2, i_node] * skyd[i_loc, i_cell]))) + (-(1.0 / 3.0) * (c[1, i_node] * skx[i_loc, i_cell] + c[2, i_node] * sky[i_loc, i_cell])) * verebd)
            upb[i_node] = upb[i_node] + (-(1.0 / 3.0) * (c[1, i_node] * skx[i_loc, i_cell] + c[2, i_node] * sky[i_loc, i_cell])) * vereb
            cbd[1, i_node] = cbd[1, i_node] + (((-(1.0 / 3.0) * (u[i_node] + up[i_node])) * vereb) * skxd[i_loc, i_cell] + skx[i_loc, i_cell] * (vereb * (-(1.0 / 3.0) * (ud[i_node] + upd[i_node])) + (-(1.0 / 3.0) * (u[i_node] + up[i_node])) * verebd))
            cb[1, i_node] = cb[1, i_node] + skx[i_loc, i_cell] * ((-(1.0 / 3.0) * (u[i_node] + up[i_node])) * vereb)
            skxbd[i_loc, i_cell] = skxbd[i_loc, i_cell] + (((-(1.0 / 3.0) * (u[i_node] + up[i_node])) * vereb) * cd[1, i_node] + c[1, i_node] * (vereb * (-(1.0 / 3.0) * (ud[i_node] + upd[i_node])) + (-(1.0 / 3.0) * (u[i_node] + up[i_node])) * verebd))
            skxb[i_loc, i_cell] = skxb[i_loc, i_cell] + c[1, i_node] * ((-(1.0 / 3.0) * (u[i_node] + up[i_node])) * vereb)
            cbd[2, i_node] = cbd[2, i_node] + (((-(1.0 / 3.0) * (u[i_node] + up[i_node])) * vereb) * skyd[i_loc, i_cell] + sky[i_loc, i_cell] * (vereb * (-(1.0 / 3.0) * (ud[i_node] + upd[i_node])) + (-(1.0 / 3.0) * (u[i_node] + up[i_node])) * verebd))
            cb[2, i_node] = cb[2, i_node] + sky[i_loc, i_cell] * ((-(1.0 / 3.0) * (u[i_node] + up[i_node])) * vereb)
            skybd[i_loc, i_cell] = skybd[i_loc, i_cell] + (((-(1.0 / 3.0) * (u[i_node] + up[i_node])) * vereb) * cd[2, i_node] + c[2, i_node] * (vereb * (-(1.0 / 3.0) * (ud[i_node] + upd[i_node])) + (-(1.0 / 3.0) * (u[i_node] + up[i_node])) * verebd))
            skyb[i_loc, i_cell] = skyb[i_loc, i_cell] + c[2, i_node] * ((-(1.0 / 3.0) * (u[i_node] + up[i_node])) * vereb)
            verebd = 0.0
            vereb = 0.0
        end
    end
    for i_ = i_njac:-1:1
        auxud = auxu_stack_d[(((div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1) * (div(4 - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + ((i_ - 1) + 1)]
        auxu = auxu_stack[(((div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1) * (div(4 - 1, 1) + 1)) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1)) + ((i_ - 1) + 1)]
        for i_node = i_nnode:-1:1
            upd[i_node] = up_stack_d[((i_ - 1) * (div(i_nnode - 1, 1) + 1) + (i_node - 1)) + 1]
            up[i_node] = up_stack[((i_ - 1) * (div(i_nnode - 1, 1) + 1) + (i_node - 1)) + 1]
            resbd[i_node] = resbd[i_node] + (upb[i_node] * (-(1.0 / node_vol[i_node] ^ 2) * node_vold[i_node]) + (1.0 / node_vol[i_node]) * upbd[i_node])
            resb[i_node] = resb[i_node] + (1.0 / node_vol[i_node]) * upb[i_node]
            mupbd[i_node] = mupbd[i_node] + -((upb[i_node] * (-(1.0 / node_vol[i_node] ^ 2) * node_vold[i_node]) + (1.0 / node_vol[i_node]) * upbd[i_node]))
            mupb[i_node] = mupb[i_node] + -((1.0 / node_vol[i_node]) * upb[i_node])
            node_volbd[i_node] = node_volbd[i_node] + (upb[i_node] * -(((1.0 / node_vol[i_node] ^ 2) * (resd[i_node] + -(mupd[i_node])) + -((res[i_node] - mup[i_node]) / (node_vol[i_node] ^ 2) ^ 2) * ((2 * node_vol[i_node]) * node_vold[i_node]))) + -((res[i_node] - mup[i_node]) / node_vol[i_node] ^ 2) * upbd[i_node])
            node_volb[i_node] = node_volb[i_node] + -((res[i_node] - mup[i_node]) / node_vol[i_node] ^ 2) * upb[i_node]
        end
        for i_cell = i_ncell:-1:1
            auxud = auxu_stack_d[((div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1) * (div(4 - 1, 1) + 1)) + (((i_ - 1) * (div(i_ncell - 1, 1) + 1) + (i_cell - 1)) + 1)]
            auxu = auxu_stack[((div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1) + (div(i_njac - 1, 1) + 1) * (div(i_ncell - 1, 1) + 1) * (div(4 - 1, 1) + 1)) + (((i_ - 1) * (div(i_ncell - 1, 1) + 1) + (i_cell - 1)) + 1)]
            for i_loc = 4:-1:1
                i_lnode = i_cell_to_node[i_loc, i_cell]
                mupd[i_lnode] = mup_stack_d[(div(i_njac - 1, 1) + 1) * (div(i_nnode - 1, 1) + 1) + (((i_ - 1) * ((div(i_ncell - 1, 1) + 1) * (div(4 - 1, 1) + 1)) + (i_cell - 1) * (div(4 - 1, 1) + 1) + (i_loc - 1)) + 1)]
                mup[i_lnode] = mup_stack[(div(i_njac - 1, 1) + 1) * (div(i_nnode - 1, 1) + 1) + (((i_ - 1) * ((div(i_ncell - 1, 1) + 1) * (div(4 - 1, 1) + 1)) + (i_cell - 1) * (div(4 - 1, 1) + 1) + (i_loc - 1)) + 1)]
                auxubd = auxubd + ((0.05 * mupb[i_lnode]) * cell_vold[i_cell] + cell_vol[i_cell] * (0.05 * mupbd[i_lnode]))
                auxub = auxub + cell_vol[i_cell] * (0.05 * mupb[i_lnode])
                upbd[i_lnode] = upbd[i_lnode] + ((0.05 * mupb[i_lnode]) * cell_vold[i_cell] + cell_vol[i_cell] * (0.05 * mupbd[i_lnode]))
                upb[i_lnode] = upb[i_lnode] + cell_vol[i_cell] * (0.05 * mupb[i_lnode])
                cell_volbd[i_cell] = cell_volbd[i_cell] + ((0.05 * mupb[i_lnode]) * (auxud + upd[i_lnode]) + (auxu + up[i_lnode]) * (0.05 * mupbd[i_lnode]))
                cell_volb[i_cell] = cell_volb[i_cell] + (auxu + up[i_lnode]) * (0.05 * mupb[i_lnode])
            end
            for i_loc = 1:4
                i_lnode = i_cell_to_node[i_loc, i_cell]
                auxud = auxud + upd[i_lnode]
                auxu = auxu + up[i_lnode]
                i_lnode = i_cell_to_node[i_loc, i_cell]
                upbd[i_lnode] = upbd[i_lnode] + auxubd
                upb[i_lnode] = upb[i_lnode] + auxub
            end
            auxud = auxu_stack_d[((i_ - 1) * (div(i_ncell - 1, 1) + 1) + (i_cell - 1)) + 1]
            auxu = auxu_stack[((i_ - 1) * (div(i_ncell - 1, 1) + 1) + (i_cell - 1)) + 1]
            auxubd = 0.0
            auxub = 0.0
        end
        for i_node = i_nnode:-1:1
            mupd[i_node] = mup_stack_d[((i_ - 1) * (div(i_nnode - 1, 1) + 1) + (i_node - 1)) + 1]
            mup[i_node] = mup_stack[((i_ - 1) * (div(i_nnode - 1, 1) + 1) + (i_node - 1)) + 1]
            mupbd[i_node] = 0.0
            mupb[i_node] = 0.0
        end
    end
    for i_node = i_nnode:-1:1
        resbd[i_node] = resbd[i_node] + (upb[i_node] * (-(1.0 / node_vol[i_node] ^ 2) * node_vold[i_node]) + (1.0 / node_vol[i_node]) * upbd[i_node])
        resb[i_node] = resb[i_node] + (1.0 / node_vol[i_node]) * upb[i_node]
        node_volbd[i_node] = node_volbd[i_node] + (upb[i_node] * -(((1.0 / node_vol[i_node] ^ 2) * resd[i_node] + -(res[i_node] / (node_vol[i_node] ^ 2) ^ 2) * ((2 * node_vol[i_node]) * node_vold[i_node]))) + -(res[i_node] / node_vol[i_node] ^ 2) * upbd[i_node])
        node_volb[i_node] = node_volb[i_node] + -(res[i_node] / node_vol[i_node] ^ 2) * upb[i_node]
        upbd[i_node] = 0.0
        upb[i_node] = 0.0
    end
    for i_cell = 1:i_ncell
        cavgxd = 0.0
        cavgx = 0.0
        cavgyd = 0.0
        cavgy = 0.0
        cavgzd = 0.0
        cavgz = 0.0
        for i_loc = 1:4
            i_node = i_cell_to_node[i_loc, i_cell]
            cavgxd = cavgxd + 0.25 * cd[1, i_node]
            cavgx = cavgx + c[1, i_node] / 4
            cavgyd = cavgyd + 0.25 * cd[2, i_node]
            cavgy = cavgy + c[2, i_node] / 4
            cavgzd = cavgzd + 0.25 * cd[3, i_node]
            cavgz = cavgz + c[3, i_node] / 4
        end
        for i_loc = 1:4
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
                i_k_node = i_cell_to_node[i_k, i_cell]
            end
        end
        for i_loc = 1:4
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
                i_k_node = i_cell_to_node[i_k, i_cell]
            end
            i_node = i_cell_to_node[i_loc, i_cell]
            for i_k = 1:4
                aereskd = ((skx[i_k, i_cell] * aerexd + aerex * skxd[i_k, i_cell]) + (sky[i_k, i_cell] * aereyd + aerey * skyd[i_k, i_cell])) + (skz[i_k, i_cell] * aerezd + aerez * skzd[i_k, i_cell])
                aeresk = aerex * skx[i_k, i_cell] + aerey * sky[i_k, i_cell] + aerez * skz[i_k, i_cell]
                factord = aeresk * (0.3333333333333333 * ((2dt) * dtd)) + (dt ^ 2 / 3) * aereskd
                factor = (dt ^ 2 / 3) * aeresk
                auxresd = resid + (beta[i_cell] * factord + factor * betad[i_cell])
                auxres = resi + factor * beta[i_cell]
                auxres2d = gamma[i_cell] * factord + factor * gammad[i_cell]
                auxres2 = factor * gamma[i_cell]
                i_k_node = i_cell_to_node[i_k, i_cell]
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
                dtbd = dtbd + ((0.3333333333333333 * (aeresk * factorb)) * (2dtd) + (2dt) * (0.3333333333333333 * (factorb * aereskd + aeresk * factorbd)))
                dtb = dtb + (2dt) * (0.3333333333333333 * (aeresk * factorb))
                aereskbd = aereskbd + (factorb * (0.3333333333333333 * ((2dt) * dtd)) + (dt ^ 2 / 3) * factorbd)
                aereskb = aereskb + (dt ^ 2 / 3) * factorb
                factorbd = 0.0
                factorb = 0.0
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
            cavgzbd = cavgzbd + (aerezb * red + re * aerezbd)
            cavgzb = cavgzb + re * aerezb
            rebd = rebd + (aerezb * cavgzd + cavgz * aerezbd)
            reb = reb + cavgz * aerezb
            aerezbd = 0.0
            aerezb = 0.0
            cavgybd = cavgybd + (aereyb * red + re * aereybd)
            cavgyb = cavgyb + re * aereyb
            rebd = rebd + (aereyb * cavgyd + cavgy * aereybd)
            reb = reb + cavgy * aereyb
            aereybd = 0.0
            aereyb = 0.0
            cavgxbd = cavgxbd + (aerexb * red + re * aerexbd)
            cavgxb = cavgxb + re * aerexb
            rebd = rebd + (aerexb * cavgxd + cavgx * aerexbd)
            reb = reb + cavgx * aerexb
            aerexbd = 0.0
            aerexb = 0.0
            verebd = verebd + (reb * (-(1.0 / cell_vol[i_cell] ^ 2) * cell_vold[i_cell]) + (1.0 / cell_vol[i_cell]) * rebd)
            vereb = vereb + (1.0 / cell_vol[i_cell]) * reb
            cell_volbd[i_cell] = cell_volbd[i_cell] + (reb * -(((1.0 / cell_vol[i_cell] ^ 2) * vered + -(vere / (cell_vol[i_cell] ^ 2) ^ 2) * ((2 * cell_vol[i_cell]) * cell_vold[i_cell]))) + -(vere / cell_vol[i_cell] ^ 2) * rebd)
            cell_volb[i_cell] = cell_volb[i_cell] + -(vere / cell_vol[i_cell] ^ 2) * reb
            rebd = 0.0
            reb = 0.0
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
        for i_loc = 1:4
            i_node = i_cell_to_node[i_loc, i_cell]
            cavgxd = cavgxd + 0.25 * cd[1, i_node]
            cavgx = cavgx + c[1, i_node] / 4
            cavgyd = cavgyd + 0.25 * cd[2, i_node]
            cavgy = cavgy + c[2, i_node] / 4
            cavgzd = cavgzd + 0.25 * cd[3, i_node]
            cavgz = cavgz + c[3, i_node] / 4
            i_node = i_cell_to_node[i_loc, i_cell]
            cbd[3, i_node] = cbd[3, i_node] + 0.25cavgzbd
            cb[3, i_node] = cb[3, i_node] + 0.25cavgzb
            cbd[2, i_node] = cbd[2, i_node] + 0.25cavgybd
            cb[2, i_node] = cb[2, i_node] + 0.25cavgyb
            cbd[1, i_node] = cbd[1, i_node] + 0.25cavgxbd
            cb[1, i_node] = cb[1, i_node] + 0.25cavgxb
        end
        cavgzbd = 0.0
        cavgzb = 0.0
        cavgybd = 0.0
        cavgyb = 0.0
        cavgxbd = 0.0
        cavgxb = 0.0
    end
    return (dtb, dtbd)
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
