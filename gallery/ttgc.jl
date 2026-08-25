# ttgc(u, uref, i_cell_to_node, cell_vol, node_vol, skx, sky, skz, i_ncell,
#      i_nnode, c, dt, beta, gamma, i_njac, res, res2, up, mup, loss)
#
# One time step of the two-step Taylor-Galerkin C (TTGC) scheme for the
# 3D advection equation on tetrahedral cells, with beta and gamma defined
# per cell. Each step's mass-matrix solve uses i_njac Jacobi iterations
# instead of an exact linear solve. The final step adds the squared error
# between the updated field and a reference field into loss.
#
# u: field value at each node before this time step, length i_nnode
# uref: reference field value at each node, length i_nnode
# i_cell_to_node: node index of each cell's 4 local nodes, shape (4, i_ncell)
# cell_vol: volume of each cell, length i_ncell
# node_vol: control-volume size around each node, length i_nnode
# skx, sky, skz: x/y/z components of each cell-local node's weighted
#   outward normal, shape (4, i_ncell)
# i_ncell: number of cells
# i_nnode: number of nodes
# c: convection velocity at each node, shape (3, i_nnode)
# dt: time step size
# beta: TTGC scheme's beta coefficient, defined per cell, length i_ncell
# gamma: TTGC scheme's gamma coefficient, defined per cell, length i_ncell
# i_njac: number of Jacobi iterations used to solve each mass-matrix system
# res: scratch/output array for the first step's nodal residual, length i_nnode
# res2: scratch/output array for the second step's nodal residual, length i_nnode
# up: scratch array for the current step's field increment, length i_nnode
# mup: scratch array for the mass-matrix product used inside the Jacobi loop,
#   length i_nnode
# loss: length-1 output array; loss[1] receives the summed squared error
#   between the updated field and uref
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
    for i_node = 1:i_nnode
        loss[1] = loss[1] + ((u[i_node] + up[i_node]) - uref[i_node]) ^ 2
    end
    return nothing
end