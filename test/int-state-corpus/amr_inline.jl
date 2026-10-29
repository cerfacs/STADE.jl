# amr_inline(xl, xr, ncell, cap, p, x0, tol, out)
#
# Integral of tanh(p * (x - x0)) over a mesh refined adaptively until no cell
# violates the curvature tolerance. The mesh and the cell count change at run time.
#
# xl: left edge of each cell, length cap (first ncell[1] entries in use on entry)
# xr: right edge of each cell, length cap
# ncell: length-1 integer array; initial cell count on entry, refined count on exit
# cap: capacity of xl and xr
# p: steepness of the front
# x0: location of the front
# tol: refinement threshold on the per-cell curvature indicator
# out: length-1 output array; out[1] receives the integral
function amr_inline(xl, xr, ncell, cap, p, x0, tol, out)
    n = ncell[1]
    # a nonzero value forces the first refinement pass
    nsplit = 1
    # refine until a full pass splits no cell; the pass count depends on the data
    while nsplit > 0
        nsplit = 0
        nold = n
        # sequential: n and the appended cells are carried between iterations
        for c = 1:nold
            a = xl[c]
            b = xr[c]
            m = 0.5 * (a + b)
            fa = tanh(p * (a - x0))
            fm = tanh(p * (m - x0))
            fb = tanh(p * (b - x0))
            err = abs(fm - 0.5 * (fa + fb)) * (b - a)
            if err > tol
                if n < cap
                    xl[n + 1] = m
                    xr[n + 1] = b
                    xr[c] = m
                    n += 1
                    nsplit += 1
                end
            end
        end
    end
    ncell[1] = n
    # Simpson rule on every final cell; commutative accumulation
    s = 0.0
    for k = 1:n
        a = xl[k]
        b = xr[k]
        m = 0.5 * (a + b)
        s += (b - a) / 6.0 * (tanh(p * (a - x0)) + 4.0 * tanh(p * (m - x0)) + tanh(p * (b - x0)))
    end
    out[1] = s
    return nothing
end
