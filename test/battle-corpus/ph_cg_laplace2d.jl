# ph_cg_laplace2d
# Matrix-free conjugate gradient for -Laplace(x) = b on an (nint+2)-square grid with zero boundary, until the squared residual falls below tol.
# LOOP i1 I K
# LOOP i2 I K
# LOOP i3 I K
# LOOP i4 I K
# LOOP i5 I K
# LOOP i6 I K
# WHILE rr H
function ph_cg_laplace2d(x, b, r, p, ap, tol, nint, its)
    # start: x = 0, r = b, p = r
    rr = 0.0
    for i1 = 1:nint * nint
        r1 = div(i1 - 1, nint) + 2
        c1 = mod(i1 - 1, nint) + 2
        x[r1, c1] = 0.0
        r[r1, c1] = b[r1, c1]
        p[r1, c1] = b[r1, c1]
        rr = rr + b[r1, c1] * b[r1, c1]
    end
    k = 0
    while rr > tol
        # ap = A p, 5-point stencil
        for i2 = 1:nint * nint
            r2 = div(i2 - 1, nint) + 2
            c2 = mod(i2 - 1, nint) + 2
            ap[r2, c2] = 4.0 * p[r2, c2] - p[r2 - 1, c2] - p[r2 + 1, c2] - p[r2, c2 - 1] - p[r2, c2 + 1]
        end
        pap = 0.0
        for i3 = 1:nint * nint
            r3 = div(i3 - 1, nint) + 2
            c3 = mod(i3 - 1, nint) + 2
            pap = pap + p[r3, c3] * ap[r3, c3]
        end
        alpha = rr / pap
        for i4 = 1:nint * nint
            r4 = div(i4 - 1, nint) + 2
            c4 = mod(i4 - 1, nint) + 2
            x[r4, c4] = x[r4, c4] + alpha * p[r4, c4]
            r[r4, c4] = r[r4, c4] - alpha * ap[r4, c4]
        end
        rrn = 0.0
        for i5 = 1:nint * nint
            r5 = div(i5 - 1, nint) + 2
            c5 = mod(i5 - 1, nint) + 2
            rrn = rrn + r[r5, c5] * r[r5, c5]
        end
        beta = rrn / rr
        for i6 = 1:nint * nint
            r6 = div(i6 - 1, nint) + 2
            c6 = mod(i6 - 1, nint) + 2
            p[r6, c6] = r[r6, c6] + beta * p[r6, c6]
        end
        rr = rrn
        k = k + 1
    end
    its[1] = k
    return nothing
end
