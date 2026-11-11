# ph_colebrook
# Colebrook friction factor for each pipe by fixed-point iteration on x = 1/sqrt(f), until x changes by less than tol.
# LOOP i1 I K
# WHILE dx T
function ph_colebrook(re, rough, fric, tol, n)
    for i1 = 1:n
        x1 = 8.0
        dx = 1.0
        while dx > tol
            # x = -2 log10( rough/3.7 + 2.51 x / Re )
            xn = -2.0 * log10(rough[i1] / 3.7 + 2.51 * x1 / re[i1])
            dx = abs(xn - x1)
            x1 = xn
        end
        fric[i1] = 1.0 / (x1 * x1)
    end
    return nothing
end
