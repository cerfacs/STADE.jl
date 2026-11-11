# ph_kepler_newton
# Kepler's equation E - e sin(E) = M, solved for each particle by Newton's method until the step is below tol.
# LOOP i1 I K
# WHILE st T
function ph_kepler_newton(mean_anom, ecc, ea, tol, n)
    for i1 = 1:n
        e1 = mean_anom[i1] + ecc[i1] * sin(mean_anom[i1])
        st = 1.0
        while st > tol
            # Newton step on g(E) = E - e sin(E) - M
            dlt = (e1 - ecc[i1] * sin(e1) - mean_anom[i1]) / (1.0 - ecc[i1] * cos(e1))
            e1 = e1 - dlt
            st = abs(dlt)
        end
        ea[i1] = e1
    end
    return nothing
end
