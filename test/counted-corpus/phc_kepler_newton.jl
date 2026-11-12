# phc_kepler_newton
# Kepler's equation E - e sin(E) = M for each particle by Newton's method, written as a counted loop with a guard (at most maxit steps) instead of a while.
# LOOP i1 I K
function phc_kepler_newton(mean_anom, ecc, ea, tol, n, maxit)
    for i1 = 1:n
        e1 = mean_anom[i1] + ecc[i1] * sin(mean_anom[i1])
        st = 1.0
        for k1 = 1:maxit
            # Newton step on g(E) = E - e sin(E) - M, computed every pass and applied while the last applied step was above tol.
            # st is assigned outside the guard: a scalar assigned only inside it keeps the adjoint of this loop on the host.
            dlt = (e1 - ecc[i1] * sin(e1) - mean_anom[i1]) / (1.0 - ecc[i1] * cos(e1))
            if st > tol
                e1 = e1 - dlt
            end
            st = abs(dlt)
        end
        ea[i1] = e1
    end
    return nothing
end
