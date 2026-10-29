# kepler(M, e, tol, out)
#
# Eccentric anomaly E solving E - e * sin(E) = M, by Newton iteration run until
# the step falls below tol. Only float temporaries inside the while loop. Control case.
#
# M: mean anomaly
# e: orbital eccentricity
# tol: convergence threshold on the Newton step
# out: length-1 output array; out[1] receives E
function kepler(M, e, tol, out)
    E = M
    # a large first step so the loop is entered
    dE = 1.0
    while abs(dE) > tol
        dE = (E - e * sin(E) - M) / (1.0 - e * cos(E))
        E = E - dE
    end
    out[1] = E
    return nothing
end
