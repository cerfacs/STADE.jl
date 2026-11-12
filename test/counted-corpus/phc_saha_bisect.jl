# phc_saha_bisect
# Ionisation fraction from x^2 / (1 - x) = s[i]: bisection as a counted loop with a guard (at most maxit steps), then two Newton steps.
# LOOP i1 I K
function phc_saha_bisect(s, w, xion, tol, n, maxit)
    for i1 = 1:n
        lo = 0.0
        hi = w[i1]
        for k1 = 1:maxit
            # mid and the sign of g are computed every pass; the bracket moves only while it is wider than tol
            wid = hi - lo
            mid = 0.5 * (lo + hi)
            gm = mid * mid - s[i1] * (1.0 - mid)
            if wid > tol
                if gm > 0.0
                    hi = mid
                else
                    lo = mid
                end
            end
        end
        x1 = 0.5 * (lo + hi)
        # Newton polish: bisection alone has a zero derivative with respect to s. One step fixes the first derivative, two the second
        x1 = x1 - (x1 * x1 - s[i1] * (1.0 - x1)) / (2.0 * x1 + s[i1])
        x1 = x1 - (x1 * x1 - s[i1] * (1.0 - x1)) / (2.0 * x1 + s[i1])
        xion[i1] = x1
    end
    return nothing
end
