# ph_saha_bisect
# Ionisation fraction x in (0,1) of each cell from x^2 / (1 - x) = s[i]: bisection on a per-cell bracket of width w[i] (so cells need different numbers of steps), then two Newton steps. One step gives the first derivative. The second makes the second derivative (the HVP) right.
# LOOP i1 I K
# WHILE wid T
function ph_saha_bisect(s, w, xion, tol, n)
    for i1 = 1:n
        lo = 0.0
        hi = w[i1]
        wid = hi - lo
        while wid > tol
            mid = 0.5 * (lo + hi)
            # g(mid) = mid^2 - s (1 - mid)
            if mid * mid - s[i1] * (1.0 - mid) > 0.0
                hi = mid
            else
                lo = mid
            end
            wid = hi - lo
        end
        x1 = 0.5 * (lo + hi)
        # Newton polish: bisection alone has a zero derivative with respect to s. One step fixes the first derivative, two the second
        x1 = x1 - (x1 * x1 - s[i1] * (1.0 - x1)) / (2.0 * x1 + s[i1])
        x1 = x1 - (x1 * x1 - s[i1] * (1.0 - x1)) / (2.0 * x1 + s[i1])
        xion[i1] = x1
    end
    return nothing
end
