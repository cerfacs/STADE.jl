# fp_ambiguous.jl
#
# NEGATIVE. Two loops both read, write and carry the declared state, so a
# keyword naming only the state cannot say which one is the fixed point.
# Tapenade never meets this case: its directive sits on a loop header and
# binds by position. It is a cost of STADE's keyword, and the refusal must
# name both loops.
function fp_ambiguous(loss, u, f, a, i_n)
    d = 1.0
    while d > 1.0e-24
        d = 0.0
        for i_k = 1:i_n
            un = 0.5 * u[i_k] + 0.25 * f[i_k]
            d = d + (un - u[i_k]) * (un - u[i_k])
            u[i_k] = un
        end
    end
    e = 1.0
    while e > 1.0e-24
        e = 0.0
        for i_j = 1:i_n
            un2 = 0.5 * u[i_j] + 0.25 * a[i_j]
            e = e + (un2 - u[i_j]) * (un2 - u[i_j])
            u[i_j] = un2
        end
    end
    for i_x = 1:i_n
        loss[1] = loss[1] + u[i_x] ^ 2
    end
    return nothing
end
