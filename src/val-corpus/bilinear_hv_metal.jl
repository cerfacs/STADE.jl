import Pkg
haskey(Pkg.project().dependencies, "Metal") || Pkg.add("Metal")
using Metal
Metal.allowscalar(false)

function initstacks_bilinear_b_metal()
    return nothing
end

function bilinear_hv_metal(loss, lossb, x, xb, a, ab, y, yb, i_m, i_n, lossd, lossbd, xd, xbd, ad, abd, yd, ybd)
    for i_seq_i = 1:i_m
        for i_seq_j = 1:i_n
            lossd[1] = lossd[1] + (((a[i_seq_i, i_seq_j] * y[i_seq_j]) * xd[i_seq_i] + (x[i_seq_i] * y[i_seq_j]) * ad[i_seq_i, i_seq_j]) + (x[i_seq_i] * a[i_seq_i, i_seq_j]) * yd[i_seq_j])
            loss[1] = loss[1] + x[i_seq_i] * a[i_seq_i, i_seq_j] * y[i_seq_j]
        end
    end
    for i_seq_i = i_m:-1:1
        for i_seq_j = i_n:-1:1
            xbd[i_seq_i] = xbd[i_seq_i] + (lossb[1] * (y[i_seq_j] * ad[i_seq_i, i_seq_j] + a[i_seq_i, i_seq_j] * yd[i_seq_j]) + (a[i_seq_i, i_seq_j] * y[i_seq_j]) * lossbd[1])
            xb[i_seq_i] = xb[i_seq_i] + (a[i_seq_i, i_seq_j] * y[i_seq_j]) * lossb[1]
            abd[i_seq_i, i_seq_j] = abd[i_seq_i, i_seq_j] + (lossb[1] * (y[i_seq_j] * xd[i_seq_i] + x[i_seq_i] * yd[i_seq_j]) + (x[i_seq_i] * y[i_seq_j]) * lossbd[1])
            ab[i_seq_i, i_seq_j] = ab[i_seq_i, i_seq_j] + (x[i_seq_i] * y[i_seq_j]) * lossb[1]
            ybd[i_seq_j] = ybd[i_seq_j] + (lossb[1] * (a[i_seq_i, i_seq_j] * xd[i_seq_i] + x[i_seq_i] * ad[i_seq_i, i_seq_j]) + (x[i_seq_i] * a[i_seq_i, i_seq_j]) * lossbd[1])
            yb[i_seq_j] = yb[i_seq_j] + (x[i_seq_i] * a[i_seq_i, i_seq_j]) * lossb[1]
        end
    end
    return nothing
end

function bilinear_metal(loss, x, a, y, i_m, i_n)
    for i_seq_i = 1:i_m
        for i_seq_j = 1:i_n
            loss[1] = loss[1] + x[i_seq_i] * a[i_seq_i, i_seq_j] * y[i_seq_j]
        end
    end
    return nothing
end
