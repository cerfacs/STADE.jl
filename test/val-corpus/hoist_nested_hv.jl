function initstacks_hoist_nested_b()
    return nothing
end

function hoist_nested_hv(y, yb, x, xb, i_n, i_nb, yd, ybd, xd, xbd)
    for i_r = 1:i_n
        i_b = mod(i_r - 1, i_nb) + 1
        for i_m = 1:2
            for i_e = 1:mod(i_r, 3) + 1
                yd[i_b] = yd[i_b] + xd[(i_r - 1) * 8 + (i_m - 1) * 4 + i_e]
                y[i_b] = y[i_b] + x[(i_r - 1) * 8 + (i_m - 1) * 4 + i_e]
            end
        end
    end
    for i_r = i_n:-1:1
        i_b = mod(i_r - 1, i_nb) + 1
        for i_m = 2:-1:1
            for i_e = mod(i_r, 3) + 1:-1:1
                xbd[(i_r - 1) * 8 + (i_m - 1) * 4 + i_e] = xbd[(i_r - 1) * 8 + (i_m - 1) * 4 + i_e] + ybd[i_b]
                xb[(i_r - 1) * 8 + (i_m - 1) * 4 + i_e] = xb[(i_r - 1) * 8 + (i_m - 1) * 4 + i_e] + yb[i_b]
            end
        end
    end
    return nothing
end

function hoist_nested(y, x, i_n, i_nb)
    for i_r = 1:i_n
        i_b = mod(i_r - 1, i_nb) + 1
        for i_m = 1:2
            for i_e = 1:mod(i_r, 3) + 1
                y[i_b] = y[i_b] + x[(i_r - 1) * 8 + (i_m - 1) * 4 + i_e]
            end
        end
    end
end
