function initstacks_remat_shapes_b(i_n, i_nc)
    wv_stack = Vector{Float64}(undef, ((((((((((((((((((((max(0, div(i_n - 1, 1) + 1) * max(0, div(i_nc - 1, 1) + 1) + max(0, div(i_n - 1, 1) + 1)) + max(0, div(i_n - 1, 1) + 1) * max(0, div(i_nc - 1, 1) + 1)) + max(0, div(i_n - 1, 1) + 1) * max(0, div(i_nc - 1, 1) + 1)) + max(0, div(i_n - 1, 1) + 1)) + max(0, div(i_n - 1, 1) + 1) * max(0, div(i_nc - 1, 1) + 1)) + max(0, div(i_n - 1, 1) + 1)) + max(0, div(i_n - 1, 1) + 1) * max(0, div(i_nc - 1, 1) + 1)) + max(0, div(i_n - 1, 1) + 1) * max(0, div(i_nc - 1, 1) + 1) * max(0, div(2 - 1, 1) + 1)) + max(0, div(i_n - 1, 1) + 1) * max(0, div(i_nc - 1, 1) + 1)) + max(0, div(i_n - 1, 1) + 1) * max(0, div(i_nc - 1, 1) + 1)) + max(0, div(i_n - 1, 1) + 1) * max(0, div(i_nc - 1, 1) + 1)) + max(0, div(i_n - 1, 1) + 1) * max(0, div(i_nc - 1, 1) + 1)) + max(0, div(i_n - 1, 1) + 1) * max(0, div(i_nc - 1, 1) + 1)) + max(0, div(i_n - 1, 1) + 1) * max(0, div(i_nc - 1, 1) + 1)) + max(0, div(i_n - 1, 1) + 1)) + max(0, div(i_n - 1, 1) + 1) * max(0, div(i_nc - 1, 1) + 1)) + max(0, div(i_n - 1, 1) + 1)) + max(0, div(i_n - 1, 1) + 1)) + max(0, div(i_n - 1, 1) + 1)) + max(0, div(i_n - 1, 1) + 1)) + max(0, div(i_n - 1, 1) + 1))
    branch_stack = Vector{Int64}(undef, max(0, div(i_n - 1, 1) + 1) * max(0, div(i_nc - 1, 1) + 1))
    u_stack = Vector{Float64}(undef, max(0, div(i_n - 1, 1) + 1) + max(0, div(i_n - 1, 1) + 1))
    return (wv_stack, branch_stack, u_stack)
end

function remat_shapes_b(x, xb, w, wb, y, yb, z, zb, u, ub, loss, lossb, s, sb, i_n, i_nc, wv_stack, branch_stack, u_stack)
    a = 0.0
    sc = 0.0
    wv = 0.0
    ab = 0.0
    scb = 0.0
    wvb = 0.0
    for t = 1:i_n
        a = 0.0
        for ci = 1:i_nc
            __icse_0 = ci - 1
            base = __icse_0 * 2
            __idx_wv_stack_1 = ((t - 1) + __icse_0 * (div(i_n - 1, 1) + 1)) + 1
            wv_stack[__idx_wv_stack_1] = wv
            wv = w[base + 1]
            a = a + wv * wv * x[(t + ci) - 1]
            wv = w[base + 2]
            a = a + wv * wv * x[t + ci]
        end
        y[t] = a
        __idx_wv_stack_3 = max(0, div(i_n - 1, 1) + 1) * max(0, div(i_nc - 1, 1) + 1) + ((t - 1) + 1)
        wv_stack[__idx_wv_stack_3] = wv
    end
    for t = 1:i_n
        a = 0.0
        for ci = 1:i_nc
            __icse_1 = div(i_n - 1, 1) + 1
            __icse_2 = max(0, __icse_1)
            __icse_3 = __icse_2 * max(0, div(i_nc - 1, 1) + 1)
            __icse_4 = __icse_3 + __icse_2
            __icse_5 = ((t - 1) + (ci - 1) * __icse_1) + 1
            __idx_wv_stack_0 = __icse_4 + __icse_5
            wv_stack[__idx_wv_stack_0] = wv
            wv = w[ci]
            a = a + wv * wv * x[(t + ci) - 1]
            wv = wv * 0.5
            __cse_6 = x[t + ci]
            a = a + wv * wv * __cse_6
            __idx_wv_stack_6 = (__icse_4 + __icse_3) + __icse_5
            wv_stack[__idx_wv_stack_6] = wv
            wv = w[ci + 1]
            a = a + wv * wv * __cse_6
        end
        z[t] = a
        __icse_7 = max(0, div(i_n - 1, 1) + 1)
        __icse_8 = __icse_7 * max(0, div(i_nc - 1, 1) + 1)
        __idx_wv_stack_3 = (((__icse_8 + __icse_7) + __icse_8) + __icse_8) + ((t - 1) + 1)
        wv_stack[__idx_wv_stack_3] = wv
    end
    for t = 1:i_n
        a = 0.0
        for ci = 1:i_nc
            __icse_9 = div(i_n - 1, 1) + 1
            __icse_10 = max(0, __icse_9)
            __icse_11 = __icse_10 * max(0, div(i_nc - 1, 1) + 1)
            __idx_wv_stack_0 = ((((__icse_11 + __icse_10) + __icse_11) + __icse_11) + __icse_10) + (((t - 1) + (ci - 1) * __icse_9) + 1)
            wv_stack[__idx_wv_stack_0] = wv
            __cse_12 = y[t]
            wv = __cse_12
            a = a + wv * wv * x[(t + ci) - 1]
            wv = __cse_12 * 0.5
            a = a + wv * wv * x[t + ci]
        end
        z[i_n + t] = a
        __icse_13 = max(0, div(i_n - 1, 1) + 1)
        __icse_14 = __icse_13 * max(0, div(i_nc - 1, 1) + 1)
        __idx_wv_stack_3 = (((((__icse_14 + __icse_13) + __icse_14) + __icse_14) + __icse_13) + __icse_14) + ((t - 1) + 1)
        wv_stack[__idx_wv_stack_3] = wv
    end
    for t = 1:i_n
        a = 0.0
        for ci = 1:i_nc
            __icse_15 = div(i_n - 1, 1) + 1
            __icse_16 = max(0, __icse_15)
            __icse_17 = __icse_16 * max(0, div(i_nc - 1, 1) + 1)
            __idx_wv_stack_0 = ((((((__icse_17 + __icse_16) + __icse_17) + __icse_17) + __icse_16) + __icse_17) + __icse_16) + (((t - 1) + (ci - 1) * __icse_15) + 1)
            wv_stack[__idx_wv_stack_0] = wv
            wv = w[ci]
            for j = 1:2
                __icse_18 = div(i_n - 1, 1) + 1
                __icse_19 = max(0, __icse_18)
                __icse_20 = __icse_19 * max(0, div(i_nc - 1, 1) + 1)
                __idx_wv_stack_0 = (((((((__icse_20 + __icse_19) + __icse_20) + __icse_20) + __icse_19) + __icse_20) + __icse_19) + __icse_20) + (((t - 1) + ((ci - 1) * (div(2 - 1, 1) + 1) + (j - 1)) * __icse_18) + 1)
                wv_stack[__idx_wv_stack_0] = wv
                wv = wv * w[ci + j]
            end
            a = a + wv * wv * x[(t + ci) - 1]
            __icse_21 = div(i_n - 1, 1) + 1
            __icse_22 = max(0, __icse_21)
            __icse_23 = max(0, div(i_nc - 1, 1) + 1)
            __icse_24 = __icse_22 * __icse_23
            __icse_25 = (((((((__icse_24 + __icse_22) + __icse_24) + __icse_24) + __icse_22) + __icse_24) + __icse_22) + __icse_24) + __icse_22 * __icse_23 * max(0, div(2 - 1, 1) + 1)
            __icse_26 = ((t - 1) + (ci - 1) * __icse_21) + 1
            __idx_wv_stack_5 = __icse_25 + __icse_26
            wv_stack[__idx_wv_stack_5] = wv
            wv = w[ci + 1]
            a = a + wv * wv * x[t + ci]
            __idx_wv_stack_9 = (__icse_25 + __icse_24) + __icse_26
            wv_stack[__idx_wv_stack_9] = wv
        end
        z[2i_n + t] = a
    end
    for t = 1:i_n
        a = 0.0
        for ci = 1:i_nc
            wv = 0.0
            if x[(t + ci) - 1] > 0.0
                __idx_branch_stack_0 = ((t - 1) + (ci - 1) * (div(i_n - 1, 1) + 1)) + 1
                branch_stack[__idx_branch_stack_0] = 1
                wv = w[ci]
            else
                __idx_branch_stack_0 = ((t - 1) + (ci - 1) * (div(i_n - 1, 1) + 1)) + 1
                branch_stack[__idx_branch_stack_0] = 0
            end
            a = a + wv * wv * x[(t + ci) - 1]
            __icse_27 = div(i_n - 1, 1) + 1
            __icse_28 = max(0, __icse_27)
            __icse_29 = max(0, div(i_nc - 1, 1) + 1)
            __icse_30 = __icse_28 * __icse_29
            __icse_31 = (((((((((__icse_30 + __icse_28) + __icse_30) + __icse_30) + __icse_28) + __icse_30) + __icse_28) + __icse_30) + __icse_28 * __icse_29 * max(0, div(2 - 1, 1) + 1)) + __icse_30) + __icse_30
            __icse_32 = ((t - 1) + (ci - 1) * __icse_27) + 1
            __idx_wv_stack_3 = __icse_31 + __icse_32
            wv_stack[__idx_wv_stack_3] = wv
            wv = w[ci + 1]
            a = a + wv * wv * x[t + ci]
            __idx_wv_stack_7 = (__icse_31 + __icse_30) + __icse_32
            wv_stack[__idx_wv_stack_7] = wv
        end
        z[3i_n + t] = a
    end
    for t = 1:i_n
        a = 0.0
        for ci = 1:i_nc
            sc = x[ci] * 2.0
            __icse_33 = div(i_n - 1, 1) + 1
            __icse_34 = max(0, __icse_33)
            __icse_35 = max(0, div(i_nc - 1, 1) + 1)
            __icse_36 = __icse_34 * __icse_35
            __icse_37 = (((((((((((__icse_36 + __icse_34) + __icse_36) + __icse_36) + __icse_34) + __icse_36) + __icse_34) + __icse_36) + __icse_34 * __icse_35 * max(0, div(2 - 1, 1) + 1)) + __icse_36) + __icse_36) + __icse_36) + __icse_36
            __icse_38 = ((t - 1) + (ci - 1) * __icse_33) + 1
            __idx_wv_stack_1 = __icse_37 + __icse_38
            wv_stack[__idx_wv_stack_1] = wv
            wv = w[ci] * sc
            a = a + wv * wv * x[(t + ci) - 1]
            __idx_wv_stack_5 = (__icse_37 + __icse_36) + __icse_38
            wv_stack[__idx_wv_stack_5] = wv
            wv = w[ci + 1] * sc
            a = a + wv * wv * x[t + ci]
        end
        z[4i_n + t] = a
        __icse_39 = max(0, div(i_n - 1, 1) + 1)
        __icse_40 = max(0, div(i_nc - 1, 1) + 1)
        __icse_41 = __icse_39 * __icse_40
        __idx_wv_stack_3 = ((((((((((((((__icse_41 + __icse_39) + __icse_41) + __icse_41) + __icse_39) + __icse_41) + __icse_39) + __icse_41) + __icse_39 * __icse_40 * max(0, div(2 - 1, 1) + 1)) + __icse_41) + __icse_41) + __icse_41) + __icse_41) + __icse_41) + __icse_41) + ((t - 1) + 1)
        wv_stack[__idx_wv_stack_3] = wv
    end
    for t = 1:i_n
        a = 0.0
        for ci = 1:i_nc
            __icse_42 = div(i_n - 1, 1) + 1
            __icse_43 = max(0, __icse_42)
            __icse_44 = max(0, div(i_nc - 1, 1) + 1)
            __icse_45 = __icse_43 * __icse_44
            __idx_wv_stack_0 = (((((((((((((((__icse_45 + __icse_43) + __icse_45) + __icse_45) + __icse_43) + __icse_45) + __icse_43) + __icse_45) + __icse_43 * __icse_44 * max(0, div(2 - 1, 1) + 1)) + __icse_45) + __icse_45) + __icse_45) + __icse_45) + __icse_45) + __icse_45) + __icse_43) + (((t - 1) + (ci - 1) * __icse_42) + 1)
            wv_stack[__idx_wv_stack_0] = wv
            wv = w[ci] * s
            a = a + wv * wv * x[(t + ci) - 1]
            wv = w[ci + 1] * s
            a = a + wv * wv * x[t + ci]
        end
        z[5i_n + t] = a
        __icse_46 = max(0, div(i_n - 1, 1) + 1)
        __icse_47 = max(0, div(i_nc - 1, 1) + 1)
        __icse_48 = __icse_46 * __icse_47
        __idx_wv_stack_3 = ((((((((((((((((__icse_48 + __icse_46) + __icse_48) + __icse_48) + __icse_46) + __icse_48) + __icse_46) + __icse_48) + __icse_46 * __icse_47 * max(0, div(2 - 1, 1) + 1)) + __icse_48) + __icse_48) + __icse_48) + __icse_48) + __icse_48) + __icse_48) + __icse_46) + __icse_48) + ((t - 1) + 1)
        wv_stack[__idx_wv_stack_3] = wv
    end
    for t = 1:i_n
        __icse_49 = max(0, div(i_n - 1, 1) + 1)
        __icse_50 = max(0, div(i_nc - 1, 1) + 1)
        __icse_51 = __icse_49 * __icse_50
        __icse_52 = ((((((((((((((((__icse_51 + __icse_49) + __icse_51) + __icse_51) + __icse_49) + __icse_51) + __icse_49) + __icse_51) + __icse_49 * __icse_50 * max(0, div(2 - 1, 1) + 1)) + __icse_51) + __icse_51) + __icse_51) + __icse_51) + __icse_51) + __icse_51) + __icse_49) + __icse_51) + __icse_49
        __icse_53 = (t - 1) + 1
        __idx_wv_stack_0 = __icse_52 + __icse_53
        wv_stack[__idx_wv_stack_0] = wv
        __cse_54 = u[t]
        wv = __cse_54
        __idx_u_stack_3 = __icse_53
        u_stack[__idx_u_stack_3] = __cse_54
        __cse_55 = x[t]
        u[t] = __cse_54 * 0.5 + __cse_55
        a = wv * wv * __cse_55
        __idx_wv_stack_7 = (__icse_52 + __icse_49) + __icse_53
        wv_stack[__idx_wv_stack_7] = wv
        wv = u[t]
        a = a + wv * wv * x[t + 1]
        z[6i_n + t] = a
    end
    for t = 1:i_n
        __icse_56 = max(0, div(i_n - 1, 1) + 1)
        __icse_57 = (t - 1) + 1
        __idx_u_stack_0 = __icse_56 + __icse_57
        u_stack[__idx_u_stack_0] = u[1]
        __cse_58 = x[t]
        __cse_59 = x[t + 1]
        u[1] = __cse_58 * 0.5 + __cse_59
        __icse_60 = max(0, div(i_nc - 1, 1) + 1)
        __icse_61 = __icse_56 * __icse_60
        __icse_62 = ((((((((((((((((((__icse_61 + __icse_56) + __icse_61) + __icse_61) + __icse_56) + __icse_61) + __icse_56) + __icse_61) + __icse_56 * __icse_60 * max(0, div(2 - 1, 1) + 1)) + __icse_61) + __icse_61) + __icse_61) + __icse_61) + __icse_61) + __icse_61) + __icse_56) + __icse_61) + __icse_56) + __icse_56) + __icse_56
        __idx_wv_stack_3 = __icse_62 + __icse_57
        wv_stack[__idx_wv_stack_3] = wv
        __cse_63 = u[1]
        wv = __cse_63
        a = wv * wv * __cse_58
        __idx_wv_stack_7 = (__icse_62 + __icse_56) + __icse_57
        wv_stack[__idx_wv_stack_7] = wv
        wv = __cse_63 * 2.0
        a = a + wv * wv * __cse_59
        z[7i_n + t] = a
    end
    for m = 1:8i_n
        __cse_64 = z[m]
        loss[1] = loss[1] + __cse_64 * __cse_64
    end
    for m = 8i_n:-1:1
        __cse_65 = z[m] * lossb[1]
        zb[m] = zb[m] + __cse_65
        zb[m] = zb[m] + __cse_65
    end
    for t = i_n:-1:1
        __oldb_0 = zb[7i_n + t]
        zb[7i_n + t] = 0.0
        ab = ab + __oldb_0
        __cse_66 = (wv * x[t + 1]) * ab
        wvb = wvb + __cse_66
        wvb = wvb + __cse_66
        xb[t + 1] = xb[t + 1] + (wv * wv) * ab
        __icse_67 = max(0, div(i_n - 1, 1) + 1)
        __icse_68 = max(0, div(i_nc - 1, 1) + 1)
        __icse_69 = __icse_67 * __icse_68
        __icse_70 = ((((((((((((((((((__icse_69 + __icse_67) + __icse_69) + __icse_69) + __icse_67) + __icse_69) + __icse_67) + __icse_69) + __icse_67 * __icse_68 * max(0, div(2 - 1, 1) + 1)) + __icse_69) + __icse_69) + __icse_69) + __icse_69) + __icse_69) + __icse_69) + __icse_67) + __icse_69) + __icse_67) + __icse_67) + __icse_67
        __icse_71 = (t - 1) + 1
        __idx_wv_stack_0 = (__icse_70 + __icse_67) + __icse_71
        wv = wv_stack[__idx_wv_stack_0]
        __oldb_2 = wvb
        wvb = 0.0
        ub[1] = ub[1] + 2.0__oldb_2
        __oldb_0 = ab
        ab = 0.0
        __cse_72 = (wv * x[t]) * __oldb_0
        wvb = wvb + __cse_72
        wvb = wvb + __cse_72
        xb[t] = xb[t] + (wv * wv) * __oldb_0
        __idx_wv_stack_0 = __icse_70 + __icse_71
        wv = wv_stack[__idx_wv_stack_0]
        __oldb_2 = wvb
        wvb = 0.0
        ub[1] = ub[1] + __oldb_2
        __idx_u_stack_0 = __icse_67 + __icse_71
        u[1] = u_stack[__idx_u_stack_0]
        __oldb_2 = ub[1]
        ub[1] = 0.0
        xb[t] = xb[t] + 0.5__oldb_2
        xb[t + 1] = xb[t + 1] + __oldb_2
    end
    for t = i_n:-1:1
        __oldb_0 = zb[6i_n + t]
        zb[6i_n + t] = 0.0
        ab = ab + __oldb_0
        __cse_73 = (wv * x[t + 1]) * ab
        wvb = wvb + __cse_73
        wvb = wvb + __cse_73
        xb[t + 1] = xb[t + 1] + (wv * wv) * ab
        __icse_74 = max(0, div(i_n - 1, 1) + 1)
        __icse_75 = max(0, div(i_nc - 1, 1) + 1)
        __icse_76 = __icse_74 * __icse_75
        __icse_77 = ((((((((((((((((__icse_76 + __icse_74) + __icse_76) + __icse_76) + __icse_74) + __icse_76) + __icse_74) + __icse_76) + __icse_74 * __icse_75 * max(0, div(2 - 1, 1) + 1)) + __icse_76) + __icse_76) + __icse_76) + __icse_76) + __icse_76) + __icse_76) + __icse_74) + __icse_76) + __icse_74
        __icse_78 = (t - 1) + 1
        __idx_wv_stack_0 = (__icse_77 + __icse_74) + __icse_78
        wv = wv_stack[__idx_wv_stack_0]
        __oldb_2 = wvb
        wvb = 0.0
        ub[t] = ub[t] + __oldb_2
        __oldb_0 = ab
        ab = 0.0
        __cse_79 = (wv * x[t]) * __oldb_0
        wvb = wvb + __cse_79
        wvb = wvb + __cse_79
        xb[t] = xb[t] + (wv * wv) * __oldb_0
        __idx_u_stack_0 = __icse_78
        u[t] = u_stack[__idx_u_stack_0]
        __oldb_2 = ub[t]
        ub[t] = 0.0
        ub[t] = ub[t] + 0.5__oldb_2
        xb[t] = xb[t] + __oldb_2
        __idx_wv_stack_0 = __icse_77 + __icse_78
        wv = wv_stack[__idx_wv_stack_0]
        __oldb_2 = wvb
        wvb = 0.0
        ub[t] = ub[t] + __oldb_2
    end
    for t = i_n:-1:1
        __icse_80 = max(0, div(i_n - 1, 1) + 1)
        __icse_81 = max(0, div(i_nc - 1, 1) + 1)
        __icse_82 = __icse_80 * __icse_81
        __idx_wv_stack_0 = ((((((((((((((((__icse_82 + __icse_80) + __icse_82) + __icse_82) + __icse_80) + __icse_82) + __icse_80) + __icse_82) + __icse_80 * __icse_81 * max(0, div(2 - 1, 1) + 1)) + __icse_82) + __icse_82) + __icse_82) + __icse_82) + __icse_82) + __icse_82) + __icse_80) + __icse_82) + ((t - 1) + 1)
        wv = wv_stack[__idx_wv_stack_0]
        __oldb_0 = zb[5i_n + t]
        zb[5i_n + t] = 0.0
        ab = ab + __oldb_0
        for ci = i_nc:-1:1
            __cse_83 = (wv * x[t + ci]) * ab
            wvb = wvb + __cse_83
            wvb = wvb + __cse_83
            xb[t + ci] = xb[t + ci] + (wv * wv) * ab
            __cse_84 = w[ci]
            wv = __cse_84 * s
            __oldb_1 = wvb
            wvb = 0.0
            wb[ci + 1] = wb[ci + 1] + s * __oldb_1
            sb = sb + w[ci + 1] * __oldb_1
            __cse_85 = (wv * x[(t + ci) - 1]) * ab
            wvb = wvb + __cse_85
            wvb = wvb + __cse_85
            xb[(t + ci) - 1] = xb[(t + ci) - 1] + (wv * wv) * ab
            __icse_86 = div(i_n - 1, 1) + 1
            __icse_87 = max(0, __icse_86)
            __icse_88 = max(0, div(i_nc - 1, 1) + 1)
            __icse_89 = __icse_87 * __icse_88
            __idx_wv_stack_0 = (((((((((((((((__icse_89 + __icse_87) + __icse_89) + __icse_89) + __icse_87) + __icse_89) + __icse_87) + __icse_89) + __icse_87 * __icse_88 * max(0, div(2 - 1, 1) + 1)) + __icse_89) + __icse_89) + __icse_89) + __icse_89) + __icse_89) + __icse_89) + __icse_87) + (((t - 1) + (ci - 1) * __icse_86) + 1)
            wv = wv_stack[__idx_wv_stack_0]
            __oldb_2 = wvb
            wvb = 0.0
            wb[ci] = wb[ci] + s * __oldb_2
            sb = sb + __cse_84 * __oldb_2
        end
        ab = 0.0
    end
    for t = i_n:-1:1
        __icse_90 = max(0, div(i_n - 1, 1) + 1)
        __icse_91 = max(0, div(i_nc - 1, 1) + 1)
        __icse_92 = __icse_90 * __icse_91
        __idx_wv_stack_0 = ((((((((((((((__icse_92 + __icse_90) + __icse_92) + __icse_92) + __icse_90) + __icse_92) + __icse_90) + __icse_92) + __icse_90 * __icse_91 * max(0, div(2 - 1, 1) + 1)) + __icse_92) + __icse_92) + __icse_92) + __icse_92) + __icse_92) + __icse_92) + ((t - 1) + 1)
        wv = wv_stack[__idx_wv_stack_0]
        __oldb_0 = zb[4i_n + t]
        zb[4i_n + t] = 0.0
        ab = ab + __oldb_0
        for ci = i_nc:-1:1
            sc = x[ci] * 2.0
            __cse_93 = (wv * x[t + ci]) * ab
            wvb = wvb + __cse_93
            wvb = wvb + __cse_93
            xb[t + ci] = xb[t + ci] + (wv * wv) * ab
            __icse_94 = div(i_n - 1, 1) + 1
            __icse_95 = max(0, __icse_94)
            __icse_96 = max(0, div(i_nc - 1, 1) + 1)
            __icse_97 = __icse_95 * __icse_96
            __icse_98 = (((((((((((__icse_97 + __icse_95) + __icse_97) + __icse_97) + __icse_95) + __icse_97) + __icse_95) + __icse_97) + __icse_95 * __icse_96 * max(0, div(2 - 1, 1) + 1)) + __icse_97) + __icse_97) + __icse_97) + __icse_97
            __icse_99 = ((t - 1) + (ci - 1) * __icse_94) + 1
            __idx_wv_stack_0 = (__icse_98 + __icse_97) + __icse_99
            wv = wv_stack[__idx_wv_stack_0]
            __oldb_2 = wvb
            wvb = 0.0
            wb[ci + 1] = wb[ci + 1] + sc * __oldb_2
            scb = scb + w[ci + 1] * __oldb_2
            __cse_100 = (wv * x[(t + ci) - 1]) * ab
            wvb = wvb + __cse_100
            wvb = wvb + __cse_100
            xb[(t + ci) - 1] = xb[(t + ci) - 1] + (wv * wv) * ab
            __idx_wv_stack_0 = __icse_98 + __icse_99
            wv = wv_stack[__idx_wv_stack_0]
            __oldb_2 = wvb
            wvb = 0.0
            wb[ci] = wb[ci] + sc * __oldb_2
            scb = scb + w[ci] * __oldb_2
            __oldb_0 = scb
            scb = 0.0
            xb[ci] = xb[ci] + 2.0__oldb_0
        end
        ab = 0.0
    end
    for t = i_n:-1:1
        __oldb_0 = zb[3i_n + t]
        zb[3i_n + t] = 0.0
        ab = ab + __oldb_0
        for ci = i_nc:-1:1
            __icse_101 = div(i_n - 1, 1) + 1
            __icse_102 = max(0, __icse_101)
            __icse_103 = max(0, div(i_nc - 1, 1) + 1)
            __icse_104 = __icse_102 * __icse_103
            __icse_105 = (((((((((__icse_104 + __icse_102) + __icse_104) + __icse_104) + __icse_102) + __icse_104) + __icse_102) + __icse_104) + __icse_102 * __icse_103 * max(0, div(2 - 1, 1) + 1)) + __icse_104) + __icse_104
            __icse_106 = ((t - 1) + (ci - 1) * __icse_101) + 1
            __idx_wv_stack_0 = (__icse_105 + __icse_104) + __icse_106
            wv = wv_stack[__idx_wv_stack_0]
            __cse_107 = (wv * x[t + ci]) * ab
            wvb = wvb + __cse_107
            wvb = wvb + __cse_107
            xb[t + ci] = xb[t + ci] + (wv * wv) * ab
            __idx_wv_stack_0 = __icse_105 + __icse_106
            wv = wv_stack[__idx_wv_stack_0]
            __oldb_2 = wvb
            wvb = 0.0
            wb[ci + 1] = wb[ci + 1] + __oldb_2
            __cse_108 = (wv * x[(t + ci) - 1]) * ab
            wvb = wvb + __cse_108
            wvb = wvb + __cse_108
            xb[(t + ci) - 1] = xb[(t + ci) - 1] + (wv * wv) * ab
            __idx_branch_stack_13 = __icse_106
            __branch = branch_stack[__idx_branch_stack_13]
            if __branch == 1
                __oldb_0 = wvb
                wvb = 0.0
                wb[ci] = wb[ci] + __oldb_0
            end
            wvb = 0.0
        end
        ab = 0.0
    end
    for t = i_n:-1:1
        __oldb_0 = zb[2i_n + t]
        zb[2i_n + t] = 0.0
        ab = ab + __oldb_0
        for ci = i_nc:-1:1
            __icse_109 = div(i_n - 1, 1) + 1
            __icse_110 = max(0, __icse_109)
            __icse_111 = max(0, div(i_nc - 1, 1) + 1)
            __icse_112 = __icse_110 * __icse_111
            __icse_113 = (((((((__icse_112 + __icse_110) + __icse_112) + __icse_112) + __icse_110) + __icse_112) + __icse_110) + __icse_112) + __icse_110 * __icse_111 * max(0, div(2 - 1, 1) + 1)
            __icse_114 = ((t - 1) + (ci - 1) * __icse_109) + 1
            __idx_wv_stack_0 = (__icse_113 + __icse_112) + __icse_114
            wv = wv_stack[__idx_wv_stack_0]
            __cse_115 = (wv * x[t + ci]) * ab
            wvb = wvb + __cse_115
            wvb = wvb + __cse_115
            xb[t + ci] = xb[t + ci] + (wv * wv) * ab
            __idx_wv_stack_0 = __icse_113 + __icse_114
            wv = wv_stack[__idx_wv_stack_0]
            __oldb_2 = wvb
            wvb = 0.0
            wb[ci + 1] = wb[ci + 1] + __oldb_2
            __cse_116 = (wv * x[(t + ci) - 1]) * ab
            wvb = wvb + __cse_116
            wvb = wvb + __cse_116
            xb[(t + ci) - 1] = xb[(t + ci) - 1] + (wv * wv) * ab
            for j = 2:-1:1
                __icse_117 = div(i_n - 1, 1) + 1
                __icse_118 = max(0, __icse_117)
                __icse_119 = __icse_118 * max(0, div(i_nc - 1, 1) + 1)
                __idx_wv_stack_0 = (((((((__icse_119 + __icse_118) + __icse_119) + __icse_119) + __icse_118) + __icse_119) + __icse_118) + __icse_119) + (((t - 1) + ((ci - 1) * (div(2 - 1, 1) + 1) + (j - 1)) * __icse_117) + 1)
                wv = wv_stack[__idx_wv_stack_0]
                __oldb_2 = wvb
                wvb = 0.0
                wvb = wvb + w[ci + j] * __oldb_2
                wb[ci + j] = wb[ci + j] + wv * __oldb_2
            end
            __icse_120 = div(i_n - 1, 1) + 1
            __icse_121 = max(0, __icse_120)
            __icse_122 = __icse_121 * max(0, div(i_nc - 1, 1) + 1)
            __idx_wv_stack_0 = ((((((__icse_122 + __icse_121) + __icse_122) + __icse_122) + __icse_121) + __icse_122) + __icse_121) + (((t - 1) + (ci - 1) * __icse_120) + 1)
            wv = wv_stack[__idx_wv_stack_0]
            __oldb_2 = wvb
            wvb = 0.0
            wb[ci] = wb[ci] + __oldb_2
        end
        ab = 0.0
    end
    for t = i_n:-1:1
        __icse_123 = max(0, div(i_n - 1, 1) + 1)
        __icse_124 = __icse_123 * max(0, div(i_nc - 1, 1) + 1)
        __idx_wv_stack_0 = (((((__icse_124 + __icse_123) + __icse_124) + __icse_124) + __icse_123) + __icse_124) + ((t - 1) + 1)
        wv = wv_stack[__idx_wv_stack_0]
        __oldb_0 = zb[i_n + t]
        zb[i_n + t] = 0.0
        ab = ab + __oldb_0
        for ci = i_nc:-1:1
            __cse_125 = (wv * x[t + ci]) * ab
            wvb = wvb + __cse_125
            wvb = wvb + __cse_125
            xb[t + ci] = xb[t + ci] + (wv * wv) * ab
            wv = y[t]
            __oldb_1 = wvb
            wvb = 0.0
            yb[t] = yb[t] + 0.5__oldb_1
            __cse_126 = (wv * x[(t + ci) - 1]) * ab
            wvb = wvb + __cse_126
            wvb = wvb + __cse_126
            xb[(t + ci) - 1] = xb[(t + ci) - 1] + (wv * wv) * ab
            __icse_127 = div(i_n - 1, 1) + 1
            __icse_128 = max(0, __icse_127)
            __icse_129 = __icse_128 * max(0, div(i_nc - 1, 1) + 1)
            __idx_wv_stack_0 = ((((__icse_129 + __icse_128) + __icse_129) + __icse_129) + __icse_128) + (((t - 1) + (ci - 1) * __icse_127) + 1)
            wv = wv_stack[__idx_wv_stack_0]
            __oldb_2 = wvb
            wvb = 0.0
            yb[t] = yb[t] + __oldb_2
        end
        ab = 0.0
    end
    for t = i_n:-1:1
        __icse_130 = max(0, div(i_n - 1, 1) + 1)
        __icse_131 = __icse_130 * max(0, div(i_nc - 1, 1) + 1)
        __idx_wv_stack_0 = (((__icse_131 + __icse_130) + __icse_131) + __icse_131) + ((t - 1) + 1)
        wv = wv_stack[__idx_wv_stack_0]
        __oldb_0 = zb[t]
        zb[t] = 0.0
        ab = ab + __oldb_0
        for ci = i_nc:-1:1
            __cse_132 = x[t + ci]
            __cse_133 = (wv * __cse_132) * ab
            wvb = wvb + __cse_133
            wvb = wvb + __cse_133
            xb[t + ci] = xb[t + ci] + (wv * wv) * ab
            __icse_134 = div(i_n - 1, 1) + 1
            __icse_135 = max(0, __icse_134)
            __icse_136 = __icse_135 * max(0, div(i_nc - 1, 1) + 1)
            __icse_137 = __icse_136 + __icse_135
            __icse_138 = ((t - 1) + (ci - 1) * __icse_134) + 1
            __idx_wv_stack_0 = (__icse_137 + __icse_136) + __icse_138
            wv = wv_stack[__idx_wv_stack_0]
            __oldb_2 = wvb
            wvb = 0.0
            wb[ci + 1] = wb[ci + 1] + __oldb_2
            __cse_139 = (wv * __cse_132) * ab
            wvb = wvb + __cse_139
            wvb = wvb + __cse_139
            xb[t + ci] = xb[t + ci] + (wv * wv) * ab
            wv = w[ci]
            __oldb_1 = wvb
            wvb = 0.0
            wvb = wvb + 0.5__oldb_1
            __cse_140 = (wv * x[(t + ci) - 1]) * ab
            wvb = wvb + __cse_140
            wvb = wvb + __cse_140
            xb[(t + ci) - 1] = xb[(t + ci) - 1] + (wv * wv) * ab
            __idx_wv_stack_0 = __icse_137 + __icse_138
            wv = wv_stack[__idx_wv_stack_0]
            __oldb_2 = wvb
            wvb = 0.0
            wb[ci] = wb[ci] + __oldb_2
        end
        ab = 0.0
    end
    for t = i_n:-1:1
        __idx_wv_stack_0 = max(0, div(i_n - 1, 1) + 1) * max(0, div(i_nc - 1, 1) + 1) + ((t - 1) + 1)
        wv = wv_stack[__idx_wv_stack_0]
        __oldb_0 = yb[t]
        yb[t] = 0.0
        ab = ab + __oldb_0
        for ci = i_nc:-1:1
            __icse_141 = ci - 1
            base = __icse_141 * 2
            __cse_142 = (wv * x[t + ci]) * ab
            wvb = wvb + __cse_142
            wvb = wvb + __cse_142
            xb[t + ci] = xb[t + ci] + (wv * wv) * ab
            wv = w[base + 1]
            __oldb_1 = wvb
            wvb = 0.0
            wb[base + 2] = wb[base + 2] + __oldb_1
            __cse_143 = (wv * x[(t + ci) - 1]) * ab
            wvb = wvb + __cse_143
            wvb = wvb + __cse_143
            xb[(t + ci) - 1] = xb[(t + ci) - 1] + (wv * wv) * ab
            __idx_wv_stack_0 = ((t - 1) + __icse_141 * (div(i_n - 1, 1) + 1)) + 1
            wv = wv_stack[__idx_wv_stack_0]
            __oldb_2 = wvb
            wvb = 0.0
            wb[base + 1] = wb[base + 1] + __oldb_2
        end
        ab = 0.0
    end
    return sb
end

function remat_shapes(x, w, y, z, u, loss, s, i_n, i_nc)
    for t = 1:i_n
        a = 0.0
        for ci = 1:i_nc
            base = (ci - 1) * 2
            wv = w[base + 1]
            a = a + wv * wv * x[(t + ci) - 1]
            wv = w[base + 2]
            a = a + wv * wv * x[t + ci]
        end
        y[t] = a
    end
    for t = 1:i_n
        a = 0.0
        for ci = 1:i_nc
            wv = w[ci]
            a = a + wv * wv * x[(t + ci) - 1]
            wv = wv * 0.5
            a = a + wv * wv * x[t + ci]
            wv = w[ci + 1]
            a = a + wv * wv * x[t + ci]
        end
        z[t] = a
    end
    for t = 1:i_n
        a = 0.0
        for ci = 1:i_nc
            wv = y[t]
            a = a + wv * wv * x[(t + ci) - 1]
            wv = y[t] * 0.5
            a = a + wv * wv * x[t + ci]
        end
        z[i_n + t] = a
    end
    for t = 1:i_n
        a = 0.0
        for ci = 1:i_nc
            wv = w[ci]
            for j = 1:2
                wv = wv * w[ci + j]
            end
            a = a + wv * wv * x[(t + ci) - 1]
            wv = w[ci + 1]
            a = a + wv * wv * x[t + ci]
        end
        z[2i_n + t] = a
    end
    for t = 1:i_n
        a = 0.0
        for ci = 1:i_nc
            wv = 0.0
            if x[(t + ci) - 1] > 0.0
                wv = w[ci]
            end
            a = a + wv * wv * x[(t + ci) - 1]
            wv = w[ci + 1]
            a = a + wv * wv * x[t + ci]
        end
        z[3i_n + t] = a
    end
    for t = 1:i_n
        a = 0.0
        for ci = 1:i_nc
            sc = x[ci] * 2.0
            wv = w[ci] * sc
            a = a + wv * wv * x[(t + ci) - 1]
            wv = w[ci + 1] * sc
            a = a + wv * wv * x[t + ci]
        end
        z[4i_n + t] = a
    end
    for t = 1:i_n
        a = 0.0
        for ci = 1:i_nc
            wv = w[ci] * s
            a = a + wv * wv * x[(t + ci) - 1]
            wv = w[ci + 1] * s
            a = a + wv * wv * x[t + ci]
        end
        z[5i_n + t] = a
    end
    for t = 1:i_n
        wv = u[t]
        u[t] = u[t] * 0.5 + x[t]
        a = wv * wv * x[t]
        wv = u[t]
        a = a + wv * wv * x[t + 1]
        z[6i_n + t] = a
    end
    for t = 1:i_n
        u[1] = x[t] * 0.5 + x[t + 1]
        wv = u[1]
        a = wv * wv * x[t]
        wv = u[1] * 2.0
        a = a + wv * wv * x[t + 1]
        z[7i_n + t] = a
    end
    for m = 1:8i_n
        loss[1] = loss[1] + z[m] * z[m]
    end
end
