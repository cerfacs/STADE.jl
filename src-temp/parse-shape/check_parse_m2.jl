# ============================================================
# check_parse_m2.jl -- quick local verification for parse_kernel /
# shape_infer against the M2 ("conditionals") corpus primals.
#
# Not part of STADE.jl itself -- a throwaway harness, same pattern
# as check_parse_m1.jl. Run with:
#     julia check_parse_m2.jl
# from a directory containing STADE.jl and all_b.jl.
# ============================================================

include("STADE.jl")

# pull `function name(...) ... end` back out of all_b.jl as a raw
# Expr, the same way io_read_kernel would from a single-function file
function grab_kernel_expr(path::String, name::Symbol)
    parsed = Meta.parseall(read(path, String))
    for e in parsed.args
        if e isa Expr && e.head == :function
            sig = e.args[1]
            if sig isa Expr && sig.head == :call && sig.args[1] == name
                return e
            end
        end
    end
    error("no `function $name(...)` found in $path")
end

m2_names = [
    :branchsel, :clamped_sumsq, :cond_field_choice, :cond_loop_choice, :relu_field,
]

println("--- M2 parse_kernel/shape_infer check ---")
for name in m2_names
    expr = grab_kernel_expr("all_b.jl", name)
    kernel = parse_kernel(expr)
    println(rpad(String(name), 18), " args=", kernel.sig.args)
    println(" "^18, " kinds=", kernel.sig.kinds)
    println(" "^18, " independents=", kernel.sig.independents,
            " dependents=", kernel.sig.dependents)
    println(" "^18, " #statements(top level)=", length(kernel.body))
end
println("--- all M2 kernels parsed without a hard error ---")