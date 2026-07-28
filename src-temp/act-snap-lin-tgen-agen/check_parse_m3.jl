# ============================================================
# check_parse_m3.jl -- quick local verification for parse_kernel /
# shape_infer against the M3 ("loop-carried recurrences") corpus
# primals.
#
# Not part of STADE.jl itself -- a throwaway harness, same pattern
# as check_parse_m1.jl. Run with:
#     julia check_parse_m3.jl
# from a directory containing STADE.jl and all_b.jl.
#
# NB: the val_fixtures.jl wrapper for this tier's advection kernel
# is named val_fixture_advection, but the underlying primal function
# in all_b.jl is named `func` -- that's the name parse_kernel sees.
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

m3_names = [:geomrecur, :stencil_loss, :func]

println("--- M3 parse_kernel/shape_infer check ---")
for name in m3_names
    expr = grab_kernel_expr("all_b.jl", name)
    kernel = parse_kernel(expr)
    println(rpad(String(name), 14), " args=", kernel.sig.args)
    println(" "^14, " kinds=", kernel.sig.kinds)
    println(" "^14, " independents=", kernel.sig.independents,
            " dependents=", kernel.sig.dependents)
    println(" "^14, " #statements(top level)=", length(kernel.body))
end
println("--- all M3 kernels parsed without a hard error ---")