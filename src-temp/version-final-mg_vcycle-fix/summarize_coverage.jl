# ============================================================
# summarize_coverage.jl -- run this AFTER check_codegen_1.jl
# has finished and exited (as a separate `julia` invocation, no
# --code-coverage flag needed here -- we're just reading text
# files off disk).
#
# Finds the STADE.jl.<pid>.cov file that check_codegen_1.jl's
# atexit hook wrote, and extracts only the lines with hit count
# > 0 into "stade_calls.log". Grep that file for "function " to
# see which functions were entered.
#
# Run with:
#     julia summarize_coverage.jl
# ============================================================
using Printf

function summarize_coverage(src_path::String, out_path::String)
    dir = dirname(abspath(src_path))
    base = basename(src_path)
    covfiles = filter(f -> startswith(f, base * ".") && endswith(f, ".cov"), readdir(dir))
    if isempty(covfiles)
        error("""
        no coverage file found for $src_path

        This means check_codegen_1.jl either:
          1. wasn't run with --code-coverage=user, or
          2. didn't exit cleanly (crashed / was killed) before
             its atexit hook could flush the .cov file, or
          3. STADE.jl lives in a different directory than expected.

        Try:
            julia --code-coverage=user check_codegen_1.jl
        and confirm it prints "done -- now run: ..." at the end
        (i.e. it reached the end of the script and exited normally)
        before running this summarizer.
        """)
    end
    # if multiple exist (e.g. from previous runs), use the most recently modified
    covfile = covfiles[argmax(mtime.(joinpath.(dir, covfiles)))]
    covpath = joinpath(dir, covfile)

    cov_lines = readlines(covpath)

    # Julia's .cov format is: optional leading whitespace, then a
    # count field that is either digits (line executed N times) or
    # "-" (non-executable line, e.g. comment/blank), then the
    # original source line. The exact delimiter/padding between the
    # count field and the source text isn't something we hardcode --
    # we just take whatever comes after the digit-run or dash.
    n_hit = 0
    open(out_path, "w") do io
        println(io, "# coverage summary for $src_path")
        println(io, "# source coverage file: $covpath")
        println(io, "# only lines with hit count > 0 are shown below")
        println(io, "#" * "-"^60)
        for line in cov_lines
            m = match(r"^\s*(\d+|-)(.*)$", line)
            m === nothing && continue
            count_str = m.captures[1]
            count_str == "-" && continue
            count = parse(Int, count_str)
            if count > 0
                n_hit += 1
                src_text = lstrip(m.captures[2])
                println(io, @sprintf("%8d  %s", count, src_text))
            end
        end
    end

    if n_hit == 0
        println("WARNING: parsed 0 hit lines. Dumping the first 10 raw lines of ",
                 covpath, " so we can see the actual format:")
        for line in first(cov_lines, 10)
            println(repr(line))  # repr() shows tabs/spaces explicitly
        end
    end

    return out_path
end

out = summarize_coverage(joinpath(@__DIR__, "STADE.jl"), joinpath(@__DIR__, "stade_calls.log"))
println("wrote coverage summary to ", out)