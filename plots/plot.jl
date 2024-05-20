using LaTeXTabulars, LaTeXStrings, CairoMakie, JLD2, Printf

results = load("results/results.jld2","results")

function varcon(n)
    if n < 1000
        @sprintf("%3i ", n)
    elseif n < 1000000'
        @sprintf("%3.0fk", n / 1000)
    else
        @sprintf("%3.0fm", n / 1000000)
    end
end

formatter(row) = [
    row.name,
    varcon(row.nvar),
    varcon(row.ncon),
    0,
    @sprintf("%1.2e", row.statuscpu ? row.tcpu : Inf),
    0,
    @sprintf("%1.2e", row.statusgpu ? row.tgpu : Inf),
]

latex_tabular(
    "results.tex",
    Tabular("|l|c|c|cc|cc|"),
    [
        # Rule(:top),
        Rule(),
        ["\\multirow{2}{*}{\\bf case}", "\\multirow{2}{*}{\\bf nvars}", "\\multirow{2}{*}{\\bf nvars}",
         "\\multicolumn{2}{c|}{\\textbf{GPU}}", "\\multicolumn{2}{c|}{\\textbf{GPU}}"],
        ["\\cline{4-7}", "", "", "iter", "solution time (sec)", "iter", "solution time (sec)"],
        Rule(),
        formatter(results[1]),
        formatter(results[2]),
        formatter(results[3]),
        formatter(results[4]),
        Rule(),
    ]
)


nvars = [d.nvar for d in results]
tgpus = [d.statusgpu ? d.tgpu : NaN for d in results]
tcpus = [d.statuscpu ? d.tcpu : NaN for d in results]


plt = Figure(size = (600, 300))
ax = Axis(
    plt[1,1];
    xscale = log10,
    yscale = log10,
    xlabel = "nvars",
    ylabel = "solution time (sec)",
    xautolimitmargin = (.1, .1),
    yautolimitmargin = (.1, .1)
)

plot!(ax, nvars, tcpus; label = "cpu", marker = :circle)
plot!(ax, nvars, tgpus; label = "gpu", marker = :diamond)

axislegend(ax; position = :lt)

save("results/results.pdf", plt)
