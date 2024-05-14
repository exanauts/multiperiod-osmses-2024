using DelimitedFiles

input = 30
output = 180
filename = "data/case9241/halfhour"
Pd = readdlm("$(filename)_$input.Pd")
Qd = readdlm("$(filename)_$input.Qd")

for i in 1:3
    revPd = Pd[:,end:-1:1]
    revQd = Qd[:,end:-1:1]

    @assert all(revPd[:,end] .== Pd[:,1])
    @assert all(revQd[:,end] .== Qd[:,1])

    global Pd = hcat(Pd, revPd)
    global Qd = hcat(Qd, revQd)
end

Pd = Pd[:,1:output]
Qd = Qd[:,1:output]

writedlm("$(filename)_$output.Pd", Pd)
writedlm("$(filename)_$output.Qd", Qd)
