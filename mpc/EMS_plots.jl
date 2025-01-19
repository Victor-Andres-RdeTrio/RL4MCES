using CairoMakie, Makie, GLMakie

makeEMSplots(rhDict, EMSData; backend="GLMakie")
makeEMSplots(rhDict, EMSData; backend="CairoMakie")
makeEBplot(rhDict, EMSData)
compareEB(results, EMSData)