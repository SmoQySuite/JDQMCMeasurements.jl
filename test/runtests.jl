using JDQMCMeasurements
using Test
using LinearAlgebra
using LatticeUtilities
using Statistics

@testset "JDQMCMeasurements.jl" begin
    
    # fermi function
    function fermi(ϵ, β)

        return (1-tanh(β/2*ϵ))/2
    end

    # calculate exact analytic value for the retarded imaginary Green's function
    function retarded_greens(τ,β,ϵ,U)
        
        gτ = similar(ϵ)
        @. gτ = inv(exp(τ*ϵ) + exp((τ-β)*ϵ))
        Gτ = U * Diagonal(gτ) * adjoint(U)
        logdetGτ, sgndetGτ = logabsdet(Diagonal(gτ))
        
        return Gτ, logdetGτ, sgndetGτ
    end

    # calculate exact analytic value for the advanced imaginary Green's function
    function advanced_greens(τ,β,ϵ,U)
        
        gτ = similar(ϵ)
        @. gτ = -inv(exp(-τ*ϵ) + exp(-(τ-β)*ϵ))
        Gτ = U * Diagonal(gτ) * adjoint(U)
        logdetGτ, sgndetGτ = logabsdet(Diagonal(gτ))
        
        return Gτ, logdetGτ, sgndetGτ
    end

    # model parameters
    t = 1.0 # hopping amplitude
    μ = 0.0 # chemical potential
    L = 3 # lattice size
    β = 3.5 # inverse temperature
    Δτ = 0.1 # discretization in imaginary time

    # calculate length of imagninary time axis
    Lτ = round(Int, β/Δτ)

    # construct neighbor table for honeycomb lattice
    unit_cell = UnitCell(lattice_vecs = [[3/2,√3/2],[3/2,-√3/2]],
                         basis_vecs = [[0.,0.],[1.,0.]])
    lattice = Lattice(L = [L,L], periodic = [true,true])
    bond_1 = Bond((1,2), [0,0])
    bond_2 = Bond((1,2), [-1,0])
    bond_3 = Bond((1,2), [0,-1])
    neighbor_table = build_neighbor_table([bond_1, bond_2, bond_3], unit_cell, lattice)
    N = nsites(unit_cell, lattice)

    # construct hopping matrix
    K = zeros(typeof(t), N, N)
    for n in axes(neighbor_table, 2)
        j = neighbor_table[1,n]
        i = neighbor_table[2,n]
        K[i,j] = -t
        K[j,i] = -conj(t)
    end

    # arrays of hopping amplitudes
    t1 = fill(t, lattice.L...)
    t2 = fill(t, lattice.L...)
    t3 = fill(t, lattice.L...)

    # construct diagonal on-site energy matrix
    V = fill(-μ, N)

    # hamiltonian matrix
    H = K + Diagonal(V)

    # diagonlize Hamiltonian matrix
    ϵ, U = eigen(H)

    # calculate equal-time Green's function
    G00 = retarded_greens(0.0, β, ϵ, U)[1]
    Gττ = G00

    # check that the total particle number ⟨N⟩ is measured correctly
    @test measure_N(G00) ≈ N/2
    @test measure_N(G00, 1, unit_cell) ≈ N/4
    @test measure_N(G00, 2, unit_cell) ≈ N/4

    # calculate density correlation at τ = 0
    DD0 = zeros(Complex{Float64}, lattice.L...)
    Gτ0 = retarded_greens(0.0, β, ϵ, U)[1]
    G0τ = advanced_greens(0.0, β, ϵ, U)[1]
    density_correlation!(DD0, 1, 1, unit_cell, lattice, Gτ0, G0τ, Gττ, G00, Gτ0, G0τ, Gττ, G00, 1.0)

    # calculate density correlation at τ = β
    DDβ = zeros(Complex{Float64}, lattice.L...)
    Gτ0 = retarded_greens(β, β, ϵ, U)[1]
    G0τ = advanced_greens(β, β, ϵ, U)[1]
    density_correlation!(DDβ, 1, 1, unit_cell, lattice, Gτ0, G0τ, Gττ, G00, Gτ0, G0τ, Gττ, G00, 1.0)

    # make sure density correlation D(0,r) ≈ D(β,r)
    @test DD0 ≈ DDβ

    # # make on-site density correlation is correct
    @test DD0[1,1] ≈ 1.5

    # make sure two methods for calculating ⟨N²⟩ agree
    DD0 = zeros(Complex{Float64}, lattice.L...)
    Gτ0 = retarded_greens(0.0, β, ϵ, U)[1]
    G0τ = advanced_greens(0.0, β, ϵ, U)[1]
    density_correlation!(DD0, 1, 1, unit_cell, lattice, Gτ0, G0τ, Gττ, G00, Gτ0, G0τ, Gττ, G00, 1.0)
    density_correlation!(DD0, 2, 2, unit_cell, lattice, Gτ0, G0τ, Gττ, G00, Gτ0, G0τ, Gττ, G00, 1.0)
    density_correlation!(DD0, 1, 2, unit_cell, lattice, Gτ0, G0τ, Gττ, G00, Gτ0, G0τ, Gττ, G00, 1.0)
    density_correlation!(DD0, 2, 1, unit_cell, lattice, Gτ0, G0τ, Gττ, G00, Gτ0, G0τ, Gττ, G00, 1.0)
    @test measure_Nsqrd(G00, G00) ≈ real(lattice.N*sum(DD0))

    # initialize correlation containers
    G = zeros(Complex{Float64}, lattice.L...)
    DD = zeros(Complex{Float64}, lattice.L...)
    PP = zeros(Complex{Float64}, lattice.L...)
    SzSz = zeros(Complex{Float64}, lattice.L...)
    SxSx = zeros(Complex{Float64}, lattice.L...)
    BB = zeros(Complex{Float64}, lattice.L...)
    CC = zeros(Complex{Float64}, lattice.L...)

    # iterate of possible imaginary time displacements
    @testset for l in 0:Lτ

        # calculate the imaginary time displacement
        τ = Δτ * l

        # calculate analytic time-displaced green's function matrices
        Gτ0 = retarded_greens(τ, β, ϵ, U)[1]
        G0τ = advanced_greens(τ, β, ϵ, U)[1]

        # measure greens function
        fill!(G, 0)
        greens!(G, 1, 1, unit_cell, lattice, Gτ0, 1.0)

        # make sure spin-x and spin-y measurements agree
        fill!(SzSz, 0)
        spin_z_correlation!(SzSz, 1, 1, unit_cell, lattice, Gτ0, G0τ, Gττ, G00, Gτ0, G0τ, Gττ, G00, 1.0)
        fill!(SxSx, 0)
        spin_x_correlation!(SxSx, 1, 1, unit_cell, lattice, Gτ0, G0τ, Gτ0, G0τ, 1.0)
        @test SzSz ≈ SxSx
        fill!(SzSz, 0)
        spin_z_correlation!(SzSz, 2, 2, unit_cell, lattice, Gτ0, G0τ, Gττ, G00, Gτ0, G0τ, Gττ, G00, 1.0)
        fill!(SxSx, 0)
        spin_x_correlation!(SxSx, 2, 2, unit_cell, lattice, Gτ0, G0τ, Gτ0, G0τ, 1.0)
        @test SzSz ≈ SxSx
        fill!(SzSz, 0)
        spin_z_correlation!(SzSz, 1, 2, unit_cell, lattice, Gτ0, G0τ, Gττ, G00, Gτ0, G0τ, Gττ, G00, 1.0)
        fill!(SxSx, 0)
        spin_x_correlation!(SxSx, 1, 2, unit_cell, lattice, Gτ0, G0τ, Gτ0, G0τ, 1.0)
        @test SzSz ≈ SxSx
        fill!(SzSz, 0)
        spin_z_correlation!(SzSz, 2, 1, unit_cell, lattice, Gτ0, G0τ, Gττ, G00, Gτ0, G0τ, Gττ, G00, 1.0)
        fill!(SxSx, 0)
        spin_x_correlation!(SxSx, 2, 1, unit_cell, lattice, Gτ0, G0τ, Gτ0, G0τ, 1.0)
        @test SzSz ≈ SxSx

        # measure bond correlation function
        fill!(BB, 0)
        bond_correlation!(BB, bond_1, bond_1, unit_cell, lattice, Gτ0, G0τ, Gττ, G00, Gτ0, G0τ, Gττ, G00, 1.0)

        # measure pair correlation function
        fill!(PP, 0)
        pair_correlation!(PP, bond_1, bond_1, unit_cell, lattice, Gτ0, Gτ0, 1.0)

        # measure density correlation function
        fill!(DD, 0)
        density_correlation!(DD, 1, 1, unit_cell, lattice, Gτ0, G0τ, Gττ, G00, Gτ0, G0τ, Gττ, G00, 1.0)

        # measure current correlation function
        fill!(CC, 0)
        current_correlation!(CC, bond_1, bond_1, t1, t1, t1, t1, unit_cell, lattice, Gτ0, G0τ, Gττ, G00, Gτ0, G0τ, Gττ, G00, 1.0)
    end

    # run a simple test for jackknife verifying that is reproduces the mean and
    # standard deviation of the mean correctly
    r = rand(100)
    r̄ = mean(r)
    Δr = std(r)/sqrt(length(r))
    r̄_jackknife, Δr_jackknife = jackknife(identity, r)
    @test r̄ ≈ r̄_jackknife
    @test Δr ≈ Δr_jackknife
end
