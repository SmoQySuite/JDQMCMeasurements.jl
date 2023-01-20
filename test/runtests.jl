using JDQMCMeasurements
using Test
using LinearAlgebra
using LatticeUtilities

@testset "JDQMCMeasurements.jl" begin
    
    # calculate exact analytic value for Green's function
    function greens(τ,β,ϵ,U)
        
        gτ = similar(ϵ)
        @. gτ = exp(-τ*ϵ)/(1+exp(-β*ϵ))
        Gτ = U * Diagonal(gτ) * adjoint(U)
        
        return Gτ
    end

    # model parameters
    t = 1.0 # hopping amplitude
    μ = 0.0 # chemical potential
    L = 3 # lattice size
    β = 3.5 # inverse temperature
    Δτ = 0.1 # discretization in imaginary time

    # calculate length of imagninary time axis
    Lτ = round(Int, β/Δτ)

    # construct neighbor table for square lattice
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

    # construct diagonal on-site energy matrix
    V = fill(-μ, N);

    # hamiltonian matrix
    H = K + Diagonal(V);

    # diagonlize Hamiltonian matrix
    ϵ, U = eigen(H);

    # calculate equal-time Green's function
    G00 = greens(0,β,ϵ,U);

    # calculate uneqaul-time Green's function
    Gτ0 = zeros(typeof(t), N, N, Lτ+1);
    Gττ = zeros(typeof(t), N, N, Lτ+1);
    for l in 0:Lτ
        Gτ0[:,:,l+1] = greens(Δτ*l, β, ϵ, U)
        Gττ[:,:,l+1] = G00
    end

    @test measure_n(G00) ≈ 0.5
    @test measure_n(G00, 1, unit_cell) ≈ 0.5
    @test measure_n(G00, 2, unit_cell) ≈ 0.5

    @test measure_double_occ(G00, G00) ≈ 0.25
    @test measure_double_occ(G00, G00, 1, unit_cell) ≈ 0.25
    @test measure_double_occ(G00, G00, 2, unit_cell) ≈ 0.25

    @test measure_N(G00) ≈ N/2
    @test measure_N(G00, 1, unit_cell) ≈ N/4
    @test measure_N(G00, 2, unit_cell) ≈ N/4

    G0 = zeros(Complex{Float64}, lattice.L...);
    greens!(G0, 1, 1, unit_cell, lattice, G00)

    Gτ = zeros(Complex{Float64}, lattice.L..., Lτ+1);
    greens!(Gτ, 1, 1, unit_cell, lattice, Gτ0)

    Gβ = -G0
    Gβ[1,1] += 1
    @test Gτ[:,:,1] ≈ G0
    @test Gτ[:,:,end] ≈ Gβ

    G0 = zeros(Complex{Float64}, lattice.L...);
    greens!(G0, 1, 2, unit_cell, lattice, G00)

    Gτ = zeros(Complex{Float64}, lattice.L..., Lτ+1);
    greens!(Gτ, 1, 2, unit_cell, lattice, Gτ0)

    Gβ = -G0
    @test Gτ[:,:,1] ≈ G0
    @test Gτ[:,:,end] ≈ Gβ

    DD0 = zeros(Complex{Float64}, lattice.L...)
    density_correlation!(DD0, 1, 1, unit_cell, lattice, G00, G00)

    DDτ = zeros(Complex{Float64}, lattice.L..., Lτ+1);
    density_correlation!(DDτ, 1, 1, unit_cell, lattice, Gτ0, Gτ0, Gττ, Gττ)

    @test DDτ[:,:,1] ≈ DD0
    @test DDτ[:,:,1] ≈ DDτ[:,:,end]
    @test DD0[1,1] ≈ 1.5

    DD0 = zeros(Complex{Float64}, lattice.L...)
    density_correlation!(DD0, 1, 1, unit_cell, lattice, G00, G00)
    density_correlation!(DD0, 2, 2, unit_cell, lattice, G00, G00)
    density_correlation!(DD0, 1, 2, unit_cell, lattice, G00, G00)
    density_correlation!(DD0, 2, 1, unit_cell, lattice, G00, G00)
    @test measure_Nsqrd(G00, G00) ≈ real(lattice.N*sum(DD0))

    SzSz0 = zeros(Complex{Float64}, lattice.L...);
    spin_z_correlation!(SzSz0, 1, 1, unit_cell, lattice, G00, G00)

    SzSzτ = zeros(Complex{Float64}, lattice.L..., Lτ+1);
    spin_z_correlation!(SzSzτ, 1, 1, unit_cell, lattice, Gτ0, Gτ0, Gττ, Gττ)

    SySy0 = zeros(Complex{Float64}, lattice.L...);
    spin_y_correlation!(SySy0, 1, 1, unit_cell, lattice, G00, G00)

    SySyτ = zeros(Complex{Float64}, lattice.L..., Lτ+1);
    spin_y_correlation!(SySyτ, 1, 1, unit_cell, lattice, Gτ0, Gτ0)

    SxSx0 = zeros(Complex{Float64}, lattice.L...);
    spin_x_correlation!(SxSx0, 1, 1, unit_cell, lattice, G00, G00)

    SxSxτ = zeros(Complex{Float64}, lattice.L..., Lτ+1);
    spin_x_correlation!(SxSxτ, 1, 1, unit_cell, lattice, Gτ0, Gτ0)

    @test SzSz0 ≈ SySy0
    @test SzSz0 ≈ SxSx0

    @test SzSzτ ≈ SySyτ
    @test SzSzτ ≈ SxSxτ

    BB0 = zeros(Complex{Float64}, lattice.L...)
    bond_correlation!(BB0, bond_1, bond_1, unit_cell, lattice, G00, G00)

    BBτ = zeros(Complex{Float64}, lattice.L..., Lτ+1)
    bond_correlation!(BBτ, bond_1, bond_1, unit_cell, lattice, Gτ0, Gτ0, Gττ, Gττ)

    bond_1s = Bond((1,1),[0,0])
    ΔΔ0 = zeros(Complex{Float64}, lattice.L...)
    pair_correlation!(ΔΔ0, bond_1s, bond_1s, unit_cell, lattice, G00, G00)

    ΔΔτ = zeros(Complex{Float64}, lattice.L..., Lτ+1);
    pair_correlation!(ΔΔτ, bond_1s, bond_1s, unit_cell, lattice, Gτ0, Gτ0)

    χ_p = zeros(Complex{Float64}, lattice.L...)
    ΔΔτ_k = copy(ΔΔτ)
    fourier_transform!(ΔΔτ_k, 1, 1, ndims(ΔΔτ_k), unit_cell, lattice)
    susceptibility!(χ_p, ΔΔτ, Δτ, ndims(ΔΔτ))
end
