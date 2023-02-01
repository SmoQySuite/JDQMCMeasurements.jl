var documenterSearchIndex = {"docs":
[{"location":"api/#API","page":"API","title":"API","text":"","category":"section"},{"location":"api/#Correlation-Measurements","page":"API","title":"Correlation Measurements","text":"","category":"section"},{"location":"api/","page":"API","title":"API","text":"greens!\ndensity_correlation!\nspin_x_correlation!\nspin_z_correlation!\npair_correlation!\nbond_correlation!","category":"page"},{"location":"api/","page":"API","title":"API","text":"greens!\ndensity_correlation!\nspin_x_correlation!\nspin_z_correlation!\npair_correlation!\nbond_correlation!","category":"page"},{"location":"api/#JDQMCMeasurements.greens!","page":"API","title":"JDQMCMeasurements.greens!","text":"greens!(G::AbstractArray{C,D}, a::Int, b::Int, \n        unit_cell::UnitCell{D}, lattice::Lattice{D},\n        G_τ0::AbstractMatrix{T}, sgn::T=one(T)) where {D, C<:Number, T<:Number}\n\nMeasure the unequal time Green's function averaged over translation symmetry\n\nG_sigmamathbfr^ab(tau)=frac1Nsum_mathbfiG_sigmamathbfi+mathbfrmathbfi^ab(tau0)\n=frac1Nsum_mathbfilanglehatmathcalThata_sigmamathbfi+mathbfr^phantomdagger(tau)hatb_sigmamathbfi^dagger(0)rangle\n\nwith the result being added to G.\n\nFields\n\nG::AbstractArray{C,D}: Array the green's function G_sigmamathbfr^ab(tau) is written to.\na::Int: Index specifying an orbital species in the unit cell.\nb::Int: Index specifying an orbital species in the unit cell.\nunit_cell::UnitCell{D}: Defines unit cell.\nlattice::Lattice{D}: Specifies size of finite lattice.\nG_τ0::AbstractMatrix{T}: The matrix G(tau0)\nsgn::T=one(T): The sign of the weight appearing in a DQMC simulation.\n\n\n\n\n\n","category":"function"},{"location":"api/#JDQMCMeasurements.density_correlation!","page":"API","title":"JDQMCMeasurements.density_correlation!","text":"density_correlation!(DD::AbstractArray{C,D}, a::Int, b::Int, unit_cell::UnitCell{D}, lattice::Lattice{D},\n                     Gup_τ0::AbstractMatrix{T}, Gup_0τ::AbstractMatrix{T}, Gup_ττ::AbstractMatrix{T}, Gup_00::AbstractMatrix{T},\n                     Gdn_τ0::AbstractMatrix{T}, Gdn_0τ::AbstractMatrix{T}, Gdn_ττ::AbstractMatrix{T}, Gdn_00::AbstractMatrix{T},\n                     sgn::T=one(T)) where {D, C<:Number, T<:Number}\n\nCalculate the unequal-time density-density (charge) correlation function\n\nmathcalD_mathbfr^ab(tau) = frac1Nsum_mathbfimathcalD_mathbfi+mathbfrmathbfi^ab(tau0)\n= frac1Nsum_mathbfilanglehatn_amathbfi+mathbfr(tau)hatn_bmathbfi(0)rangle\n\nwhere hatn_bmathbfi = (hatn_uparrow b mathbfi + hatn_downarrow b mathbfi) and hatn_sigma bmathbfi = hatb^dagger_sigma mathbfi hatb_sigma mathbfi is the number operator for an electron with spin sigma on orbital b in unit cell mathbfi, with the result being added to the array DD.\n\nFields\n\nDD::AbstractArray{C,D}: Array the density correlation function mathcalD_mathbfr^ab(tau) is added to.\na::Int: Index specifying an orbital species in the unit cell.\nb::Int: Index specifying an orbital species in the unit cell.\nunit_cell::UnitCell{D}: Defines unit cell.\nlattice::Lattice{D}: Specifies size of finite lattice.\nGup_τ0::AbstractMatrix{T}: The matrix G_uparrow(tau0)\nGup_0τ::AbstractMatrix{T}: The matrix G_uparrow(0tau)\nGup_ττ::AbstractMatrix{T}: The matrix G_uparrow(tautau)\nGup_00::AbstractMatrix{T}: The matrix G_uparrow(00)\nGdn_τ0::AbstractMatrix{T}: The matrix G_downarrow(tau0)\nGdn_0τ::AbstractMatrix{T}: The matrix G_downarrow(0tau)\nGdn_ττ::AbstractMatrix{T}: The matrix G_downarrow(tautau)\nGdn_00::AbstractMatrix{T}: The matrix G_downarrow(00)\nsgn::T=one(T): The sign of the weight appearing in a DQMC simulation.\n\n\n\n\n\n","category":"function"},{"location":"api/#JDQMCMeasurements.spin_x_correlation!","page":"API","title":"JDQMCMeasurements.spin_x_correlation!","text":"spin_x_correlation!(SxSx::AbstractArray{C}, a::Int, b::Int,\n                    unit_cell::UnitCell{D}, lattice::Lattice{D},\n                    Gτ0up::AbstractArray{T,3}, Gτ0dn::AbstractArray{T,3},\n                    sgn::T=one(T)) where {D, C<:Complex, T<:Number}\n\nCalculate the unequal-time spin-spin correlation function in the hatx direction, given by\n\nmathcalS_xmathbfr^ab(tau)=frac1Nsum_mathbfimathcalS_xmathbfi+mathbfrmathbfi^ab(tau0)\n=frac1Nsum_mathbfibiglanglehatS_xamathbfi+mathbfr(tau)hatS_xbmathbfi(0)bigrangle\n\nwhere the spin-hatx operator is given by\n\nbeginalign*\nhatS_xmathbfia=  (hata_uparrowmathbfi^daggerhata_downarrowmathbfi^dagger)leftbeginarraycc\n0  1\n1  0\nendarrayrightleft(beginarrayc\nhata_uparrowmathbfi\nhata_downarrowmathbfi\nendarrayright)\n=  hata_uparrowmathbfi^daggerhata_downarrowmathbfi+hata_downarrowmathbfi^daggerhata_uparrowmathbfi\nendalign*\n\nFields\n\nSxSx::AbstractArray{C,D}: Array the spin-x correlation function mathcalS_xmathbfr^ab(tau) is added to.\na::Int: Index specifying an orbital species in the unit cell.\nb::Int: Index specifying an orbital species in the unit cell.\nunit_cell::UnitCell{D}: Defines unit cell.\nlattice::Lattice{D}: Specifies size of finite lattice.\nGup_τ0::AbstractMatrix{T}: The matrix G_uparrow(tau0)\nGup_0τ::AbstractMatrix{T}: The matrix G_uparrow(0tau)\nGdn_τ0::AbstractMatrix{T}: The matrix G_downarrow(tau0)\nGdn_0τ::AbstractMatrix{T}: The matrix G_downarrow(0tau)\nsgn::T=one(T): The sign of the weight appearing in a DQMC simulation.\n\n```\n\n\n\n\n\n","category":"function"},{"location":"api/#JDQMCMeasurements.spin_z_correlation!","page":"API","title":"JDQMCMeasurements.spin_z_correlation!","text":"spin_z_correlation!(SzSz::AbstractArray{C,D}, a::Int, b::Int, unit_cell::UnitCell{D}, lattice::Lattice{D},\n                    Gup_τ0::AbstractMatrix{T}, Gup_0τ::AbstractMatrix{T}, Gup_ττ::AbstractMatrix{T}, Gup_00::AbstractMatrix{T},\n                    Gdn_τ0::AbstractMatrix{T}, Gdn_0τ::AbstractMatrix{T}, Gdn_ττ::AbstractMatrix{T}, Gdn_00::AbstractMatrix{T},\n                    sgn::T=one(T)) where {D, C<:Complex, T<:Number}\n\nCalculate the unequal-time spin-spin correlation function in the hatz direction, given by\n\nmathcalS_zmathbfr^ab(tau)=frac1Nsum_mathbfimathcalS_zmathbfi+mathbfrmathbfi^ab(tau0)\n=frac1Nsum_mathbfibiglanglehatS_zamathbfi+mathbfr(tau)hatS_zbmathbfi(0)bigrangle\n\nwhere the spin-hatz operator is given by\n\nbeginalign*\nhatS_zamathbfi=  (hata_uparrowmathbfi^daggerhata_downarrowmathbfi^dagger)leftbeginarraycc\n1  0\n0  -1\nendarrayrightleft(beginarrayc\nhata_uparrowmathbfi\nhata_downarrowmathbfi\nendarrayright)\n=  hatn_uparrowamathbfi-hatn_downarrowamathbfi\nendalign*\n\nFields\n\nSzSz::AbstractArray{C,D}: Array the spin-z correlation function mathcalS_zmathbfr^ab(tau) is added to.\na::Int: Index specifying an orbital species in the unit cell.\nb::Int: Index specifying an orbital species in the unit cell.\nunit_cell::UnitCell{D}: Defines unit cell.\nlattice::Lattice{D}: Specifies size of finite lattice.\nGup_τ0::AbstractMatrix{T}: The matrix G_uparrow(tau0)\nGup_0τ::AbstractMatrix{T}: The matrix G_uparrow(0tau)\nGup_ττ::AbstractMatrix{T}: The matrix G_uparrow(tautau)\nGup_00::AbstractMatrix{T}: The matrix G_uparrow(00)\nGdn_τ0::AbstractMatrix{T}: The matrix G_downarrow(tau0)\nGdn_0τ::AbstractMatrix{T}: The matrix G_downarrow(0tau)\nGdn_ττ::AbstractMatrix{T}: The matrix G_downarrow(tautau)\nGdn_00::AbstractMatrix{T}: The matrix G_downarrow(00)\nsgn::T=one(T): The sign of the weight appearing in a DQMC simulation.\n\n\n\n\n\n","category":"function"},{"location":"api/#JDQMCMeasurements.pair_correlation!","page":"API","title":"JDQMCMeasurements.pair_correlation!","text":"pair_correlation!(P::AbstractArray{C,D}, b″::Bond{D}, b′::Bond{D}, unit_cell::UnitCell{D}, lattice::Lattice{D},\n                  Gup_τ0::AbstractMatrix{T}, Gdn_τ0::AbstractMatrix{T}, sgn::T=one(T)) where {D, C<:Number, T<:Number}\n\nCalculate the unequal-time pair correlation function\n\nmathcalP_mathbfr^(abr)(cdr)(tau)=frac1Nsum_mathbfimathcalP_mathbfi+mathbfrmathbfi^(abr)(cdr)(tau0)\n=frac1Nsum_mathbfilanglehatDelta_mathbfi+mathbfrabmathbfr(tau)hatDelta_mathbficdmathbfr^dagger(0)rangle\n\nwhere the bond b″ defines the pair creation operator\n\nhatDelta_mathbfiabmathbfr^dagger=hata_uparrowmathbfi+mathbfr^daggerhatb_downarrowmathbfi^dagger\n\nand the bond  b′ defines the pair creation operator\n\nhatDelta_mathbficdmathbfr^dagger=hatc_uparrowmathbfi+mathbfr^daggerhatd_downarrowmathbfi^dagger\n\nFields\n\nP::AbstractArray{C,D}: Array the pair correlation function mathcalP_mathbfr^(abr)(cdr)(tau) is added to.\nb″::Bond{D}: Bond defining pair annihilation operator appearing in pair correlation function.\nb′::Bond{D}: Bond defining pair creation operator appearing in pair correlation function.\nunit_cell::UnitCell{D}: Defines unit cell.\nlattice::Lattice{D}: Specifies size of finite lattice.\nGup_τ0::AbstractMatrix{T}: The matrix G_uparrow(tau0)\nGdn_τ0::AbstractMatrix{T}: The matrix G_downarrow(tau0)\nsgn::T=one(T): The sign of the weight appearing in a DQMC simulation.\n\n\n\n\n\n","category":"function"},{"location":"api/#JDQMCMeasurements.bond_correlation!","page":"API","title":"JDQMCMeasurements.bond_correlation!","text":"bond_correlation!(BB::AbstractArray{C,D}, b′::Bond{D}, b″::Bond{D}, unit_cell::UnitCell{D}, lattice::Lattice{D},\n                  Gup_τ0::AbstractMatrix{T}, Gup_0τ::AbstractMatrix{T}, Gup_ττ::AbstractMatrix{T}, Gup_00::AbstractMatrix{T},\n                  Gdn_τ0::AbstractMatrix{T}, Gdn_0τ::AbstractMatrix{T}, Gdn_ττ::AbstractMatrix{T}, Gdn_00::AbstractMatrix{T},\n                  sgn::T=one(T)) where {D, C<:Number, T<:Number}\n\nCalculate the uneqaul-time bond-bond correlation function\n\nmathcalB_mathbfr^(mathbfrab)(mathbfrcd)(tau) =\n    frac1Nsum_mathbfi langlehatB_uparrowmathbfi+mathbfr(mathbfrab)(tau)+hatB_downarrowmathbfi+mathbfr(mathbfrab)(tau)\n                                   cdothatB_uparrowmathbfi(mathbfrcd)(0)+hatB_downarrowmathbfi(mathbfrcd)(0)rangle\n\nwhere the\n\nhatB_sigmamathbfi(mathbfrab) = hata_sigmamathbfi+mathbfr^daggerhatb_sigmamathbfi^phantomdagger\n                                             + hatb_sigmamathbfi^daggerhata_sigmamathbfi+mathbfr^phantomdagger\n\nis the bond operator.\n\nFields\n\nBB::AbstractArray{C,D}: Array the bond correlation function mathcalB_mathbfr^(mathbfrab)(mathbfrcd)(tau) is added to.\nb′::Bond{D}: Bond defining the bond operator appearing on the left side of the bond correlation function.\nb″::Bond{D}: Bond defining the bond operator appearing on the right side of the bond correlation function.\nunit_cell::UnitCell{D}: Defines unit cell.\nlattice::Lattice{D}: Specifies size of finite lattice.\nGup_τ0::AbstractMatrix{T}: The matrix G_uparrow(tau0)\nGup_0τ::AbstractMatrix{T}: The matrix G_uparrow(0tau)\nGup_ττ::AbstractMatrix{T}: The matrix G_uparrow(tautau)\nGup_00::AbstractMatrix{T}: The matrix G_uparrow(00)\nGdn_τ0::AbstractMatrix{T}: The matrix G_downarrow(tau0)\nGdn_0τ::AbstractMatrix{T}: The matrix G_downarrow(0tau)\nGdn_ττ::AbstractMatrix{T}: The matrix G_downarrow(tautau)\nGdn_00::AbstractMatrix{T}: The matrix G_downarrow(00)\nsgn::T=one(T): The sign of the weight appearing in a DQMC simulation.\n\n\n\n\n\n","category":"function"},{"location":"api/#Scalar-Measurements","page":"API","title":"Scalar Measurements","text":"","category":"section"},{"location":"api/","page":"API","title":"API","text":"measure_n\nmeasure_double_occ\nmeasure_N\nmeasure_Nsqrd","category":"page"},{"location":"api/","page":"API","title":"API","text":"measure_n\nmeasure_double_occ\nmeasure_N\nmeasure_Nsqrd","category":"page"},{"location":"api/#JDQMCMeasurements.measure_n","page":"API","title":"JDQMCMeasurements.measure_n","text":"measure_n(G::AbstractMatrix{T}) where {T}\n\nMeasure the average density langle hatn_sigma rangle given the equal-time Green's function matrix G_sigma(tautau)\n\n\n\n\n\nmeasure_n(G::AbstractMatrix{T}, a::Int, unit_cell::UnitCell) where {T}\n\nMeasure the average density langle hatn_sigmaa rangle for orbital species a given the equal-time Green's function matrix G_sigma(tautau)\n\n\n\n\n\n","category":"function"},{"location":"api/#JDQMCMeasurements.measure_double_occ","page":"API","title":"JDQMCMeasurements.measure_double_occ","text":"measure_double_occ(Gup::AbstractMatrix{T}, Gdn::AbstractMatrix{T}) where {T}\n\nMeasure the double-occupancy langle hatn_uparrow hatn_downarrow rangle given both the spin-up and spin-down equal-time Green's function matrices G_uparrow(tautau) and G_downarrow(tautau) respectively.\n\n\n\n\n\nmeasure_double_occ(Gup::AbstractMatrix{T}, Gdn::AbstractMatrix{T}, a::Int, unit_cell::UnitCell) where {T}\n\nMeasure the double-occupancy langle hatn_uparrowa hatn_downarrowa rangle for orbital species a given both the spin-up and spin-down equal-time Green's function matrices G_uparrow(tautau) and G_downarrow(tautau) respectively.\n\n\n\n\n\n","category":"function"},{"location":"api/#JDQMCMeasurements.measure_N","page":"API","title":"JDQMCMeasurements.measure_N","text":"measure_N(G::AbstractMatrix{T}) where {T}\n\nMeasure the total particle number langle hatN_sigma rangle given an equal-time Green's function matrix G_sigma(tautau)\n\n\n\n\n\nmeasure_N(G::AbstractMatrix{T}, a::Int, unit_cell::UnitCell) where {T}\n\nMeasure the total particle number langle hatN_sigmaa rangle in orbital species a  given an equal-time Green's function matrix G_sigma(tautau)\n\n\n\n\n\n","category":"function"},{"location":"api/#JDQMCMeasurements.measure_Nsqrd","page":"API","title":"JDQMCMeasurements.measure_Nsqrd","text":"measure_Nsqrd(Gup::AbstractMatrix{T}, Gdn::AbstractMatrix{T}) where {T}\n\nMeasure the expectation value of the total particle number squared langle hatN^2 rangle given both the spin-up and spin-down equal-time Green's function matrices G_uparrow(tautau) and G_downarrow(tautau) respectively.\n\n\n\n\n\n","category":"function"},{"location":"api/#Utility-Functions","page":"API","title":"Utility Functions","text":"","category":"section"},{"location":"api/","page":"API","title":"API","text":"fourier_transform!\nsusceptibility!\nsusceptibility","category":"page"},{"location":"api/","page":"API","title":"API","text":"fourier_transform!\nsusceptibility!\nsusceptibility","category":"page"},{"location":"api/#JDQMCMeasurements.fourier_transform!","page":"API","title":"JDQMCMeasurements.fourier_transform!","text":"fourier_transform!(C::AbstractArray{Complex{T}}, a::Int, b::Int, dim::Int,\n                   unit_cell::UnitCell{D,T}, lattice::Lattice{D}) where {D, T<:AbstractFloat}\n\nfourier_transform!(C::AbstractArray{Complex{T}}, a::Int, b::Int,\n                   unit_cell::UnitCell{D,T}, lattice::Lattice{D}) where {D, T<:AbstractFloat}\n\nCalculate the fourier transform from position to momentum space\n\nbeginalign*\nC_mathbfk^ab=  sum_mathbfre^rm -imathbfkcdot(mathbfr+mathbfr_a-mathbfr_b)C_mathbfr^ab\nendalign*\n\nwhere a and b specify orbital species in the unit cell. Note that the array C is modified in-place. If dim is passed, iterate over this dimension of the array, performing a fourier transform on each slice.\n\n\n\n\n\n","category":"function"},{"location":"api/#JDQMCMeasurements.susceptibility!","page":"API","title":"JDQMCMeasurements.susceptibility!","text":"susceptibility!(χ::AbstractArray{T}, S::AbstractArray{T}, Δτ::E, dim::Int) where {T<:Number, E<:AbstractFloat}\n\nCalculate the susceptibilities\n\nchi_mathbfn = int_0^beta S_mathbfn(tau) dtau\n\nwhere the chi_mathbfn susceptibilities are written to χ, and S contains the S_mathbfn(tau) correlations that need to be integrated over. The parameter Δτ is the discretization in imaginary time tau and is the step size used in Simpson's method to numerically evaluate the integral over imaginary time. The argument dim specifies which dimension of S corresponds to imaginary time, and needs to be integrated over. Accordingly,\n\nndim(χ)+1 == ndim(S)\n\nand\n\nsize(χ) == size(selectdim(S, dim, 1))\n\nmust both be true.\n\n\n\n\n\n","category":"function"},{"location":"api/#JDQMCMeasurements.susceptibility","page":"API","title":"JDQMCMeasurements.susceptibility","text":"susceptibility(S::AbstractVector{T}, Δτ::E) where {T<:Number, E<:AbstractFloat}\n\nCalculate the suceptibility\n\nchi = int_0^beta S(tau) dtau\n\nwhere the correlation data is stored in S. The integration is performed using Simpson's method using a step size of Δτ.\n\n\n\n\n\n","category":"function"},{"location":"api/#Developer-API","page":"API","title":"Developer API","text":"","category":"section"},{"location":"api/","page":"API","title":"API","text":"JDQMCMeasurements.contract_Gr0!\nJDQMCMeasurements.contract_G00!\nJDQMCMeasurements.contract_δGr0!\nJDQMCMeasurements.contract_Grr_G00!\nJDQMCMeasurements.contract_G00_Grr!\nJDQMCMeasurements.contract_Gr0_Gr0!\nJDQMCMeasurements.contract_G0r_G0r!\nJDQMCMeasurements.contract_G0r_Gr0!\nJDQMCMeasurements.contract_Gr0_G0r!\nJDQMCMeasurements.simpson","category":"page"},{"location":"api/","page":"API","title":"API","text":"JDQMCMeasurements.contract_Gr0!\nJDQMCMeasurements.contract_G00!\nJDQMCMeasurements.contract_δGr0!\nJDQMCMeasurements.contract_Grr_G00!\nJDQMCMeasurements.contract_G00_Grr!\nJDQMCMeasurements.contract_Gr0_Gr0!\nJDQMCMeasurements.contract_G0r_G0r!\nJDQMCMeasurements.contract_G0r_Gr0!\nJDQMCMeasurements.contract_Gr0_G0r!\nJDQMCMeasurements.simpson","category":"page"},{"location":"api/#JDQMCMeasurements.contract_Gr0!","page":"API","title":"JDQMCMeasurements.contract_Gr0!","text":"contract_Gr0!(S::AbstractArray{C}, G::AbstractMatrix{T}, r′::Bond, α::Int,\n              unit_cell::UnitCell{D,E}, lattice::Lattice{D},\n              sgn::T=one(T)) where {D, C<:Complex, T<:Number, E<:AbstractFloat}\n\nEvaluate the sum\n\nS_mathbfr=S_mathbfr+fracalphaNsum_mathbfiG_sigmamathbfi+mathbfr+mathbfr_1mathbfi^ab(tau0)\n\nfor all mathbfr where the bond r′ represents the static displacement mathbfr_1+(mathbfr_a-mathbfr_b)\n\n\n\n\n\n","category":"function"},{"location":"api/#JDQMCMeasurements.contract_G00!","page":"API","title":"JDQMCMeasurements.contract_G00!","text":"contract_G00!(S::AbstractArray{C}, G::AbstractMatrix{T}, a::Int, b::Int, α::Int,\n              unit_cell::UnitCell{D,E}, lattice::Lattice{D},\n              sgn::T=one(T)) where {D, C<:Complex, T<:Number, E<:AbstractFloat}\n\nEvaluate the sum\n\nbeginalign*\nS_mathbfr = S_mathbfr + fracalphaNsum_mathbfiG_sigmamathbfimathbfi^ab(tau0)\nendalign*\n\nfor all mathbfr\n\n\n\n\n\n","category":"function"},{"location":"api/#JDQMCMeasurements.contract_δGr0!","page":"API","title":"JDQMCMeasurements.contract_δGr0!","text":"contract_δGr0!(S::AbstractArray{C}, G::AbstractMatrix{T}, δ::Bond, α::Int,\n               unit_cell::UnitCell{D,E}, lattice::Lattice{D},\n               sgn::T=one(T)) where {D, C<:Complex, T<:Number, E<:AbstractFloat}\n\nEvaluate the sum\n\nS_mathbfr=S_mathbfr+fracalphaNsum_mathbfidelta_abdelta_mathbfrmathbfr_2G_sigmamathbfi+mathbfr+mathbfr_1mathbfi^cd(tau0)\n\nfor all mathbfr where the bond δ represents the static displacement mathbfr_2+(mathbfr_a-mathbfr_b) and the bond r′ represents the static displacement mathbfr_1+(mathbfr_c-mathbfr_d)\n\n\n\n\n\n","category":"function"},{"location":"api/#JDQMCMeasurements.contract_Grr_G00!","page":"API","title":"JDQMCMeasurements.contract_Grr_G00!","text":"contract_Grr_G00!(S::AbstractArray{C}, G₂::AbstractMatrix{T}, G₁::AbstractMatrix{T}, b₂::Bond{D}, b₁::Bond{D},\n                  α::Int, unit_cell::UnitCell{D,E}, lattice::Lattice{D},\n                  sgn::T=one(T)) where {D, C<:Complex, T<:Number, E<:AbstractFloat}\n\nEvaluate the sum\n\nS_mathbfr=S_mathbfr+fracalphaNsum_mathbfiG_sigma_2mathbfi+mathbfr+mathbfr_2mathbfi+mathbfr^ab(tau_20)cdot G_sigma_1mathbfi+mathbfr_1mathbfi^cd(tau_10)\n\nfor all mathbfr where the bond b₂ represents the static displacement mathbfr_2 + (mathbfr_a - mathbfr_b) and the bond b₁ represents the static displacement mathbfr_1 + (mathbfr_c - mathbfr_d)\n\n\n\n\n\ncontract_Grr_G00!(S::AbstractArray{C}, G₂::AbstractMatrix{T}, G₁::AbstractMatrix{T},\n                  a::Int, b::Int, c::Int, d::Int,\n                  r₄::AbstractVector{Int}, r₃::AbstractVector{Int}, r₂::AbstractVector{Int}, r₁::AbstractVector{Int},\n                  α::Int, unit_cell::UnitCell{D,E}, lattice::Lattice{D},\n                  sgn::T=one(T)) where {D, C<:Complex, T<:Number, E<:AbstractFloat}\n\nEvaluate the sum\n\nS_mathbfr=S_mathbfr+fracalphaNsum_mathbfiG_sigma_2mathbfi+mathbfr+mathbfr_4mathbfi+mathbfr+mathbfr_3^ab(tau_20)cdot G_sigma_1mathbfi+mathbfr_2mathbfi+mathbfr_1^cd(tau_10)\n\nfor all mathbfr\n\n\n\n\n\n","category":"function"},{"location":"api/#JDQMCMeasurements.contract_G00_Grr!","page":"API","title":"JDQMCMeasurements.contract_G00_Grr!","text":"contract_G00_Grr!(S::AbstractArray{C}, G₂::AbstractMatrix{T}, G₁::AbstractMatrix{T},\n                  a::Int, b::Int, c::Int, d::Int,\n                  r₄::AbstractVector{Int}, r₃::AbstractVector{Int}, r₂::AbstractVector{Int}, r₁::AbstractVector{Int},\n                  α::Int, unit_cell::UnitCell{D,E}, lattice::Lattice,\n                  sgn::T=one(T)) where {D, C<:Complex, T<:Number, E<:AbstractFloat}\n\nEvaluate the sum\n\nS_mathbfr=S_mathbfr+fracalphaNsum_mathbfiG_sigma_2mathbfi+mathbfr_4mathbfi+mathbfr_3^ab(tau_20)cdot G_sigma_1mathbfi+mathbfr+mathbfr_2mathbfi+mathbfr+mathbfr_1^cd(tau_10)\n\nfor all mathbfr\n\n\n\n\n\n","category":"function"},{"location":"api/#JDQMCMeasurements.contract_Gr0_Gr0!","page":"API","title":"JDQMCMeasurements.contract_Gr0_Gr0!","text":"contract_Gr0_Gr0!(S::AbstractArray{C}, G₂::AbstractMatrix{T}, G₁::AbstractMatrix{T}, b₂::Bond, b₁::Bond,\n                  α::Int, unit_cell::UnitCell{D,E}, lattice::Lattice{D},\n                  sgn::T=one(T)) where {D, C<:Complex, T<:Number, E<:AbstractFloat}\n\nEvaluate the sum\n\nS_mathbfr=S_mathbfr+fracalphaNsum_mathbfiG_sigma_2mathbfi+mathbfr+mathbfr_2mathbfi+mathbfr_1^ac(tau_20)cdot G_sigma_1mathbfi+mathbfrmathbfi^bd(tau_10)\n\nfor all mathbfr where the bond b₂ represents the static displacement mathbfr_2 + (mathbfr_a - mathbfr_b) and the bond b₁ represents the static displacement mathbfr_1 + (mathbfr_c - mathbfr_d)\n\n\n\n\n\ncontract_Gr0_Gr0!(S::AbstractArray{C}, G₂::AbstractMatrix{T}, G₁::AbstractMatrix{T},\n                  a::Int, b::Int, c::Int, d::Int,\n                  r₄::AbstractVector{Int}, r₃::AbstractVector{Int}, r₂::AbstractVector{Int}, r₁::AbstractVector{Int},\n                  α::Int, unit_cell::UnitCell{D,E}, lattice::Lattice{D},\n                  sgn::T=one(T)) where {D, C<:Complex, T<:Number, E<:AbstractFloat}\n\nEvaluate the sum\n\nS_mathbfr=S_mathbfr+fracalphaNsum_mathbfiG_sigma_2mathbfi+mathbfr+mathbfr_4mathbfi+mathbfr_3^ab(tau_20)cdot G_sigma_1mathbfi+mathbfr+mathbfr_2mathbfi+mathbfr_1^cd(tau_10)\n\nfor all mathbfr\n\n\n\n\n\n","category":"function"},{"location":"api/#JDQMCMeasurements.contract_G0r_G0r!","page":"API","title":"JDQMCMeasurements.contract_G0r_G0r!","text":"contract_G0r_G0r!(S::AbstractArray{C}, G₂::AbstractMatrix{T}, G₁::AbstractMatrix{T},\n                  a::Int, b::Int, c::Int, d::Int,\n                  r₄::AbstractVector{Int}, r₃::AbstractVector{Int}, r₂::AbstractVector{Int}, r₁::AbstractVector{Int},\n                  α::Int, unit_cell::UnitCell{D,E}, lattice::Lattice{D},\n                  sgn::T=one(T)) where {D, C<:Complex, T<:Number, E<:AbstractFloat}\n\nEvaluate the sum\n\nS_mathbfr=S_mathbfr+fracalphaNsum_mathbfiG_sigma_2mathbfi+mathbfr_4mathbfi+mathbfr+mathbfr_3^ab(tau_20)cdot G_sigma_1mathbfi+mathbfr_2mathbfi+mathbfr+mathbfr_1^cd(tau_10)\n\nfor all mathbfr\n\n\n\n\n\n","category":"function"},{"location":"api/#JDQMCMeasurements.contract_G0r_Gr0!","page":"API","title":"JDQMCMeasurements.contract_G0r_Gr0!","text":"contract_G0r_Gr0!(S::AbstractArray{C}, G₂::AbstractMatrix{T}, G₁::AbstractMatrix{T}, b₂::Bond, b₁::Bond,\n                  α::Int, unit_cell::UnitCell{D,E}, lattice::Lattice{D},\n                  sgn::T=one(T)) where {D, C<:Complex, T<:Number, E<:AbstractFloat}\n\nEvaluate the sum\n\nS_mathbfr=S_mathbfr+fracalphaNsum_mathbfiG_sigma_2mathbfi+mathbfr_2mathbfi+mathbfr^ab(tau_20)cdot G_sigma_1mathbfi+mathbfr+mathbfr_1mathbfi^cd(tau_10)\n\nfor all mathbfr where the bond b₂ represents the static displacement mathbfr_2 + (mathbfr_a - mathbfr_b) and the bond b₁ represents the static displacement mathbfr_1 + (mathbfr_c - mathbfr_d)\n\n\n\n\n\ncontract_G0r_Gr0!(S::AbstractArray{C}, G₂::AbstractMatrix{T}, G₁::AbstractMatrix{T},\n                  a::Int, b::Int, c::Int, d::Int,\n                  r₄::AbstractVector{Int}, r₃::AbstractVector{Int}, r₂::AbstractVector{Int}, r₁::AbstractVector{Int},\n                  α::Int, unit_cell::UnitCell{D,E}, lattice::Lattice{D},\n                  sgn::T=one(T)) where {D, C<:Complex, T<:Number, E<:AbstractFloat}\n\nEvaluate the sum\n\nS_mathbfr=S_mathbfr+fracalphaNsum_mathbfiG_sigma_2mathbfi+mathbfr_4mathbfi+mathbfr+mathbfr_3^ab(tau_20)cdot G_sigma_1mathbfi+mathbfr+mathbfr_2mathbfi+mathbfr_1^cd(tau_10)\n\nfor all mathbfr\n\n\n\n\n\n","category":"function"},{"location":"api/#JDQMCMeasurements.contract_Gr0_G0r!","page":"API","title":"JDQMCMeasurements.contract_Gr0_G0r!","text":"contract_Gr0_G0r!(S::AbstractArray{C}, G₂::AbstractMatrix{T}, G₁::AbstractMatrix{T},\n                  a::Int, b::Int, c::Int, d::Int,\n                  r₄::AbstractVector{Int}, r₃::AbstractVector{Int}, r₂::AbstractVector{Int}, r₁::AbstractVector{Int},\n                  α::Int, unit_cell::UnitCell{D,E}, lattice::Lattice{D},\n                  sgn::T=one(T)) where {D, C<:Complex, T<:Number, E<:AbstractFloat}\n\nEvaluate the sum\n\nS_mathbfr=S_mathbfr+fracalphaNsum_mathbfiG_sigma_2mathbfi+mathbfr+mathbfr_4mathbfi+mathbfr_3^ab(tau_20)cdot G_sigma_1mathbfi+mathbfr_2mathbfi+mathbfr+mathbfr_1^cd(tau_10)\n\nfor all mathbfr\n\n\n\n\n\n","category":"function"},{"location":"api/#JDQMCMeasurements.simpson","page":"API","title":"JDQMCMeasurements.simpson","text":"simpson(f::AbstractVector{T}, dx::E) where {T<:Number, E<:AbstractFloat}\n\nApplying Simpson's rule, integrate over the vector f using a stepsize dx.\n\n\n\n\n\n","category":"function"},{"location":"","page":"Home","title":"Home","text":"CurrentModule = JDQMCMeasurements","category":"page"},{"location":"#JDQMCMeasurements","page":"Home","title":"JDQMCMeasurements","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Documentation for JDQMCMeasurements.","category":"page"},{"location":"","page":"Home","title":"Home","text":"This package implements a variety of functions that can be called to measure various correlation functions in a determinant quantum Monte Carlo (DQMC) simulation. The exported correlation function measurements support arbitrary lattice geometries. This package also exports several additional utility functions for transforming measurements from position space to momentum space, and also measuring susceptibilities by integrating correlation functions over the imaginary time axis.","category":"page"},{"location":"","page":"Home","title":"Home","text":"This package relies on the LatticeUtilities.jl to represent arbitary lattice geometries.","category":"page"},{"location":"#Funding","page":"Home","title":"Funding","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"The development of this code was supported by the U.S. Department of Energy, Office of Science, Basic Energy Sciences, under Award Number DE-SC0022311.","category":"page"},{"location":"#Installation","page":"Home","title":"Installation","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"To install JDQMCMeasurements.jl run following in the Julia REPL:","category":"page"},{"location":"","page":"Home","title":"Home","text":"] add JDQMCFramework","category":"page"},{"location":"#Notation","page":"Home","title":"Notation","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"The operators hata_sigmamathbfi^daggerhatb_sigmamathbfi^daggerhatc_sigmamathbfi^daggerhatd_sigmamathbfi^daggerbig(hata_sigmamathbfihatb_sigmamathbfihatc_sigmamathbfihatd_sigmamathbfibig) are fermion creation (annihilation) operators for an electron with spin sigma=uparrowdownarrow. The letter of the operator is treated as a variable denoting a speicific orbital in the unit cell. Bolded variables absent a subscript, like mathbfi, denote a displacement in unit cells. As an example, the term","category":"page"},{"location":"","page":"Home","title":"Home","text":"hata_uparrowmathbfi+mathbfr^daggerhatb_uparrowmathbfi","category":"page"},{"location":"","page":"Home","title":"Home","text":"describes a spin up electron being annihilated from orital b in the unit cell located at mathbfi, and a spin up electron being created in orbital a in the unit cell located at mathbfi+mathbfr. The locations of orbitals a and b in the lattice are mathbfi+mathbfr+mathbfu_a and mathbfi+mathbfu_b respectively, with the corresponding displacement vector between the two orbitals given by mathbfr+(mathbfu_a-mathbfu_b). Here mathbfu_a and mathbfu_b are basis vectors describing the relative location of the a and b orbitals within the unit cell. In the docs we assume a finite lattice of N unit cells with periodic boundary conditions.","category":"page"},{"location":"#Green's-Function-Definition","page":"Home","title":"Green's Function Definition","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"The single-particle electron Green's function is given by","category":"page"},{"location":"","page":"Home","title":"Home","text":"beginalign*\nG_sigmamathbfi+mathbfrmathbfi^ab(tautau) =  langlehatmathcalThata_sigmamathbfi+mathbfr(tau)hatb_sigmamathbfi^dagger(tau)rangle\n                                                              =  begincases\n     langlehata_sigmamathbfi+mathbfr(tau)hatb_sigmamathbfi^dagger(tau)rangle  beta(tau-tau)ge0\n    -langlehatb_sigmamathbfi^dagger(tau)hata_sigmamathbfi+mathbfr(tau)rangle  -betale(tau-tau)0\nendcases\nendalign*","category":"page"},{"location":"","page":"Home","title":"Home","text":"subject to the aperiodic boundary condition G_sigmamathbfi+mathbfrmathbfi^ab(tau-betatau)=-G_sigmamathbfi+mathbfrmathbfi^ab(tautau)","category":"page"},{"location":"","page":"Home","title":"Home","text":"For reasons of algebraic convenience and clarity, in the remainder of the note we will assume beta  tau ge 0 and tau=0, with the retarded and advanced imaginary time Green's functions given by","category":"page"},{"location":"","page":"Home","title":"Home","text":"G_sigmamathbfi+mathbfrmathbfi^ab(tau0) = langlehata_sigmamathbfi+mathbfr(tau)hatb_sigmamathbfi^dagger(0)rangle","category":"page"},{"location":"","page":"Home","title":"Home","text":"and","category":"page"},{"location":"","page":"Home","title":"Home","text":"G_sigmamathbfi+mathbfrmathbfi^ab(0tau) = -langlehatb_sigmamathbfi^dagger(tau)hata_sigmamathbfi+mathbfr(0)rangle","category":"page"},{"location":"","page":"Home","title":"Home","text":"respectively. Note that in cases where G_sigmamathbfi+mathbfrmathbfi^ab(0tau) terms appear in a final expression, when tau=0 they are equal to","category":"page"},{"location":"","page":"Home","title":"Home","text":"beginalign*\nG_sigmamathbfi+mathbfrmathbfi^ab(0tau=0)  =-langlehatb_sigmamathbfi^dagger(0)hata_sigmamathbfi+mathbfr(0)rangle \n  =-(delta_mathbfr0delta_ab-langlehata_sigmamathbfi+mathbfr(0)hatb_sigmamathbfi^dagger(0)rangle) \n  =-(delta_mathbfr0delta_ab-G_sigmamathbfi+mathbfrmathbfi^ab(00)) \n  =G_sigmamathbfi+mathbfrmathbfi^ab(00)-delta_mathbfr0delta_ab\nendalign*","category":"page"},{"location":"","page":"Home","title":"Home","text":"where G_sigmamathbfi+mathbfrmathbfi^ab(00) = langlehata_sigmamathbfi+mathbfr^phantomdagger(0)hatb_sigmamathbfi^dagger(0)rangle=langlehata_sigmamathbfi+mathbfr^phantomdaggerhatb_sigmamathbfi^daggerrangle is the equal-time Green's function. This assumes equal-time Green's function is defined as","category":"page"},{"location":"","page":"Home","title":"Home","text":"G_sigmamathbfi+mathbfrmathbfi^ab(tautau)=G_sigmamathbfi+mathbfrmathbfi^ab(tau^+tau)=langlehata_sigmamathbfi+mathbfr(tau)hatb_sigmamathbfi^dagger(tau)rangle","category":"page"},{"location":"","page":"Home","title":"Home","text":"with the operator order reflecting the retarded imaginary time Green's functions G_sigmamathbfi+mathbfrmathbfi^ab(tau0) defintion.","category":"page"},{"location":"","page":"Home","title":"Home","text":"Also, because tau in 0 beta), we also introduce the defintions","category":"page"},{"location":"","page":"Home","title":"Home","text":"G_sigmamathbfi+mathbfrmathbfi^ab(beta0) = lim_delta^+rightarrow0G_sigmamathbfi+mathbfrmathbfi^ab(beta-delta^+0)=delta_mathbfr0delta_ab-G_sigmamathbfi+mathbfrmathbfi^ab(00)","category":"page"},{"location":"","page":"Home","title":"Home","text":"and","category":"page"},{"location":"","page":"Home","title":"Home","text":"beginequation\nG_sigmamathbfi+mathbfrmathbfi^ab(0beta)=lim_delta^+rightarrow0G_sigmamathbfi+mathbfrmathbfi^ab(0beta-delta^+)=-G_sigmamathbfi+mathbfrmathbfi^ab(00)\nendequation","category":"page"},{"location":"","page":"Home","title":"Home","text":"that enforce appropriate boundary conditions. Finally, we define G_sigmamathbfi+mathbfrmathbfi^ab(betabeta)=G_sigmamathbfi+mathbfrmathbfi^ab(00)","category":"page"},{"location":"#Boundary-Conditions-using-Matrix-Notation","page":"Home","title":"Boundary Conditions using Matrix Notation","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"In order to make equal-time correlation measurements where tau = 0 and for tau = beta, boundary conditions need to be handled appropriately. This information was given in the section above, but here we repeat the most relevant points using matrix notation in terms of the equal-time Green's function matrix G_sigma(00):","category":"page"},{"location":"","page":"Home","title":"Home","text":"beginalign*\nG_sigma(beta beta) = G_sigma(00) \nG_sigma(0tau = 0) = G_sigma(00) - I \nG_sigma(tau = beta 0) = I - G_sigma(00) \nG_sigma(0 tau = beta) = -G_sigma(00) \nendalign*","category":"page"},{"location":"#Comment-on-Representing-Arbitrary-Lattice-Geometries","page":"Home","title":"Comment on Representing Arbitrary Lattice Geometries","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"This package uses the LatticeUtilities.jl to represent arbitary lattice geometries. The LatticeUtilities.jl exports three types that are used in this package. The UnitCell type describes the unit cell, the Lattice type describes the size of the finite periodic lattice in question, and the Bond type defines a bond/static displacement in the lattice.","category":"page"},{"location":"","page":"Home","title":"Home","text":"With regard to the Bond type in particular, a displacement from orbital b to orbital a displaced mathbfr unit cells away, with the corresponding static displacement vector given by mathbfr + (mathbfu_a-mathbfu_b) should be defined as follows:","category":"page"},{"location":"","page":"Home","title":"Home","text":"b = 2\na = 1\nr = [1,0,0]\nbond = Bond((b,a),r)","category":"page"},{"location":"","page":"Home","title":"Home","text":"In particular, note the ordering of the tuple when declaring bond above, the tuple (b,a) indicates a bond going from orbital b to orbital a Refer to the LatticeUtilities.jl documentation for more information.","category":"page"}]
}
