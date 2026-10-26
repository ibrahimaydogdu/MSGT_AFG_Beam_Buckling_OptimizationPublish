"""Constituent phases and axial rule of mixtures (Section 2.3, Eqs. 35-37).

Phi(x) = Phi_L + (Phi_R - Phi_L) V_r(x) for E, G, rho; the shear modulus of each
phase is G_i = E_i / (2 (1 + nu_i)) and is blended afterwards (Eqs. 36-37);
nu itself is never blended. V_r is the volume fraction of the RIGHT-end phase.
Units: SI (Pa, kg/m^3).
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class Phase:
    name: str
    E: float            # Young's modulus [Pa]
    nu: float           # Poisson's ratio [-]
    rho: float | None   # mass density [kg/m^3]; None = not specified
    source: str = ""

    @property
    def G(self):
        return self.E / (2.0 * (1.0 + self.nu))


@dataclass(frozen=True)
class MaterialPair:
    left: Phase
    right: Phase
    name: str = ""

    def mix(self, Vr):
        """E(x), G(x), rho(x) at the nodes (Eqs. 35-37)."""
        E = self.left.E + (self.right.E - self.left.E) * Vr
        G = self.left.G + (self.right.G - self.left.G) * Vr
        if self.left.rho is None or self.right.rho is None:
            rho = None
        else:
            rho = self.left.rho + (self.right.rho - self.left.rho) * Vr
        return E, G, rho


# Engineering pair (Table 2 of the manuscript); left = Al (metal), right = SiC (ceramic).
AL = Phase("Al", 67e9, 0.345, 2700.0,
           "E, rho: Civalek et al. 2023 (ZAMM) Table 1 (secondary); nu: Voyiadjis et al. 2011 Table 2")
SIC = Phase("SiC", 427e9, 0.17, 3200.0,
            "E, nu: Liu et al. 2013 p.5; rho: Civalek et al. 2023 Table 1 (secondary)")
AL_SIC = MaterialPair(AL, SIC, "Al/SiC")

# Synthetic benchmark pair (E_L/E_R = 2, nu = 0.3). Densities are NOT specified in the
# manuscript; mass objectives for this pair stay undefined until the user fixes them.
SYN_L = Phase("synthetic-L", 2.0, 0.3, None, "Akgoz & Civalek 2022 Tables 4-5: E_L = 2 E_R")
SYN_R = Phase("synthetic-R", 1.0, 0.3, None, "Akgoz & Civalek 2022 Tables 4-5")
SYNTHETIC = MaterialPair(SYN_L, SYN_R, "synthetic")
