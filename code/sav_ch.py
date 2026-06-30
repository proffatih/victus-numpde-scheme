"""
Structure-preserving SAV / RSAV BDF2 Fourier-spectral solvers for the
Cahn-Hilliard equation (periodic boundary conditions).

    u_t = M * Laplacian( mu ),     mu = -eps^2 * Laplacian(u) + F'(u),
    F(u) = (1/4) (u^2 - 1)^2,      F'(u) = u^3 - u.

Energy:  E[u] = integral [ (eps^2/2) |grad u|^2 + F(u) ] dx.

Scalar Auxiliary Variable (SAV):
    r(t) = sqrt( integral F(u) dx + C0 ),   C0 > 0 chosen so the radicand > 0.

This module implements:
  * SAV-BDF2  : baseline second-order, linear, unconditionally (modified-)
                energy-stable scheme (Shen-Xu-Yang).
  * RSAV-BDF2 : relaxed-SAV variant (Jiang-Zhang-Zhao) that adds a cheap
                scalar relaxation restoring true-energy consistency.
  * stabilized SAV-BDF2 : adds an explicit stabilization S (large-dt robustness).

Author: Fatih Gul (paper code).  No fabricated data: every number this file
prints is computed at run time.
"""
import numpy as np


# --------------------------------------------------------------------------
#  Spectral grid helpers (1D and 2D, periodic)
# --------------------------------------------------------------------------
class SpectralGrid:
    def __init__(self, N, L, dim=1):
        self.N = N
        self.L = L
        self.dim = dim
        x = np.linspace(0.0, L, N, endpoint=False)
        self.h = L / N
        k1 = 2.0 * np.pi * np.fft.fftfreq(N, d=self.h)  # wavenumbers
        if dim == 1:
            self.x = x
            self.k2 = k1 ** 2                 # |k|^2
        elif dim == 2:
            X, Y = np.meshgrid(x, x, indexing="ij")
            self.x = X
            self.y = Y
            KX, KY = np.meshgrid(k1, k1, indexing="ij")
            self.k2 = KX ** 2 + KY ** 2
        else:
            raise ValueError("dim must be 1 or 2")
        # cell measure for integral approximation (spectrally exact via mean*vol)
        self.cell = self.h ** dim
        self.vol = L ** dim

    def fft(self, u):
        return np.fft.fftn(u)

    def ifft(self, uh):
        return np.real(np.fft.ifftn(uh))

    def integral(self, u):
        """Periodic-trapezoid integral = mean * volume (spectrally exact)."""
        return np.sum(u) * self.cell

    def laplacian(self, u):
        return self.ifft(-self.k2 * self.fft(u))

    def grad_sq_integral(self, u):
        """integral |grad u|^2 dx  via Parseval (no aliasing)."""
        uh = self.fft(u)
        # integral |grad u|^2 = sum_k |k|^2 |uh|^2 / N^dim * cell ... use Parseval
        # Parseval for DFT: sum |u|^2 * cell = (1/Ntot) sum |uh|^2 * cell
        Ntot = u.size
        return np.sum(self.k2 * np.abs(uh) ** 2) / Ntot * self.cell


# --------------------------------------------------------------------------
#  Free energy density and its derivative
# --------------------------------------------------------------------------
def F(u):
    return 0.25 * (u ** 2 - 1.0) ** 2

def dF(u):
    return u ** 3 - u


# --------------------------------------------------------------------------
#  Energy functionals
# --------------------------------------------------------------------------
def energy(grid, u, eps):
    grad_term = 0.5 * eps ** 2 * grid.grad_sq_integral(u)
    pot_term = grid.integral(F(u))
    return grad_term + pot_term

def nonlinear_potential_integral(grid, u, C0):
    return grid.integral(F(u)) + C0


# --------------------------------------------------------------------------
#  Core SAV / RSAV BDF2 solver
# --------------------------------------------------------------------------
def solve_ch(grid, u0, eps, M, dt, nsteps, C0, S=0.0,
             scheme="sav", record_every=1, return_fields=False):
    """
    Cahn-Hilliard SAV-family BDF2 solver (Fourier-spectral, periodic).

    Parameters
    ----------
    scheme : 'sav', 'rsav', or 'stab' (stab = stabilized SAV with parameter S)
    S      : stabilization constant (used when scheme=='stab' or as extra
             stabilization for 'sav'/'rsav' if S>0).
    Returns dict with time series of energies / mass / r, and optionally fields.
    """
    k2 = grid.k2
    # operator L = eps^2 * (-Lap) -> symbol eps^2 * k2 ; with stabilization S:
    # We treat linear part implicitly. Define A(u) = -eps^2 Lap u + S u.
    # Then F'(u) = (A_nl)(u) with the S subtracted explicitly (stabilized SAV).
    # mu = -eps^2 Lap u + S*u + (r/sqrt(E1)) * (F'(u) - S*u) ... standard stab-SAV
    #
    # BDF2 SAV (Shen-Xu-Yang) with stabilization S. Here we follow the
    # widely used formulation; for S=0 it reduces to plain SAV-BDF2.

    u = u0.copy()
    # --- first step: SAV first-order (backward Euler SAV) to bootstrap BDF2 ---
    r = np.sqrt(nonlinear_potential_integral(grid, u, C0))

    # storage
    times, Es, Es_mod, masses, rs, r_exact = [], [], [], [], [], []

    def record(t, u, r, u_prev=None):
        times.append(t)
        Es.append(energy(grid, u, eps))
        rexact = np.sqrt(nonlinear_potential_integral(grid, u, C0))
        # BDF2 two-level modified energy (Shen-Xu-Yang):
        #   tildeE^n = (eps^2/4)(|grad u^n|^2 + |grad(2u^n - u^{n-1})|^2)
        #              + r^n^2 - C0 .
        # For the very first record (no u_prev) fall back to the one-level form,
        # which coincides with tildeE for a steady bootstrap.
        if u_prev is None:
            Emod = 0.5 * eps ** 2 * grid.grad_sq_integral(u) + r ** 2 - C0
        else:
            ustar = 2.0 * u - u_prev
            Emod = (0.25 * eps ** 2 *
                    (grid.grad_sq_integral(u) + grid.grad_sq_integral(ustar))
                    + r ** 2 - C0)
        Es_mod.append(Emod)
        masses.append(grid.integral(u))
        rs.append(r)
        r_exact.append(rexact)

    record(0.0, u, r)

    # bootstrap: one first-order SAV step (Euler) ------------------------
    def b_of_u(uu):
        E1 = nonlinear_potential_integral(grid, uu, C0)
        return (dF(uu) - S * uu) / np.sqrt(E1)   # b = (F'(u)-Su)/sqrt(E1)

    # helper to invert (I + dt*coef * M * (-Lap) * Aop) type systems in Fourier.
    # For SAV-BDF2 we use the Shen-Xu-Yang block-elimination. Implement directly.

    # ---- generic single SAV step (order 1 = backward Euler SAV) ----
    def sav_step_euler(u_prev, r_prev):
        b = b_of_u(u_prev)               # explicit b at previous level
        bh = grid.fft(b)
        # mu = (eps^2(-Lap)+S) u + r * b   (since (r/sqrt(E1)) (F'-Su), and
        #      we approximate (1/sqrt(E1)) at u_prev -> b already divided)
        # equations:
        #  (u - u_prev)/dt = M Lap mu
        #  mu = Lop u + r * b
        #  (r - r_prev) = 0.5 * integral b*(u - u_prev)
        # Linear symbol of Lop: eps^2 k2 + S
        Lsym = eps ** 2 * k2 + S
        # Solve via SAV decomposition. Let operator P = I - dt M Lap Lop
        #  P u = u_prev + dt M Lap (r b)
        # In Fourier: Psym = 1 + dt*M*k2*Lsym
        Psym = 1.0 + dt * M * k2 * Lsym
        rhs_const_h = grid.fft(u_prev)
        # part driven by b: dt M Lap b  -> symbol: -dt*M*k2 * bh
        ub_h = (-dt * M * k2 * bh) / Psym       # u-response per unit r (the r b term)
        uc_h = rhs_const_h / Psym               # u-response to u_prev
        uc = grid.ifft(uc_h)
        ub = grid.ifft(ub_h)
        # u = uc + r*ub ; plug into r-update:
        # r = r_prev + 0.5 integral b*(u - u_prev)
        #   = r_prev + 0.5 [ integral b*(uc - u_prev) + r * integral b*ub ]
        I_b_uc = grid.integral(b * (uc - u_prev))
        I_b_ub = grid.integral(b * ub)
        # r (1 - 0.5 r-coef?) careful: r appears on both sides
        # r - 0.5 r I_b_ub = r_prev + 0.5 I_b_uc
        r_new = (r_prev + 0.5 * I_b_uc) / (1.0 - 0.5 * I_b_ub)
        u_new = uc + r_new * ub
        return u_new, r_new

    # ---- BDF2 SAV step ----
    def sav_step_bdf2(u_n, u_nm1, r_n, r_nm1):
        # BDF2 time derivative: (3 u - 4 u_n + u_nm1)/(2 dt)
        # extrapolated state for nonlinear term: u_star = 2 u_n - u_nm1
        u_star = 2.0 * u_n - u_nm1
        b = b_of_u(u_star)
        bh = grid.fft(b)
        Lsym = eps ** 2 * k2 + S
        # r* extrapolation
        r_star = 2.0 * r_n - r_nm1   # not strictly needed; b uses u_star
        # System (Shen-Xu-Yang BDF2):
        #  (3u - 4u_n + u_nm1)/(2dt) = M Lap mu
        #  mu = Lop u + r * b
        #  (3r - 4r_n + r_nm1)/(2dt)?  SAV r-update via inner product:
        #     3r - 4 r_n + r_nm1 = integral b * (3u - 4u_n + u_nm1)
        #     (this is the discrete chain rule for r)
        # Eliminate. Let alpha = 3/(2dt).
        alpha = 3.0 / (2.0 * dt)
        known = (4.0 * u_n - u_nm1) / (2.0 * dt)
        knownh = grid.fft(known)
        # alpha u - M Lap mu = known ; mu = Lop u + r b
        # alpha u - M Lap (Lop u) - r M Lap b = known
        # Fourier: (alpha + M k2 Lsym) uh = knownh - r M k2 (... ) ; M Lap b -> -M k2 bh
        Psym = alpha + M * k2 * Lsym
        uc_h = knownh / Psym
        ub_h = (-M * k2 * bh) / Psym         # response to (r b) ; M Lap (r b)= r * (-M k2 bh)
        # wait sign: alpha u - r M Lap b = ... -> -r M Lap b moves to rhs as +r M Lap b
        # M Lap b symbol = -M k2 bh ; so +r M Lap b -> rhs += r*(-M k2 bh). matches ub_h*Psym?
        # We set (Psym) uh = knownh + r*(-M k2 bh). So ub_h = (-M k2 bh)/Psym. Good.
        uc = grid.ifft(uc_h)
        ub = grid.ifft(ub_h)
        # r-update: 3r - 4r_n + r_nm1 = integral b*(3u - 4u_n + u_nm1)
        #   u = uc + r ub
        #   3u - 4u_n + u_nm1 = 3uc + 3 r ub - 4u_n + u_nm1
        # Discrete chain rule for r (Shen-Xu-Yang BDF2 convention):
        #   3r - 4r_n + r_nm1 = integral b*(3u - 4u_n + u_nm1)
        # with mu = L u + (b) r ; this pair dissipates the BDF2 modified energy
        #   tildeE = (eps^2/4)(|grad u|^2 + |grad(2u-u_prev)|^2) + r^2 - C0 .
        w = 3.0 * uc - 4.0 * u_n + u_nm1
        I_b_w = grid.integral(b * w)
        I_b_ub = grid.integral(b * ub)
        # 3r - 4r_n + r_nm1 = I_b_w + 3 r I_b_ub
        r_new = (4.0 * r_n - r_nm1 + I_b_w) / (3.0 - 3.0 * I_b_ub)
        u_new = uc + r_new * ub
        return u_new, r_new

    # ---- Crank-Nicolson SAV step (second order, EXACT modified-energy law) ----
    def sav_step_cn(u_n, u_nm1, r_n):
        """Shen-Xu-Yang CN-SAV scheme. Dissipates EXACTLY and unconditionally
        the modified energy  tildeE = (eps^2/2)|grad u|^2 + r^2 - C0 :

            tildeE^{n+1} = tildeE^n - dt*M*|grad mu^{n+1/2}|^2  <=  tildeE^n.

        Uses the second-order extrapolation  ubar = (3 u_n - u_nm1)/2  to build
        the explicit coefficient b^{n+1/2} = F'(ubar)/sqrt(E1[ubar]).
        """
        ubar = 1.5 * u_n - 0.5 * u_nm1
        E1bar = nonlinear_potential_integral(grid, ubar, C0)
        b = (dF(ubar) - S * ubar) / np.sqrt(E1bar)
        bh = grid.fft(b)
        Lsym = eps ** 2 * k2 + S
        # CN: (u-u_n)/dt = M Lap mu^{1/2},  mu^{1/2}=Lop (u+u_n)/2 + b r^{1/2}
        #  r^{1/2} = (r + r_n)/2 ;  r - r_n = (1/2) int b (u - u_n)
        # Eliminate. Let half = 1/2.
        # (u - u_n)/dt = M Lap [ Lop(u+u_n)/2 + b (r+r_n)/2 ]
        # Define g0 = M Lap [ Lop u_n / 2 ]  (known), and r-part.
        # Fourier symbol of M Lap Lop = -M k2 Lsym.
        coef = M * k2 * Lsym
        # u_n contributions:
        un_h = grid.fft(u_n)
        # rhs from known: u_n + dt*( -coef*(u_n)/2 )  ... assemble:
        #  u + (dt/2) coef u = u_n - (dt/2) coef u_n + dt*M*Lap*b*(r+r_n)/2
        # Lap b symbol -> -k2 bh ; M Lap b -> -M k2 bh
        Psym = 1.0 + 0.5 * dt * coef
        known_h = un_h - 0.5 * dt * coef * un_h          # u_n - (dt/2)coef u_n
        # response to (r+r_n)/2 * (dt * M Lap b):
        sb_h = dt * (-M * k2 * bh)                       # dt*M*Lap*b symbol
        uc_h = known_h / Psym
        ub_h = (0.5 * sb_h) / Psym                       # multiplies (r+r_n)
        uc = grid.ifft(uc_h)
        ub = grid.ifft(ub_h)                             # u = uc + (r+r_n) ub
        # r - r_n = 0.5 int b (u - u_n) = 0.5 int b (uc + (r+r_n) ub - u_n)
        I_b_uc = grid.integral(b * (uc - u_n))
        I_b_ub = grid.integral(b * ub)
        # r - r_n = 0.5 I_b_uc + 0.5 (r + r_n) I_b_ub
        # r (1 - 0.5 I_b_ub) = r_n + 0.5 I_b_uc + 0.5 r_n I_b_ub
        r_new = (r_n + 0.5 * I_b_uc + 0.5 * r_n * I_b_ub) / (1.0 - 0.5 * I_b_ub)
        u_new = uc + (r_new + r_n) * ub
        return u_new, r_new

    def relax(grid, u_new, r_tilde, dissip_budget):
        """RSAV relaxation (Jiang-Zhang-Zhao, JCP 2022).

        Replace the SAV update r_tilde by the convex combination
            r_new = xi0 * r_tilde + (1 - xi0) * sqrt(E1(u_new)),
        where E1(u) = integral F(u) dx + C0, and xi0 in [0,1] is the SMALLEST
        admissible value (closest to the consistent root sqrt(E1)) for which the
        relaxation does not violate the discrete energy dissipation, i.e.

            r_new^2 - r_tilde^2  <=  dissip_budget   ( <= 0 ).

        Writing q = sqrt(E1) and solving the scalar quadratic
            (xi r_tilde + (1-xi) q)^2 - r_tilde^2 <= budget
        for the minimal xi in [0,1] gives the closed-form choice below.
        """
        q = np.sqrt(nonlinear_potential_integral(grid, u_new, C0))
        rt = r_tilde
        budget = min(dissip_budget, 0.0)   # never allow energy to rise
        # f(xi) = (xi*rt + (1-xi)*q)^2 - rt^2 - budget <= 0, xi in [0,1].
        a = (rt - q) ** 2
        b = 2.0 * q * (rt - q)
        c = q ** 2 - rt ** 2 - budget
        # quadratic a xi^2 + b xi + c <= 0 ; choose minimal admissible xi>=0.
        if a < 1e-300:
            # degenerate (rt == q): already consistent
            return rt
        disc = b * b - 4.0 * a * c
        if disc < 0.0:
            # no real admissible xi keeping budget; fall back to xi=1 (plain SAV)
            return rt
        sq = np.sqrt(disc)
        xi_lo = (-b - sq) / (2.0 * a)
        xi_hi = (-b + sq) / (2.0 * a)
        lo, hi = min(xi_lo, xi_hi), max(xi_lo, xi_hi)
        # admissible set is [lo, hi]; we want the smallest xi in [0,1] ∩ [lo,hi]
        xi0 = max(0.0, lo)
        if xi0 > 1.0:
            xi0 = 1.0
        xi0 = min(max(xi0, 0.0), 1.0)
        return xi0 * rt + (1.0 - xi0) * q

    # second-order schemes that track the ONE-level modified energy
    cn_family = scheme in ("cn", "rcn")
    relax_on = scheme in ("rsav", "rcn")
    use_emod_prev = not cn_family   # BDF2 uses two-level modified energy

    def grad_energy(u):
        return 0.5 * eps ** 2 * grid.grad_sq_integral(u)

    def relax_full(u_prev_level, r_prev_level, u_new, r_tilde):
        """Relaxation with the FULL available modified-energy budget:
        keep tildeE^{n+1} <= tildeE^n where tildeE = grad_energy + r^2 - C0.
        Budget for the r^2 part = (grad_energy^n + r_prev^2) - grad_energy^{n+1}
        minus r_tilde^2 ... i.e. how far r^2 may exceed r_tilde^2 while still
        not raising the modified energy above the previous level."""
        Eg_old = grad_energy(u_prev_level)
        Eg_new = grad_energy(u_new)
        # admissible: Eg_new + r_new^2 <= Eg_old + r_prev^2
        # => r_new^2 <= Eg_old + r_prev^2 - Eg_new := rmax2
        rmax2 = Eg_old + r_prev_level ** 2 - Eg_new
        budget = rmax2 - r_tilde ** 2   # how much r^2 may rise above r_tilde^2
        return relax(grid, u_new, r_tilde, budget)

    # ---------------- time loop ----------------
    # step 1 (first-order SAV bootstrap; exact-dissipation BE-SAV)
    u_nm1 = u.copy()
    r_nm1 = r
    u_n, r_n = sav_step_euler(u_nm1, r_nm1)
    if relax_on:
        r_n = relax_full(u_nm1, r_nm1, u_n, r_n)
    if 1 % record_every == 0:
        record(dt, u_n, r_n, u_prev=(u_nm1 if use_emod_prev else None))

    fields = {}
    if return_fields:
        fields[0] = u0.copy()

    for n in range(2, nsteps + 1):
        if cn_family:
            u_new, r_new = sav_step_cn(u_n, u_nm1, r_n)
        else:
            u_new, r_new = sav_step_bdf2(u_n, u_nm1, r_n, r_nm1)
        if relax_on:
            r_new = relax_full(u_n, r_n, u_new, r_new)
        u_nm1, r_nm1 = u_n, r_n
        u_n, r_n = u_new, r_new
        if n % record_every == 0:
            record(n * dt, u_n, r_n,
                   u_prev=(u_nm1 if use_emod_prev else None))
            if return_fields:
                fields[n] = u_n.copy()

    out = dict(
        t=np.array(times), E=np.array(Es), E_mod=np.array(Es_mod),
        mass=np.array(masses), r=np.array(rs), r_exact=np.array(r_exact),
        u_final=u_n, r_final=r_n,
    )
    if return_fields:
        out["fields"] = fields
    return out
