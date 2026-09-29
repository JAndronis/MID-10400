from typing import Literal, NamedTuple, get_args

import extra_data as ex
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from numpy.typing import ArrayLike, NDArray
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit

N_A: float = 6.02214076e23
 
M_APOFERRITIN: float = 481_000.0  # g/mol, horse spleen (Gilbert et al. 1987)
V_BAR_APOFERRITIN: float = 0.738  # mL/g, 20 C
M_FE: float = 55.845
M_FEOOH: float = 88.852  # g/mol of core per Fe atom, ferrihydrite approximation
RHO_CORE: float = 4.1  # g/mL, ferritin ferrihydrite core
D_FERRITIN_NM: float = 12.0  # outer diameter of the cage
V_BAR_PEG: float = 0.833  # mL/g
RHO_WATER_25C: float = 0.99705  # g/mL
M_NACL: float = 58.44  # g/mol

# Mass attenuation coefficients mu/rho [cm^2/g] at 9.04 keV, the photon energy the
# SAXS pass integrates with; the XGM reports 9.00 keV (CLAUDE.md open task 15), where
# they are ~1.3 % higher. xraydb 4.5.8 (Elam, Ravel & Sieber 2002); "protein" is
# C5H8NO2S0.05 and the ferritin core is FeOOH, as in holoferritin_properties.
MU_RHO_9P04_KEV: dict[str, float] = {
    "H2O": 7.1949,
    "NaCl": 54.5082,
    "PEG": 4.6956,
    "FeOOH": 143.142,
    "protein": 5.4658,
}
# Droplet camera pixel size: droplet_fit(px=13.9) in src/amore/context.py.
DROPLET_PX_MM: float = 13.9e-3

DB_VARIABLES = Literal["agipd_saxs", "jungfrau_waxs_jf1", "jungfrau_waxs_jf2", "jungfrau_waxs_combined"]


def plot_saxs_series(dataset, ax, step=100, bg=None):
    q = dataset.q.values
    for i in range(0, len(dataset.trainId)-step, step):
        iq = dataset.isel(trainId=slice(i, i+step)).mean(["trainId", "pulseId"]).data
        if bg is not None:
            iq -= bg
        ax.loglog(q, iq)
    return ax


def read_intensity(run_nr, db, variable: DB_VARIABLES):
    assert variable in get_args(DB_VARIABLES), f"Variable {variable} not in {DB_VARIABLES}"
    return xr.open_dataset(db[run_nr].file, group=variable).intensity


def read_volume(run_nr, db):
     return xr.open_dataset(db[run_nr].file, group="droplet_fit").volume


def get_train_timestamps(run_nr, db):
    run = ex.RunDirectory(f"/gpfs/exfel/exp/MID/202601/p010400/raw/r{run_nr:04d}")
    ts = run.train_timestamps()
    return ts

    
def read_volume_series(run_series, db):
    v0 = xr.open_dataset(db[run_series[0]].file, group="droplet_fit").volume
    ts = get_train_timestamps(run_series[0], db)
    v = v0.assign_coords(time = ("trainId", ts))
    for run in run_series[1:]:
        _v = xr.open_dataset(db[run].file, group="droplet_fit").volume
        run = ex.RunDirectory(f"/gpfs/exfel/exp/MID/202601/p010400/raw/r{run:04d}")
        ts = run.train_timestamps()
        _v = _v.assign_coords(time = ("trainId", ts))
        v = xr.concat([v, _v], dim="trainId")
    return v

    
def gauss(x, mu, sigma, a):
    return a * np.exp(-(x-mu)**2/(2*sigma**2))


class FerritinProperties(NamedTuple):
    molar_mass: float  # g/mol, whole particle including core
    v_bar: float  # mL/g, mass-weighted shell + core
    iron_mass_fraction: float
    protein_mass_fraction: float  # apoferritin mass / whole-particle mass
 
 
def holoferritin_properties(
    n_fe: float = 1800.0,
    *,
    rho_core: float = RHO_CORE,
    m_apo: float = M_APOFERRITIN,
    v_bar_apo: float = V_BAR_APOFERRITIN,
) -> FerritinProperties:
    """Effective properties of iron-loaded ferritin at a given mean core loading.
 
    `n_fe` is the mean number of Fe atoms per molecule. Commercial horse spleen
    ferritin is typically 1600-2500; literature values span 1000-4500 and the
    distribution within one batch is broad. Measure it (ICP-MS, AAS, or A420/A280)
    rather than trusting this default.
    """
    m_core = n_fe * M_FEOOH
    m_total = m_apo + m_core
    v_bar = (m_apo * v_bar_apo + m_core / rho_core) / m_total
    return FerritinProperties(
        molar_mass=m_total,
        v_bar=v_bar,
        iron_mass_fraction=n_fe * M_FE / m_total,
        protein_mass_fraction=m_apo / m_total,
    )
 
 
class DropletComposition(NamedTuple):
    """Arrays broadcast to the shape of `volume`; scalars for the last three fields."""
 
    water_volume_fraction: NDArray[np.float64]
    water_mass_fraction: NDArray[np.float64]
    ferritin_mg_per_ml: NDArray[np.float64]  # whole-particle mass basis
    ferritin_molar: NDArray[np.float64]  # mol/L
    peg_mg_per_ml: NDArray[np.float64]
    peg_mg_per_ml_free_volume: NDArray[np.float64]  # per mL of PEG-accessible volume
    ferritin_volume_fraction_mass: NDArray[np.float64]  # from partial specific volume
    ferritin_volume_fraction_hs: NDArray[np.float64]  # from 12 nm hard spheres
    peg_volume_fraction: NDArray[np.float64]
    concentration_factor: NDArray[np.float64]
    volume_dry: float
    ferritin: FerritinProperties
 
 
def droplet_composition(
    volume: ArrayLike,
    volume_initial: float,
    *,
    ferritin_mg_per_ml_initial: float = 50.0,
    ferritin_basis: Literal["particle", "protein"] = "particle",
    peg_percent_wv_initial: float = 5.0,
    n_fe: float = 1800.0,
    d_ferritin_nm: float = D_FERRITIN_NM,
    v_bar_peg: float = V_BAR_PEG,
    rho_water: float = RHO_WATER_25C,
) -> DropletComposition:
    """Composition of a droplet that has evaporated from `volume_initial` to `volume`.
 
    Parameters
    ----------
    volume, volume_initial
        Volumes in mL (masses are reported per mL; only the ratio sets the fractions).
    ferritin_mg_per_ml_initial, ferritin_basis
        `"particle"`: the stated concentration is whole-ferritin mass, core included
        (what a gravimetric or vendor-lot figure usually means).
        `"protein"`: it is apoferritin mass only (what Lowry/Bradford/BCA report).
        The two differ by 1/protein_mass_fraction ~ 1.33 at 1800 Fe, which propagates
        directly into number density and every volume fraction below.
    n_fe
        Mean Fe atoms per molecule; sets the effective partial specific volume.
    """
    v = np.asarray(volume, dtype=np.float64)
    fer = holoferritin_properties(n_fe)
 
    c_fer0 = ferritin_mg_per_ml_initial * 1e-3  # g/mL, on the stated basis
    if ferritin_basis == "protein":
        c_fer0 /= fer.protein_mass_fraction  # convert to whole-particle mass
    c_peg0 = peg_percent_wv_initial * 10.0 * 1e-3  # g/mL
 
    mass_ferritin = c_fer0 * volume_initial
    mass_peg = c_peg0 * volume_initial
    vol_ferritin_mass = mass_ferritin * fer.v_bar
    vol_peg = mass_peg * v_bar_peg
    volume_dry = float(vol_ferritin_mass + vol_peg)
 
    n_particles = mass_ferritin / fer.molar_mass * N_A
    v_hs_ml = np.pi / 6.0 * (d_ferritin_nm * 1e-7) ** 3  # nm -> cm, cm^3 == mL
    vol_ferritin_hs = n_particles * v_hs_ml
 
    v_safe = np.where(v >= volume_dry, v, np.nan)
    volume_water = v_safe - volume_dry
    mass_water = rho_water * volume_water
    factor = volume_initial / v_safe
 
    phi_hs = vol_ferritin_hs / v_safe
    peg_nominal = peg_percent_wv_initial * 10.0 * factor
 
    return DropletComposition(
        water_volume_fraction=volume_water / v_safe,
        water_mass_fraction=mass_water / (mass_water + mass_ferritin + mass_peg),
        ferritin_mg_per_ml=c_fer0 * 1e3 * factor,
        ferritin_molar=n_particles / N_A / (v_safe * 1e-3),
        peg_mg_per_ml=peg_nominal,
        peg_mg_per_ml_free_volume=peg_nominal / (1.0 - phi_hs),
        ferritin_volume_fraction_mass=vol_ferritin_mass / v_safe,
        ferritin_volume_fraction_hs=phi_hs,
        peg_volume_fraction=vol_peg / v_safe,
        concentration_factor=factor,
        volume_dry=volume_dry,
        ferritin=fer,
    )


class DropletAttenuation(NamedTuple):
    """Arrays broadcast to the shape of `volume`."""

    v_over_v0: NDArray[np.float64]
    mu_per_mm: NDArray[np.float64]
    droplet_transmission: NDArray[np.float64]  # exp(-mu * chord)
    path_mm: NDArray[np.float64]  # chord * exp(-mu * chord)
    phi_ferritin: NDArray[np.float64]  # buffer volume fraction the ferritin displaces


def droplet_attenuation(
    volume: ArrayLike,
    volume_initial: float,
    chord_mm: ArrayLike,
    *,
    ferritin_mg_per_ml_initial: float = 50.0,
    peg_percent_wv_initial: float = 5.0,
    nacl_mM_initial: float = 150.0,
    n_fe: float = 1800.0,
) -> DropletAttenuation:
    """X-ray attenuation along the beam's path through an evaporating droplet.

    Every photon scattered at small angle inside a path of length t travels t through
    the droplet in total, so the droplet's measured I(q) is its cross-section per unit
    volume times `path_mm` = t * exp(-mu * t). That factor peaks at t = 1/mu, so a
    droplet thicker than one attenuation length scatters *less* per unit flux as it
    grows, and a subtraction between two droplets must scale each by its own.

    mu sums water, apoferritin, the FeOOH core, PEG and NaCl at the concentrations of
    `droplet_composition`, using MU_RHO_9P04_KEV. Every field is NaN where `volume` is
    below the dry volume.

    Parameters
    ----------
    volume, volume_initial
        Droplet volume and its value at injection, in the same unit.
    chord_mm
        Beam path through the droplet [mm], broadcast against `volume`.
    ferritin_mg_per_ml_initial, peg_percent_wv_initial, nacl_mM_initial
        Composition at `volume_initial`; ferritin on the whole-particle basis.
    n_fe
        Fe atoms per ferritin (see `holoferritin_properties`). The core dominates the
        attenuation, so this is the largest uncertainty in mu.
    """
    comp = droplet_composition(
        volume,
        volume_initial,
        ferritin_mg_per_ml_initial=ferritin_mg_per_ml_initial,
        peg_percent_wv_initial=peg_percent_wv_initial,
        n_fe=n_fe,
    )
    c_ferritin = comp.ferritin_mg_per_ml * 1e-3  # g/mL
    c_apo = c_ferritin * comp.ferritin.protein_mass_fraction
    c_nacl = nacl_mM_initial * 1e-6 * M_NACL * comp.concentration_factor  # g/mL
    mu_per_cm = (
        comp.water_volume_fraction * RHO_WATER_25C * MU_RHO_9P04_KEV["H2O"]
        + c_apo * MU_RHO_9P04_KEV["protein"]
        + (c_ferritin - c_apo) * MU_RHO_9P04_KEV["FeOOH"]
        + comp.peg_mg_per_ml * 1e-3 * MU_RHO_9P04_KEV["PEG"]
        + c_nacl * MU_RHO_9P04_KEV["NaCl"]
    )
    mu = mu_per_cm / 10.0
    chord = np.asarray(chord_mm, dtype=np.float64)
    transmission = np.exp(-mu * chord)
    return DropletAttenuation(
        v_over_v0=1.0 / comp.concentration_factor,
        mu_per_mm=mu,
        droplet_transmission=transmission,
        path_mm=chord * transmission,
        phi_ferritin=comp.ferritin_volume_fraction_mass,
    )


def train_pulse_energy(raw, n_pulses: int) -> xr.DataArray:
    """XGM pulse energy [uJ] per train, averaged over the first `n_pulses` pulses.

    Per train, not per frame, on purpose: at 2.26 MHz the per-pulse XGM reads the first
    ~20 pulses of a train 15-20 % high while the detector's own signal per pulse is
    flat (r370, r464), so dividing frame by frame writes that ramp into I(q). Two runs
    with different pulse counts therefore share a flux scale only over the same pulse
    ranks, which is what `n_pulses` fixes. A train missing any of those pulses is NaN
    rather than an average over fewer.

    `raw` is a raw (or "all") DataCollection: the XGM is a control source, which proc
    does not carry.
    """
    from extra.components import XGM

    pulse_energy = XGM(raw).pulse_energy()
    n_available = pulse_energy.sizes["pulseIndex"]
    if n_available < n_pulses:
        raise ValueError(
            f"the XGM has {n_available} pulses per train, fewer than {n_pulses=}"
        )
    first = pulse_energy.isel(pulseIndex=slice(0, n_pulses))
    return first.mean("pulseIndex", skipna=False)


def droplet_scaling(
    run_nr: int,
    db,
    *,
    proposal: int = 10400,
    trains=None,
    n_pulses: int = 155,
    v0_run: int | None = None,
    raw=None,
    ferritin_mg_per_ml_initial: float = 50.0,
    peg_percent_wv_initial: float = 5.0,
    nacl_mM_initial: float = 150.0,
    n_fe: float = 1800.0,
) -> xr.Dataset:
    """Per-train factors that put one droplet's I(q) on a common scale.

    I(q) * scale is proportional to the droplet's scattering cross-section per unit
    volume, so a buffer droplet is subtracted from a sample droplet as

        D(q) = I_s(q) * scale_s - (1 - phi_ferritin_s) * I_b(q) * scale_b

    with both I(q) averaged over the same pulses, the first `n_pulses` of each train.
    Equivalently, the buffer's factor on unscaled curves is
    alpha = (1 - phi_s) * flux_s * path_s / (flux_b * path_b).

    Parameters
    ----------
    run_nr, db
        Run number and `damnit.Damnit` database (total_transmission, droplet_fit).
    trains
        Optional positional selection (int, slice, list) over the run's trains,
        applied after aligning every input on trainId. The result keeps its trainId
        labels, so arithmetic with an intensity grid aligns by label, not position.
    n_pulses
        Pulses per train the XGM is averaged over; see `train_pulse_energy`.
    v0_run
        Run whose first droplet volume is V0, i.e. the run the droplet was injected
        in. Defaults to `run_nr`, which is only right if that run starts a fresh
        droplet.
    raw
        An already-open raw DataCollection of `run_nr`; opened if not given.
    ferritin_mg_per_ml_initial, peg_percent_wv_initial, nacl_mM_initial, n_fe
        Composition at V0, as in `droplet_attenuation`. DAMNIT's `sample` label is
        not reliable for this; for a buffer droplet pass ferritin_mg_per_ml_initial=0.

    Returns
    -------
    Dataset over trainId: xgm_uJ, total_transmission, flux (their product),
    volume_mm3, chord_mm, the `DropletAttenuation` fields and scale. Trains missing
    any input are dropped; the inputs are recorded in attrs.
    """
    if raw is None:
        raw = ex.open_run(proposal, run_nr, data="raw")
    fit = xr.open_dataset(db[run_nr].file, group="droplet_fit")
    ds = xr.Dataset(
        {
            "xgm_uJ": train_pulse_energy(raw, n_pulses),
            "total_transmission": db[run_nr]["total_transmission"].read(),
            "volume_mm3": fit.volume,
            # Through an oblate droplet whose symmetry axis is vertical, assuming the
            # beam crosses its centre: twice the horizontal semi-axis (4/3 pi a_x^2 a_y
            # reproduces droplet_fit's volume).
            "chord_mm": 2 * fit.radius.sel(dim="x") * DROPLET_PX_MM,
        }
    ).drop_vars("dim", errors="ignore")
    ds = ds.dropna("trainId", how="any")  # the Dataset aligned with an outer join
    if trains is not None:
        ds = ds.isel(trainId=[trains] if np.isscalar(trains) else trains)

    if v0_run is None or v0_run == run_nr:
        v0_fit = fit
    else:
        v0_fit = xr.open_dataset(db[v0_run].file, group="droplet_fit")
    v0 = float(v0_fit.volume.dropna("trainId")[0])

    att = droplet_attenuation(
        ds.volume_mm3.values,
        v0,
        ds.chord_mm.values,
        ferritin_mg_per_ml_initial=ferritin_mg_per_ml_initial,
        peg_percent_wv_initial=peg_percent_wv_initial,
        nacl_mM_initial=nacl_mM_initial,
        n_fe=n_fe,
    )
    for name, values in att._asdict().items():
        ds[name] = ("trainId", values)
    ds["flux"] = ds.xgm_uJ * ds.total_transmission
    ds["scale"] = 1.0 / (ds.flux * ds.path_mm)
    ds.attrs.update(
        run=run_nr,
        v0_run=run_nr if v0_run is None else v0_run,
        v0_mm3=v0,
        n_pulses=n_pulses,
        ferritin_mg_per_ml_initial=ferritin_mg_per_ml_initial,
        peg_percent_wv_initial=peg_percent_wv_initial,
        nacl_mM_initial=nacl_mM_initial,
        n_fe=n_fe,
    )
    return ds


def volume_to_deff_sq(V):
    """Effective sphere-equivalent diameter squared, from volume."""
    D_eff = (6.0 * V / np.pi) ** (1.0 / 3.0)
    return D_eff ** 2


def deff_sq_to_volume(D2):
    """Inverse of volume_to_deff_sq: back to volume from D_eff^2."""
    D_eff = np.sqrt(D2)
    return (np.pi / 6.0) * D_eff ** 3

    
def piecewise_D2(t, D0_sq, k1, k2, t_star):
    """
    Continuous piecewise-linear D^2(t):
      t <= t_star : D0_sq - k1*t                (D2-law / evaporation regime)
      t >  t_star : D0_sq - k1*t_star - k2*(t-t_star)   (post-evaporation regime)
    k1, k2 are evaporation rate constants (>=0 for shrinking droplet).
    """
    t = np.asarray(t)
    branch1 = D0_sq - k1 * t
    branch2 = (D0_sq - k1 * t_star) - k2 * (t - t_star)
    return np.where(t <= t_star, branch1, branch2)

    
def fit_D2_law(t, V, t_star_guess=None, bounds=None):
    """
    t : array of times
    V : array of volumes (same length)
    Returns fit params (D0_sq, k1, k2, t_star) and covariance.
    """
    t = np.asarray(t, dtype=float)
    D2 = volume_to_deff_sq(np.asarray(V, dtype=float))

    if t_star_guess is None:
        t_star_guess = t[len(t) // 2]

    p0 = [D2[0], 
          (D2[0] - D2[len(D2)//2]) / (t_star_guess - t[0] + 1e-12),
          1e-3,
          t_star_guess]

    if bounds is None:
        bounds = ([0, 0, -np.inf, t.min()],
                  [np.inf, np.inf, np.inf, t.max()])

    popt, pcov = curve_fit(piecewise_D2, t, D2, p0=p0, bounds=bounds, maxfev=20000)
    return popt, pcov


def style_plot(ax: plt.Axes = None, ylabel: str = r"$I(q)~$(a.u.)", xlabel: str = r"$q~$(nm$^{-1}$)", fig_legend: bool = False) -> None:
    ax.set_ylabel(ylabel, fontsize=14)
    ax.set_xlabel(xlabel, fontsize=14)
    handles, labels = ax.get_legend_handles_labels()
    if len(handles)>=1 and len(labels)>=1:
        if not fig_legend:
            ax.legend(handles, labels, fontsize=12)
        else:
            fig = ax.get_figure()
            fig.legend(handles, labels, fontsize=12, loc=7)
            fig.subplots_adjust(right=0.75)
    ax.tick_params(labelsize=12)
    ax.grid(which="major", ls="-", alpha=0.3)
    ax.grid(which="minor", ls=":", alpha=0.1)
    return

    
def log_interpolate(orig_x:np.ndarray, orig_y:np.ndarray, bins=500) -> tuple[np.ndarray, np.ndarray]:
    new_x_log = np.logspace(np.log10(orig_x.min()), np.log10(orig_x.max()), num=bins)
    interpolator = interp1d(orig_x, orig_y, kind='linear')
    new_y_log = interpolator(new_x_log)
    return new_x_log, new_y_log