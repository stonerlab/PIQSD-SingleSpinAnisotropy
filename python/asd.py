import numpy as np
from numba import njit
from scipy import constants as scp
import analytic

muB = scp.value("Bohr magneton")  # J T^-1
g_factor = np.fabs(scp.value("electron g factor"))  # dimensionless
gyro = scp.value("electron gyromag. ratio") * 1e-9  # in rad GHz T^-1
kB = scp.k  # J K^-1
hbar = scp.hbar

g_muB_by_kB = g_factor * muB / kB


@njit
def rescale_spin(spin):
    """Returns spin renormalised to a unit vector"""
    return spin / np.linalg.norm(spin)


@njit
def random_field(time_step, temperature, alpha, quantum_spin):
    """Returns the stochastic thermal field which obeys the statistical properties

    〈ηᵢ(t)〉= 0
    〈ηᵢ(t) ηⱼ(t')〉= 2 α δᵢⱼ δ(t−t') / (β g μ_B s γ)

    corresponding to a classical white noise.

    Note the time step appears on the bottom below because we multiply by the time step
    in the numerical integration, so the overall effect is √(Δt) for the Wiener process.
    """
    gamma = np.sqrt((2 * kB * alpha * temperature)
                    / (time_step * g_factor * muB * quantum_spin * gyro))
    return np.random.normal(0, gamma, 3)


@njit
def spin_advance_symplectic(spin, field, time_step, temperature, alpha, quantum_spin):
    """Given an initial spin s(t), returns s(t+dt) by symplectic integration
     of the damped precession around the effective field
     """
    effective_field = field(spin, temperature) + random_field(time_step, temperature, alpha, quantum_spin)

    effective_precession = (gyro / (1 + alpha ** 2)) * (effective_field + alpha * np.cross(spin, effective_field))

    torque = np.cross(effective_precession, spin)
    energy = np.dot(effective_precession, spin)

    precession_norm = np.linalg.norm(effective_precession)

    norm_by_timestep = precession_norm * time_step
    energy_over_norm = energy / precession_norm
    cos_precession = np.cos(norm_by_timestep)
    sin_precession = np.sin(norm_by_timestep)

    return cos_precession * spin + ((sin_precession * torque) + energy_over_norm * (
                1.0 - cos_precession) * effective_precession) / precession_norm


@njit
def rhs_runge_kutta_4(spin, field, noise, alpha):
    """Returns the RHS of Landau-Lifshitz-Gilbert equation for RK4 integration
    """
    torque = np.cross(spin, field + noise)
    damping = np.cross(spin, torque)

    rhs = -(gyro / (1 + alpha ** 2)) * (torque + alpha * damping)

    return rhs


@njit
def spin_advance_runge_kutta_4(spin, field, time_step, temperature, alpha, quantum_spin):
    """Given an initial spin s(t), returns s(t+dt) by RK4 integration
     of the damped precession around the effective field
     """
    noise = random_field(time_step, temperature, alpha, quantum_spin)

    rk_step_1 = rhs_runge_kutta_4(spin, field(spin, temperature), noise, alpha)
    spin_step_1 = rescale_spin(spin + (time_step / 2) * rk_step_1)

    rk_step_2 = rhs_runge_kutta_4(spin_step_1, field(spin_step_1, temperature), noise, alpha)
    spin_step_2 = rescale_spin(spin + (time_step / 2) * rk_step_2)

    rk_step_3 = rhs_runge_kutta_4(spin_step_2, field(spin_step_2, temperature), noise, alpha)
    spin_step_3 = rescale_spin(spin + time_step * rk_step_3)

    rk_step_4 = rhs_runge_kutta_4(spin_step_3, field(spin_step_3, temperature), noise, alpha)

    new_spin = spin + time_step * (rk_step_1 + 2 * rk_step_2 + 2 * rk_step_3 + rk_step_4) / 6

    return rescale_spin(new_spin)


def solver_factory(method, approximation, order, quantum_spin, a_1, a_2, alpha, time_step):
    """Returns the atomistic solver corresponding to the method of integration and approximation
    for the computation of the effective field
    """
    if approximation == 'classical-limit':
        field_from_hamiltonian = analytic.generate_field_function(quantum_spin, 2)

        @njit
        def field_function(spin, temperature):
            return np.array(([0, 0, field_from_hamiltonian(1.0/(kB * temperature), a_2, a_1, spin[2], g_factor, muB)]))
    elif approximation == "quantum-approximation":
        field_from_hamiltonian = analytic.generate_field_function(quantum_spin, order)

        @njit
        def field_function(spin, temperature):
            return np.array(([0, 0, field_from_hamiltonian(1.0/(kB * temperature), a_2, a_1, spin[2], g_factor, muB)]))
    elif approximation == "quantum-exact":
        field_from_hamiltonian = analytic.generate_field_function_exact(quantum_spin)

        @njit
        def field_function(spin, temperature):
            return np.array(([0, 0, field_from_hamiltonian(1.0/(kB * temperature), a_2, a_1, spin[2], g_factor, muB)]))
    else:
        raise RuntimeError(f'Unknown approximation: {approximation}')

    if method == 'runge-kutta-4':
        @njit
        def solver_function(spin, temperature):
            return spin_advance_runge_kutta_4(
                spin, field_function, time_step, temperature, alpha, quantum_spin)
    elif method == 'symplectic':
        @njit
        def solver_function(spin, temperature):
            return spin_advance_symplectic(
                spin, field_function, time_step, temperature, alpha, quantum_spin)
    else:
        raise RuntimeError(f'Unknown integrator: {method}')

    return solver_function


# Result computation
@njit
def calculate_sz_asd(solver, spin_initial, temperature, num_eq_steps, num_production_steps,
                     num_realisations):
    """Returns the value of the expectation value of the z-component of the spin by averaging over
    time and realisations of the noise
    """
    sz_realisations = 0.0
    for _ in range(num_realisations):
        # Incase the initial spin is not properly normalised
        spin = rescale_spin(spin_initial)

        for _ in range(0, num_eq_steps):
            spin = solver(spin, temperature)

        spin_z = 0.0
        for _ in range(0, num_production_steps):
            spin = solver(spin, temperature)
            spin_z += spin[2]

        sz_realisations += spin_z / num_production_steps

    return sz_realisations / num_realisations


@njit
def compute_temperature_dependence(solver, temperatures, low_high_t, quantum_spin, time_step,
                                   equilibration_time, production_time, num_realisation,
                                   spin_initial):
    """Returns an array of expectation values of the z-component of the spin corresponding to the
    input temperatures"""
    sz_expectation = np.zeros(np.shape(temperatures))
    i = 0

    if low_high_t in {'high-temperature-first-order', 'high-temperature-second-order', 'high-temperature-9th-order'}:
        renormalisation = (quantum_spin + 1.0) / quantum_spin
    else:
        renormalisation = 1

    for temperature in temperatures:
        sz_expectation[i] = renormalisation * calculate_sz_asd(solver, spin_initial, temperature,
                                                   int(equilibration_time / time_step)
                                                   , int(production_time / time_step),
                                                   num_realisation)
        i += 1

    return sz_expectation


def save_to_file(file_name, x_data, y_data):
    """Saves numpy arrays x and y to specified file"""
    np.savetxt(file_name, np.column_stack((x_data, y_data)), fmt='%.8e')
