import asd
import os
import numpy as np
import matplotlib.pyplot as plt
import time
from fractions import Fraction

# local imports
import analytic


def main():
    plt.style.use('resources/aps-paper.mplstyle')
    data_path = 'figures/figure_data'
    os.makedirs(data_path, exist_ok=True)

    integrator = 'symplectic'

    # Simulation conditions
    alpha = 0.5  # Gilbert Damping parameter.
    s0 = np.array([1 / np.sqrt(3), 1.0 / np.sqrt(3), -1.0 / np.sqrt(3)])  # Initial spin
    quantum_spin = 1/2
    order = 3
    stress = 0
    field = 1
    K = -2 * asd.g_factor * asd.muB * field

    temperatures = np.linspace(0.02, 10, 200)

    # Equilibration time, final time and time step
    num_realisation = 20
    equilibration_time = 5  # Equilibration time ns
    production_time = 15  # Final time ns
    time_step = 0.00005  # Time step ns, "linspace" so needs to turn num into int

    a_0 = stress
    a_1 = asd.g_factor * asd.muB * field
    a_2 = K - a_0

    # --- calculate and save exact quantum solution ---
    quantum_solution = analytic.quantum_state_sz(quantum_spin, temperatures, a_0, a_1, a_2)
    np.savetxt(f"{data_path}/analytical_quantum_state_solution_s{quantum_spin:.1f}.tsv",
               np.column_stack((temperatures, quantum_solution)), fmt='%.8e',
               header='temperature_kelvin sz-expectation_hbar')

    # --- calculate approximate quantum result from classical approximation and classical limit to compare to ---
    asd_data_file_quantum = f'{data_path}/qsd_quantum-approximation_{order}_solution_s{quantum_spin:.1f}.tsv'

    if order == 2:
        normalisation = 1
    else:
        normalisation = (quantum_spin + 1) / quantum_spin

    if os.path.exists(asd_data_file_quantum):
        temperatures, sz_quantum = np.loadtxt(asd_data_file_quantum, unpack=True)
    else:
        solver_quantum = asd.solver_factory(integrator, 'quantum-approximation', order, quantum_spin, a_1, a_2, alpha, time_step)
        sz_quantum = asd.compute_temperature_dependence(solver_quantum, temperatures, 'quantum-approximation', quantum_spin, time_step,
                                                equilibration_time, production_time, num_realisation, s0)

        np.savetxt(f"{data_path}/qsd_quantum-approximation_{order}_solution_s{quantum_spin:.1f}.tsv",
                   np.column_stack((temperatures, sz_quantum)), fmt='%.8e',
                   header='temperature_kelvin sz-expectation_hbar')

    asd_data_file_classical = f'{data_path}/qsd_classical-limit_solution_s{quantum_spin:.1f}.tsv'
    if os.path.exists(asd_data_file_classical):
        temperatures, sz_classical = np.loadtxt(asd_data_file_classical, unpack=True)
    else:
        solver_classical = asd.solver_factory(integrator, 'classical-limit', order, quantum_spin, a_1, a_2, alpha, time_step)
        sz_classical = asd.compute_temperature_dependence(solver_classical, temperatures, 'classical-limit', quantum_spin, time_step,
                                                equilibration_time, production_time, num_realisation, s0)

        np.savetxt(f"{data_path}/qsd_classical-limit_solution_s{quantum_spin:.1f}.tsv",
                   np.column_stack((temperatures, sz_classical)), fmt='%.8e',
                   header='temperature_kelvin sz-expectation_hbar')

    # --- plotting ---
    plt.plot(temperatures, quantum_solution, label='quantum solution', color="red")
    plt.plot(temperatures, sz_quantum*normalisation, linestyle=(0, (4, 6)), label='$1^{st}$ quantum correction', color="#FF9900")
    plt.plot(temperatures, sz_classical, linestyle=(0, (4, 6)), label='classical limit', color="black")

    plt.xlabel(r"$T$ (K)")
    plt.ylabel(r"$\langle\hat{S}_z\rangle/s$ ($\hbar$)")
    plt.legend(title=rf'$s={str(Fraction(quantum_spin))}$')

    # plt.show()
    plt.savefig('figures/figure_a.pdf', transparent=True)


if __name__ == "__main__":
    start = time.process_time()
    main()
    end = time.process_time()
    print(f'runtime: {end - start:.3f} (s)')
