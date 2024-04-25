import asd
import os
import numpy as np
import matplotlib.pyplot as plt
import time
from fractions import Fraction

# local imports
import analytic


def main():
    plt.style.use('../resources/aps-paper.mplstyle')
    data_path = '../figures/figure3_data'
    os.makedirs(data_path, exist_ok=True)

    # Simulation conditions
    quantum_spin = 5/2

    stress = 0
    field = 1
    K = -2 * asd.g_factor * asd.muB * field
    temperatures = np.linspace(0.05, 100, 200)
    a_0 = stress
    a_1 = asd.g_factor * asd.muB * field
    a_2 = K - a_0

    # --- calculate and save exact quantum solution ---
    quantum_solution = analytic.quantum_state_sz(quantum_spin, temperatures, a_0, a_1, a_2)
    np.savetxt(f"{data_path}/analytical_quantum_state_solution_s{quantum_spin:.1f}.tsv",
               np.column_stack((temperatures, quantum_solution)), fmt='%.8e',
               header='temperature_kelvin sz-expectation_hbar')

    quantum_solution_square = analytic.quantum_state_sz_square(quantum_spin, temperatures, a_0, a_1, a_2)
    np.savetxt(f"{data_path}/analytical_quantum_state_square_solution_s{quantum_spin:.1f}.tsv",
               np.column_stack((temperatures, quantum_solution)), fmt='%.8e',
               header='temperature_kelvin sz-expectation-square_hbar')

    second_order_moment = analytic.quantum_state_sz_second_order_moment(quantum_spin, temperatures, a_0, a_1, a_2)
    np.savetxt(f"{data_path}/analytical_quantum_state_variance_solution_s{quantum_spin:.1f}.tsv",
               np.column_stack((temperatures, quantum_solution)), fmt='%.8e',
               header='temperature_kelvin sz-square-expectation_hbar')

    result = np.sqrt(np.abs(quantum_solution_square-second_order_moment)) / (quantum_solution * quantum_spin)
    # result2 = second_order_moment / second_order_moment[0]

    # --- plotting ---
    plt.plot(temperatures, quantum_solution, label='quantum solution', color="red")
    plt.plot(temperatures, result, linestyle=(0, (4, 6)), label='thermal fluctuations', color="blue")
    # plt.plot(temperatures, result2, linestyle=(0, (4, 6)), label='thermal fluctuations', color="blue")



    plt.xlabel(r"$T$ (K)")
    plt.ylabel(r"$\langle\hat{S}_z\rangle/s$ ($\hbar$)")
    plt.legend(title=rf'$s={str(Fraction(quantum_spin))}$')

    plt.show()
    # plt.savefig('../figures/figure3.pdf', transparent=True)


if __name__ == "__main__":
    start = time.process_time()
    main()
    end = time.process_time()
    print(f'runtime: {end - start:.3f} (s)')