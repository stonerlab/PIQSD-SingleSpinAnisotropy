import numpy as np
from sympy import *
from numba import njit
import asd
u = Symbol('|z|', real=True)
p = Symbol('p', real=True)
s = Symbol('s', real=True)
B = Symbol('beta', real=True)
C = Symbol('A_2', real=True)
D = Symbol('A_1', real=True)
n = Symbol('n_z', real=True)
o = Symbol('g', real=True)
m = Symbol('\mu_B', real=True)


def quantum_state_sz(quantum_spin, temperature, a_0, a_1, a_2):
    """Returns the normalised expectation value of z component of spin computed by

    <S_z>=(1/s)*(Sum_[m=-s..s] <s,m|m*exp(beta * (A_0 + A_1 * m + A_2 * m^2))|s,m>)
                                            /(Sum_[m=-s..s] <s,m|exp(beta * (A_0 + A_1 * m + A_2 * m^2))|s,m>)
    """
    denominator = np.zeros(np.array(temperature).shape)
    for q_m in np.arange(-quantum_spin, quantum_spin + 1):
        denominator = denominator + np.exp((a_0 + a_1 * q_m + a_2 * q_m**2) / (asd.kB * temperature))

    num = np.zeros(np.array(temperature).shape)
    for q_m in np.arange(-quantum_spin, quantum_spin + 1):
        num = num + q_m * np.exp((a_0 + a_1 * q_m + a_2 * q_m**2) / (asd.kB * temperature))

    return (1.0/quantum_spin) * num / denominator


def quantum_state_sz_square(quantum_spin, temperature, a_0, a_1, a_2):
    """Returns the expectation value of z component of spin computed by

    <S_z>^2=[(Sum_[m=-s..s] <s,m|m*exp(beta * (A_0 + A_1 * m + A_2 * m^2))|s,m>)
                                            /(Sum_[m=-s..s] <s,m|exp(beta * (A_0 + A_1 * m + A_2 * m^2))|s,m>)]^2
    """
    denominator = np.zeros(np.array(temperature).shape)
    for q_m in np.arange(-quantum_spin, quantum_spin + 1):
        denominator = denominator + np.exp((a_0 + a_1 * q_m + a_2 * q_m**2) / (asd.kB * temperature))

    num = np.zeros(np.array(temperature).shape)
    for q_m in np.arange(-quantum_spin, quantum_spin + 1):
        num = num + q_m * np.exp((a_0 + a_1 * q_m + a_2 * q_m**2) / (asd.kB * temperature))

    return num*num / (denominator*denominator)


def quantum_state_sz_second_order_moment(quantum_spin, temperature, a_0, a_1, a_2):
    """Returns the expectation value of z component of spin computed by

    <(S_z)^2>=(Sum_[m=-s..s] <s,m|m**2 * exp(beta * (A_0 + A_1 * m + A_2 * m^2))|s,m>)
                                            /(Sum_[m=-s..s] <s,m|exp(beta * (A_0 + A_1 * m + A_2 * m^2))|s,m>)
    """
    denominator = np.zeros(np.array(temperature).shape)
    for q_m in np.arange(-quantum_spin, quantum_spin + 1):
        denominator = denominator + np.exp((a_0 + a_1 * q_m + a_2 * q_m**2) / (asd.kB * temperature))

    num = np.zeros(np.array(temperature).shape)
    for q_m in np.arange(-quantum_spin, quantum_spin + 1):
        num = num + q_m**2 * np.exp((a_0 + a_1 * q_m + a_2 * q_m**2) / (asd.kB * temperature))

    return num / denominator


def l_function(two_s):
    """Returns a sympy function resulting from expression (12) of the z-dependent part of the integrand
        of the partition function.
        """
    return simplify(expand(summation(binomial(two_s, p) * u**(2*p) * exp(B * C * p**2)
                                     * exp(-B * (two_s*C + D)*p), (p, 0, two_s))))


def integrand_exponent(two_s):
    """Returns the exponent of the integrand of the partition function written as an exponential using
        the ln function.
        """
    return simplify(expand(-ln(l_function(two_s))+(two_s*ln(1+u**2))))


def eff_hamiltonian(two_s, order):
    """Returns a sympy polynomial approximation in powers of beta of the exponential
        approximation of the integrand of the partition function
        """
    return simplify(series(integrand_exponent(two_s), B, 0, order, dir='+').removeO()/B)


def eff_hamiltonian_classical(two_s, order):
    """Returns a converted version of the effective Hamiltonian in terms of the z-component of the
        spin coherent state vector n.
        """
    return simplify(eff_hamiltonian(two_s, order).subs(u**2, (1-n)/(1+n)))


def eff_field_formal(two_s, order):
    """Returns the effective field corresponding to the effective Hamiltonian.
        """
    return simplify((-2/(two_s*o*m)) * diff(eff_hamiltonian_classical(two_s, order), n))


def generate_field_function(quantum_spin, order):
    """Casts the sympy expression from the method eff_field_formal into a njit python function
    with variables B = beta,  C = A_2,  D = A_1,  n = n_z,  o = g,  m = mu_B.

    Usable as:

    effective_field = generate_field_function(quantum_spin, order)
    effective_field(beta, A_2, A_1, n_z, g, mu_B)

    """
    g = lambdify([B, C, D, n, o, m], eff_field_formal(int(quantum_spin*2), order), 'numpy')
    return njit(g)

# Exact test

def eff_hamiltonian_exact(two_s):
    """Returns a sympy polynomial approximation in powers of beta of the exponential
        approximation of the integrand of the partition function
        """
    return integrand_exponent(two_s)/B


def eff_hamiltonian_classical_exact(two_s):
    """Returns a converted version of the effective Hamiltonian in terms of the z-component of the
        spin coherent state vector n.
        """
    return simplify(eff_hamiltonian_exact(two_s).subs(u**2, (1-n)/(1+n)))


def eff_field_formal_exact(two_s):
    """Returns the effective field corresponding to the effective Hamiltonian.
        """
    return simplify((-2/(two_s*o*m)) * diff(eff_hamiltonian_classical_exact(two_s), n))


def generate_field_function_exact(quantum_spin):
    """Casts the sympy expression from the method eff_field_formal into a njit python function
    with variables B = beta,  C = A_2,  D = A_1,  n = n_z,  o = g,  m = mu_B.

    Usable as:

    effective_field = generate_field_function_exact(quantum_spin)
    effective_field(beta, A_2, A_1, n_z, g, mu_B)

    """
    g = lambdify([B, C, D, n, o, m], eff_field_formal_exact(int(quantum_spin*2)), 'numpy')
    return njit(g)




def main():
    print_latex(eff_field_formal(1, 2))
    print_latex(eff_field_formal(2, 2))
    print_latex(eff_field_formal(3, 2))
    print_latex(eff_field_formal(4, 2))


if __name__ == "__main__":
    main()
