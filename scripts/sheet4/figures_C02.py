"""Figures for the companion notebook C02 (neutrino decoupling).

Plotting code only; contains no physics that is not in the notebook.
Every quantity shown is computed in the notebook and passed in as an argument.
"""
import numpy as np
import matplotlib.pyplot as plt

# Okabe-Ito colours (colour-blind friendly)
BLUE, ORANGE, GREEN, PINK, YELLOW, SKY, BLACK = ('#0072B2', '#D55E00', '#009E73', '#CC79A7',
                                                 '#E69F00', '#56B4E9', '#000000')
COLORS = [BLUE, ORANGE, GREEN, PINK, YELLOW, SKY]

STYLE = {'font.size': 14, 'axes.labelsize': 15, 'axes.titlesize': 15, 'legend.fontsize': 12,
         'xtick.labelsize': 13, 'ytick.labelsize': 13, 'lines.linewidth': 2.5,
         'figure.dpi': 100}


def plot_chain_check(chains, background):
    """p*a along chains of observers with different spacings.

    chains: list of (dchi, a, pa) with pa normalised to its start value.
    background: text naming the expansion law, used in the title.
    """
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(8, 5))
        for i, (dchi, a, pa) in enumerate(chains):
            ax.plot(a, pa, color=COLORS[i], label=f'spacing δχ = {dchi:g}')
        ax.axhline(1, color=BLACK, ls=':', lw=1.5, label='p a = const, eq. (5.3)')
        ax.set_xscale('log')
        ax.set_xlabel('scale factor a / a_start')
        ax.set_ylabel('p a / (p a)_start')
        ax.set_title(f'Photon handed along a chain of observers ({background})')
        ax.legend(loc='lower left')
        fig.tight_layout()
        plt.show()


def plot_massive_particle(a, p_over_m, v, p_over_m_expected, a_nr, v_photon):
    """Momentum and velocity of a massive particle along the chain of observers.

    a_nr: scale factor at which p = m (drawn as a vertical line).
    v_photon: velocity of a massless particle, drawn for comparison.
    """
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(8, 5.5))
        ax.loglog(a, p_over_m, color=BLUE, label='momentum p / m (chain of observers)')
        ax.loglog(a, p_over_m_expected, color=BLACK, ls=':', lw=2, label='∝ 1/a, eq. (5.3)')
        ax.loglog(a, v, color=ORANGE, label='velocity v / c (chain of observers)')
        ax.loglog(a, v_photon, color='gray', lw=4, alpha=0.5, zorder=1, label='velocity of a photon')
        ax.axvline(a_nr, color='gray', ls='--', lw=1.2)
        ax.text(a_nr * 1.1, 0.9 * p_over_m.max(), 'p = m', color='gray', va='top')
        ax.set_xlabel('scale factor a / a_start')
        ax.set_ylabel('p / m   and   v / c')
        ax.set_title('A massive free particle: p ∝ 1/a, but v is not')
        ax.legend(loc='lower left')
        fig.tight_layout()
        plt.show()


def plot_free_streaming(p, occupations, x_kin, thermometer, mass_label):
    """Two panels: (left) occupation number vs momentum at several times;
    (right) thermometer test ln(1/F - 1) vs kinetic energy / T_nu.

    occupations: list of (label, F(p)) for the left panel.
    thermometer: list of (label, y(x_kin)) for the right panel.
    mass_label: text for the right-panel title.
    """
    with plt.rc_context(STYLE):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
        for i, (label, F) in enumerate(occupations):
            ax1.semilogx(p, F, color=COLORS[i], label=label)
        ax1.set_xlabel(r'momentum $p\,/\,T_D$')
        ax1.set_ylabel('occupation number F')
        ax1.set_title('No collisions: the distribution slides to lower p')
        ax1.legend()

        ax2.plot(x_kin, x_kin, color=BLACK, lw=5, alpha=0.25, label=r'thermal distribution at $T_\nu$')
        for i, (label, y) in enumerate(thermometer):
            ax2.plot(x_kin, y, color=COLORS[i + 2], label=label)
        ax2.set_xlim(x_kin[0], x_kin[-1])
        ax2.set_ylim(0, 3 * x_kin[-1])
        ax2.set_xlabel(r'kinetic energy $(E - m)\,/\,T_\nu$')
        ax2.set_ylabel(r'$\ln(1/F - 1)$')
        ax2.set_title(f'Thermometer test, {mass_label}')
        ax2.legend(loc='upper left', fontsize=11)
        fig.tight_layout()
        plt.show()


def plot_decoupling(T, prob, T_marks, t_of_T, T_of_t):
    """Probability of at least one more collision after temperature T.

    T_marks: list of (T, text) drawn as vertical lines.
    t_of_T, T_of_t: conversion functions for the time axis on top.
    """
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(9, 5.8))
        ax.semilogx(T, prob, color=BLUE)
        for i, (Tm, text) in enumerate(T_marks):
            ax.axvline(Tm, color='gray', ls='--', lw=1.2)
            ax.text(Tm * 0.96, 0.97 - 0.09 * i, text, color='gray', ha='left', va='top', fontsize=12)
        ax.set_xlim(T[0], T[-1])
        ax.set_ylim(0, 1.02)
        ax.set_xlabel('temperature T [MeV]')
        ax.set_ylabel('probability of ≥ 1 more collision')
        sec = ax.secondary_xaxis('top', functions=(t_of_T, T_of_t))
        sec.set_xlabel('time t [s]')
        ax.set_title('How sharp is neutrino decoupling?', pad=12)
        fig.tight_layout()
        plt.show()


def plot_annihilation(T, ratio_curves, ratio_limit, m_e, g_tot, q_tot, after_values, T_D_used):
    """Two panels: (left) T_gamma/T_nu vs T for several decoupling temperatures;
    (right) g_tot and q_tot vs T.

    ratio_curves: list of (T_D, ratio(T)); the first is drawn thick.
    ratio_limit: (11/4)^(1/3), drawn as a dotted line.
    after_values: list of (value, text, above) for horizontal guide lines in the right panel;
        above=True puts the text above the line.
    T_D_used: decoupling temperature used in the right panel (for the title).
    """
    with plt.rc_context(STYLE):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.6))
        for i, (T_D, ratio) in enumerate(ratio_curves):
            ax1.semilogx(T, ratio, color=COLORS[i], lw=4 if i == 0 else 2,
                         label=rf'$T_D$ = {T_D:g} MeV')
        ax1.axhline(ratio_limit, color=BLACK, ls=':', lw=1.5)
        ax1.text(T[-1] * 1.5, ratio_limit + 0.008, r'$(11/4)^{1/3}$', ha='right')
        ax1.axvline(m_e, color='gray', ls='--', lw=1.2)
        ax1.text(m_e * 0.92, 1.25, r'$T = m_e$', color='gray', ha='left')
        ax1.set_xlim(T[0], T[-1])
        ax1.set_ylim(0.98, 1.46)
        ax1.set_xlabel(r'photon temperature $T_\gamma$ [MeV]')
        ax1.set_ylabel(r'$T_\gamma\,/\,T_\nu$')
        ax1.set_title('Photons are heated, neutrinos are not')
        ax1.legend(loc='upper left', title='neutrinos decouple at', fontsize=11, title_fontsize=11)

        ax2.semilogx(T, g_tot, color=BLUE, label=r'$g_{\rm tot}$, energy density (5.13)')
        ax2.semilogx(T, q_tot, color=ORANGE, label=r'$q_{\rm tot}$, entropy density (5.15)')
        for value, text, above in after_values:
            ax2.axhline(value, color='gray', ls=':', lw=1.2)
            ax2.text(T[0] * 0.8, value + (0.15 if above else -0.15), text, color='gray', ha='left',
                     va='bottom' if above else 'top', fontsize=12)
        ax2.set_xlim(T[0], T[-1])
        ax2.set_ylim(0, 12)
        ax2.set_xlabel(r'photon temperature $T_\gamma$ [MeV]')
        ax2.set_ylabel('effective number of degrees of freedom')
        ax2.set_title(rf'Degrees of freedom ($T_D$ = {T_D_used:g} MeV)')
        ax2.legend(loc='upper right')
        fig.tight_layout()
        plt.show()
