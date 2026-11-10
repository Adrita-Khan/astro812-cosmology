# %% [markdown]
# Figures for the lecture notebook L01 - From the Peacock diagram to the horizon problem
#
# Plotting code only; contains no physics that is not in the notebook.
# Every function receives everything it draws as arguments; nothing is read from
# variables of the notebook. Colours: Okabe-Ito palette (colour-blind friendly).

# %%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch, Polygon

# %%
COLORS = ['#0072B2', '#D55E00', '#009E73', '#CC79A7', '#E69F00', '#56B4E9', '#000000']

# %%
_STYLE = {
    'font.size': 15,
    'axes.labelsize': 17,
    'axes.titlesize': 16,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 13,
    'lines.linewidth': 2.8,
    'axes.grid': True,
    'grid.alpha': 0.3,
}


# %%
def _draw_classification(ax, Om_grid, OL_grid, fate_code, z_max, z_max_levels, region_colors):
    """Coloured regions, flat line and z_max contours in one panel; returns the contour set (or None)."""
    ax.pcolormesh(Om_grid, OL_grid, fate_code, cmap=ListedColormap(region_colors), vmin=-0.5, vmax=2.5,
                  shading='nearest')
    flat_Om = np.array([Om_grid[0], Om_grid[-1]])
    ax.plot(flat_Om, 1 - flat_Om, '-.', color='#555555', lw=2.5)
    contours = None
    if np.any(np.isfinite(z_max)):
        contours = ax.contour(Om_grid, OL_grid, z_max, levels=list(z_max_levels), colors='black',
                              linewidths=1.8, linestyles=['solid', 'dashed', 'dotted'][:len(z_max_levels)])
    ax.set_xlim(Om_grid[0], Om_grid[-1])
    ax.set_ylim(OL_grid[0], OL_grid[-1])
    ax.set_xlabel(r'matter density $\Omega_m$')
    ax.set_ylabel(r'vacuum energy density $\Omega_\Lambda$')
    ax.grid(False)
    return contours


# %%
def plot_peacock(Om_grid, OL_grid, fate_code, z_max, models, zoom=None, z_max_levels=(1, 2, 6), extra_curve=None):
    """Classification of models in the (Om, OL) plane (Fig. 4.1 of the lecture notes).

    fate_code[i, j] for OL_grid[i], Om_grid[j]: 0 = Big Bang, expands forever;
    1 = Big Bang, recollapses; 2 = no Big Bang (bounce). z_max: maximum redshift of the
    bounce models (NaN elsewhere), drawn as contours. models: list of (label, Om, OL) to mark.
    zoom: optional (Om_grid, OL_grid, fate_code, z_max) of a finer grid, drawn as a second panel.
    extra_curve: optional (Om_values, OL_values, label), drawn as a red line in both panels.
    """
    region_colors = ['#cfe3f2', '#f6d2b8', '#bdbdbd']
    region_labels = ['Big Bang, expands forever', 'Big Bang, recollapses', 'no Big Bang (bounce)']
    level_styles = ['solid', 'dashed', 'dotted']
    with plt.rc_context(_STYLE):
        if zoom is None:
            fig, ax = plt.subplots(figsize=(11, 9))
            axes = [ax]
        else:
            fig, axes = plt.subplots(1, 2, figsize=(17, 8.5), gridspec_kw={'width_ratios': [1.45, 1]})
            ax = axes[0]
        _draw_classification(ax, Om_grid, OL_grid, fate_code, z_max, z_max_levels, region_colors)
        x_max = Om_grid[-1]
        for k, (label, Om, OL) in enumerate(models):
            ax.plot(Om, OL, 'o', ms=11, color=COLORS[k % len(COLORS)], mec='black', mew=1.2, zorder=5)
            left = Om > 0.85 * x_max
            ax.annotate(label, (Om, OL), textcoords='offset points', xytext=(-9 if left else 9, 7),
                        ha='right' if left else 'left', fontsize=13, zorder=6,
                        bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='none', alpha=0.75))
        ax.text(1.62, -0.25, 'closed ($K > 0$)', ha='left', va='center', fontsize=14)
        ax.text(1.38, -0.75, 'open ($K < 0$)', ha='right', va='center', fontsize=14)
        ax.set_title('Fate of Friedmann-Lemaitre models from the roots of (4.12)\n'
                     f'grid of {len(Om_grid)} x {len(OL_grid)} models')
        handles = [Patch(color=c, label=l) for c, l in zip(region_colors, region_labels)]
        handles.append(plt.Line2D([], [], color='#555555', ls='-.', lw=2.5, label=r'flat: $\Omega_m + \Omega_\Lambda = 1$'))
        for level, style in zip(z_max_levels, level_styles):
            handles.append(plt.Line2D([], [], color='black', ls=style, lw=1.8, label=f'bounce models with $z_{{max}} = {level:g}$'))
        if extra_curve is not None:
            for panel in axes:
                panel.plot(extra_curve[0], extra_curve[1], color='#D55E00', lw=3)
            handles.append(plt.Line2D([], [], color='#D55E00', lw=3, label=extra_curve[2]))
        if zoom is not None:
            Om_zoom, OL_zoom, code_zoom, z_max_zoom = zoom
            ax2 = axes[1]
            _draw_classification(ax2, Om_zoom, OL_zoom, code_zoom, z_max_zoom, z_max_levels, region_colors)
            for k, (label, Om, OL) in enumerate(models):
                if Om_zoom[0] <= Om <= Om_zoom[-1] and OL_zoom[0] <= OL <= OL_zoom[-1]:
                    ax2.plot(Om, OL, 'o', ms=11, color=COLORS[k % len(COLORS)], mec='black', mew=1.2, zorder=5)
            ax2.set_title(f'zoom on the models without Big Bang\ngrid of {len(Om_zoom)} x {len(OL_zoom)} models')
            ax.add_patch(plt.Rectangle((Om_zoom[0], OL_zoom[0]), Om_zoom[-1] - Om_zoom[0], OL_zoom[-1] - OL_zoom[0],
                                       fill=False, ec='black', lw=1.5))
            fig.legend(handles=handles, loc='lower center', ncol=3, framealpha=0.95, bbox_to_anchor=(0.5, -0.01))
            plt.tight_layout(rect=(0, 0.13, 1, 1))
        else:
            ax.legend(handles=handles, loc='upper right', framealpha=0.95)
            plt.tight_layout()
        plt.show()


# %%
def plot_expansion_histories(histories, Om_values, t0_flat, t0_open, planck_point, tau_range=(-2.5, 2.5),
                             a_limit=3.0):
    """Left: a(tau) with tau = H0 (t - t0) for several models (Fig. 4.2, top, of the lecture notes).
    Right: age t0 H0 as a function of Om for flat models and for OL = 0 (Fig. 4.2, bottom).

    histories: list of dicts with keys 'label', 'tau', 'a', 'end_tau', 'end_a' (Big Bang / Big Crunch,
    i.e. a reaching the stop value) and 'turn_tau', 'turn_a' (da/dtau = 0).
    planck_point: (Om, t0 H0) of the Planck model.
    """
    with plt.rc_context(_STYLE):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7), gridspec_kw={'width_ratios': [1.35, 1]})
        for k, history in enumerate(histories):
            color = COLORS[k % len(COLORS)]
            ax1.plot(history['tau'], history['a'], color=color, label=history['label'])
            ax1.plot(history['end_tau'], history['end_a'], 'o', color=color, ms=10, mec='black', zorder=5)
            ax1.plot(history['turn_tau'], history['turn_a'], 'D', color=color, ms=10, mec='black', zorder=5)
        ax1.plot([], [], 'o', color='gray', mec='black', ms=10, label='Big Bang / Big Crunch ($a \\to 0$)')
        ax1.plot([], [], 'D', color='gray', mec='black', ms=10, label='turning point ($\\dot a = 0$)')
        ax1.axvline(0, color='black', lw=1)
        ax1.axhline(1, color='black', lw=1)
        ax1.set_xlim(*tau_range)
        ax1.set_ylim(0, a_limit)
        ax1.set_xlabel(r'time since today $H_0\,(t - t_0)$')
        ax1.set_ylabel(r'scale factor $a$')
        ax1.set_title('Expansion histories from the second Friedmann equation\n(all curves pass through $a = 1$ with slope 1 today)')
        ax1.legend(loc='upper left', fontsize=12, framealpha=0.95)

        ax2.plot(Om_values, t0_flat, color=COLORS[0], label=r'flat: $\Omega_\Lambda = 1 - \Omega_m - \Omega_r$')
        ax2.plot(Om_values, t0_open, '--', color=COLORS[1], label=r'$\Omega_\Lambda = 0$')
        ax2.plot(*planck_point, '*', color='black', ms=20, label='Planck 2018', zorder=5)
        ax2.axhline(1, color='gray', lw=1.5, ls=':')
        ax2.axhline(2 / 3, color='gray', lw=1.5, ls=':')
        ax2.set_xlabel(r'matter density $\Omega_m$')
        ax2.set_ylabel(r'age today $t_0\,H_0$')
        ax2.set_title('Age in units of the Hubble time $1/H_0$')
        ax2.legend(loc='upper right')
        plt.tight_layout()
        plt.show()


# %%
def plot_distances(z_planck, chi, D_ang, D_lum, z_models, D_ang_models, labels, dashed, z_peak, D_peak):
    """Left: comoving, angular-diameter and luminosity distance of one model in Gpc (log-log).
    Right: angular-diameter distance in units of c/H0 for several models (Fig. 4.3 of the lecture notes),
    with the maxima (z_peak, D_peak; NaN if there is none) marked; dashed[k] = True draws model k dashed."""
    with plt.rc_context(_STYLE):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        ax1.loglog(z_planck, chi, color=COLORS[0], label=r'comoving distance $\chi$')
        ax1.loglog(z_planck, D_ang, color=COLORS[1], label=r'angular-diameter distance $D_{ang}$')
        ax1.loglog(z_planck, D_lum, color=COLORS[2], label=r'luminosity distance $D_{lum}$')
        ax1.set_xlabel('redshift $z$')
        ax1.set_ylabel('distance [Gpc]')
        ax1.set_title('Three distances in the Planck 2018 model')
        ax1.legend(loc='upper left')
        ax1.grid(True, which='both', alpha=0.3)

        for k, (D, label) in enumerate(zip(D_ang_models, labels)):
            style = '--' if dashed[k] else '-'
            ax2.plot(z_models, D, style, color=COLORS[k % len(COLORS)], label=label)
            if np.isfinite(z_peak[k]):
                ax2.plot(z_peak[k], D_peak[k], 'o', color=COLORS[k % len(COLORS)], ms=11, mec='black', zorder=5)
        ax2.plot([], [], 'o', color='gray', mec='black', ms=11, label='maximum of $D_{ang}$')
        ax2.set_xlabel('redshift $z$')
        ax2.set_ylabel(r'angular-diameter distance $D_{ang}$ [$c/H_0$]')
        ax2.set_title('Angular-diameter distance for several models')
        ax2.set_xlim(0, z_models[-1])
        ax2.set_ylim(0, None)
        ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.14), ncol=2, fontsize=12)
        plt.tight_layout()
        plt.show()


# %%
def plot_horizon(z, r_exact, r_matter, r_radiation, z_eq, z_rec):
    """Left: comoving horizon r_H,com(z) in Mpc, numerical and the two limiting forms (4.41), (4.42).
    Right: ratio of the limiting forms to the numerical result. Vertical lines: z_eq and z_rec."""
    with plt.rc_context(_STYLE):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
        ax1.loglog(z, r_exact, color='black', label='numerical, eq. (4.40)')
        ax1.loglog(z, r_matter, '--', color=COLORS[0], label='matter dominated, eq. (4.41)')
        ax1.loglog(z, r_radiation, ':', color=COLORS[1], label='radiation dominated, eq. (4.42)')
        ax2.semilogx(z, r_matter / r_exact, '--', color=COLORS[0], label='(4.41) / numerical')
        ax2.semilogx(z, r_radiation / r_exact, ':', color=COLORS[1], label='(4.42) / numerical')
        ax2.axhline(1, color='black', lw=1.5)
        for ax in (ax1, ax2):
            ax.axvline(z_eq, color='gray', lw=2, alpha=0.7)
            ax.axvline(z_rec, color=COLORS[3], lw=2, alpha=0.9)
            ax.grid(True, which='both', alpha=0.3)
            ax.set_xlabel('redshift $z$')
        ymin, ymax = ax1.get_ylim()
        ax1.text(z_eq * 1.15, ymin * 3, f'$z_{{eq}} = {z_eq:.0f}$', color='gray')
        ax1.text(z_rec / 1.15, ymin * 3, f'$z_{{rec}} = {z_rec:.0f}$', color=COLORS[3], ha='right')
        ax1.set_ylabel(r'comoving horizon $r_{H,com}$ [Mpc]')
        ax1.set_title('Comoving horizon and its limiting forms')
        ax1.legend(loc='upper right')
        ax2.set_ylim(0, 4)
        ax2.set_ylabel('approximation / numerical')
        ax2.set_title('How good are the limiting forms?')
        ax2.legend(loc='upper center')
        plt.tight_layout()
        plt.show()


# %%
def plot_conformal_diagram(eta_today, eta_rec, chi_rec, theta_H_deg, z_rec):
    """Spacetime diagram in comoving distance and conformal time (both in units of c/H0) for a flat model.

    eta_today, eta_rec: conformal time since the Big Bang today and at recombination;
    chi_rec: comoving distance to the last-scattering surface. Light rays are lines at 45 degrees.
    Draws our past light cone, the last-scattering surface and the past light cones of two points on it.
    """
    def past_cone(x0, eta0, ax, color, alpha):
        triangle = Polygon([[x0 - eta0, 0], [x0 + eta0, 0], [x0, eta0]], closed=True, color=color, alpha=alpha, lw=0)
        ax.add_patch(triangle)

    with plt.rc_context(_STYLE):
        fig, ax = plt.subplots(figsize=(15, 8.5))
        x_limit = 1.08 * eta_today
        ax.fill([-eta_today, eta_today, 0], [0, 0, eta_today], color=COLORS[5], alpha=0.12, lw=0)
        ax.plot([-eta_today, 0, eta_today], [0, eta_today, 0], color=COLORS[0], lw=3,
                label='our past light cone (light rays reaching us today)')
        ax.axhline(eta_rec, color=COLORS[3], lw=2.5, ls='--', label=f'recombination, $z = {z_rec:.0f}$')
        for sign in (-1, 1):
            past_cone(sign * chi_rec, eta_rec, ax, COLORS[1], 0.9)
            ax.plot(sign * chi_rec, eta_rec, 'o', color=COLORS[1], ms=9, mec='black', zorder=5)
        ax.plot([], [], 's', color=COLORS[1], ms=12, label='causal past of two CMB points on opposite sides of the sky')
        ax.plot(0, eta_today, '*', color='black', ms=20, label='we, today', zorder=6)
        ax.set_xlim(-x_limit, x_limit)
        ax.set_ylim(0, 1.12 * eta_today)
        ax.set_aspect('equal')
        ax.set_xlabel(r'comoving distance [$c/H_0$]')
        ax.set_ylabel(r'conformal time $\eta = \int c\,dt/a$ [$c/H_0$]')
        ax.set_title(f'Spacetime diagram of a flat model: light rays are lines at 45 degrees\n'
                     f'horizon at recombination seen from today: $\\theta_H = {theta_H_deg:.2f}$ degrees')
        ax.legend(loc='upper left', fontsize=12, framealpha=0.95)

        # zoom on the right point of the last-scattering surface
        inset = ax.inset_axes([0.665, 0.42, 0.33, 0.33])
        inset.fill([-eta_today, eta_today, 0], [0, 0, eta_today], color=COLORS[5], alpha=0.12, lw=0)
        inset.plot([-eta_today, 0, eta_today], [0, eta_today, 0], color=COLORS[0], lw=3)
        inset.axhline(eta_rec, color=COLORS[3], lw=2, ls='--')
        past_cone(chi_rec, eta_rec, inset, COLORS[1], 0.9)
        inset.plot(chi_rec, eta_rec, 'o', color=COLORS[1], ms=9, mec='black', zorder=5)
        inset.annotate('', xy=(chi_rec + eta_rec, 0.12 * eta_rec), xytext=(chi_rec, 0.12 * eta_rec),
                       arrowprops=dict(arrowstyle='<->', color='black', lw=1.5))
        inset.text(chi_rec + 0.5 * eta_rec, 0.2 * eta_rec, r'$r_{H,com}$', ha='center', va='bottom', fontsize=13)
        inset.set_xlim(chi_rec - 3 * eta_rec, chi_rec + 3 * eta_rec)
        inset.set_ylim(0, 3 * eta_rec)
        inset.set_aspect('equal')
        inset.tick_params(labelsize=11)
        inset.set_title('zoom on one CMB point', fontsize=13)
        ax.indicate_inset_zoom(inset, edgecolor='black')
        plt.tight_layout()
        plt.show()
