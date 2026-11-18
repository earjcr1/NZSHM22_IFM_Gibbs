"""The notebook's derived settings, the settings fixed since 2026-11-09, and the run's name (moved here from the
configuration cell on 2026-11-17, at the user's request).

    from run_config import derived_settings
    globals().update(derived_settings(globals()))

derived_settings(g) reads the configuration cell's settings from g and returns: each data group's misfit mode for
gibbs.py (SLIP_MISFIT, MFD_MISFIT, PALEO_MISFIT) and MISFIT, their common setting or 'mixed'; the fixed settings below
(or the overrides tools/run_nb.py passes in g['_OVERRIDES'], for sensitivity tests); SD_MULTIPLIER_; the start time _T0;
RUN_PLAN (None: the runs planned from RUN_MINUTES, tools/run_plan.py; or, through _OVERRIDES only, a fixed plan for tests);
RUN_NAME and RUN_DIR.
"""
import os
import time

from mfd_gr import MAGUNCERT                     # RECONSIDER_MAGUNCERT = True: the magnitude-uncertainty factor's default values

MISFIT_CHOICES = {'slip': ('linear', 'log10', 'half'), 'mfd': ('linear', 'log10', 'half'),
                  'paleo': ('rate', 'years', 'half')}
MISFIT_GROUP_MODES = {'slip': {'linear': 'split', 'log10': 'log10', 'half': 'split+log10'},   # as cru_common.GROUP_MISFIT_MODES:
                      'mfd': {'linear': 'gaussian', 'log10': 'log10', 'half': 'gaussian+log10'},   # each setting's mode
                      'paleo': {'rate': 'gaussian', 'years': 'years', 'half': 'gaussian+years'}}  # for gibbs.py

# Fixed since 2026-11-09, no longer settings (tools/run_nb.py NAME=VALUE still overrides each one, for sensitivity tests):
BLOCKS = None                                  # row groups by original row numbers ('S=0-387,MFD=695-713,P=714-753'); None: the slip-rate rows (up to
                                               # the first soft zero-forcing penalty row) and the magnitude-bin rows detected, any group named checked
SLIP_SECTIONS = 'auto'                         # each run of subsections holding copies of one fault section's slip rate one measurement (its rows
                                               # divided by sqrt(k)), stopping at each parent fault's end; 'none': each subsection its own measurement;
                                               # or a file in DATA_DIR of 'original_row section' lines. Runs are named SLIP_SECTIONS... only when not 'auto'
RUPTURE_SET = 'auto'                           # the OpenSHA rupture set: the one folder in DATA_DIR with a ruptures/ (a rupture set, NZSHM22_RuptureSet-...,
                                               # or an inversion solution, NZSHM22_InversionSolution-211k; its solution/rates.csv, if any, drawn for comparison);
                                               # or that folder's name, or its ruptures/ folder, in DATA_DIR; None: none (only with the CFM off)
CROSS_SECTIONS = 'all'                         # SLIP_SUM's and PLATE_RATE's strips of cross-sections perpendicular to the plate motion, each the mean
                                               # subsection trace length wide, over all of New Zealand (166 at 211,000 ruptures); 'between': only the 72
                                               # between the Alpine Fault's offshore end and the Needles-Wairarapa step-over (plate_rate.py)
CFM_SLIPRATES = True                           # the slip rates from the NZ Community Fault Model (cru_common.CFM_CSV, data/NZ_CFM_v1_0.csv;
                                               # preliminaries.py --cfm-sliprates, with the rupture set), no longer a setting since 2026-11-09:
                                               # each subsection's slip rate a split normal in mm/yr from its parent fault's line, mode
                                               # SLIP_RATE_FACTOR x SR_pref, SR_min and SR_max at 2 sd (each side's sd SLIP_RATE_FACTOR x half its
                                               # width; with CFM_TIMEFRAME the corrected distribution's mode, 2.5% and 97.5% points); the slip sums
                                               # follow. False only for sensitivity tests (tools/run_nb.py CFM_SLIPRATES=False): the inputs' Gaussians
# CFM_RATE_LINKS = CFM_TEMPLATE_RHO is not None: the CFM's borrowed rates correlated with their sources whenever CFM_TEMPLATE_RHO is on
# Fixed since 2026-11-17 (the user's request, moving them out of the notebook):
DOWN_THE_MIDDLE = False                        # the balance term (gibbs.py --down-the-middle): each data type's relative shortfall s
                                               # = sum of (prediction - target) / sum of targets, a misfit term A s^2 penalizing a type
                                               # predicted all low or all high; off since early November 2026 (REPORT.md, "Balance
                                               # between over- and underprediction"); True (A from the fit check) or 'zero-model'
                                               # (each type's own weight) for sensitivity tests (tools/run_nb.py DOWN_THE_MIDDLE=True;
                                               # the smoke test's reference values have it on)
MFD_UNCERTAINTY = 'as-given'                   # the MFD bins' uncertainties (preliminaries.py --mfd-uncertainty): as the inputs give
                                               # them; out of the notebook since 2026-11-18 (the user's choice: DATATYPES_EQUAL does
                                               # what it was for). 'two-measurements' or the M7.0-7.1 bin's relative sd for
                                               # sensitivity tests (tools/run_nb.py MFD_UNCERTAINTY='two-measurements')
SLIP_BLOCK, BIN_BLOCK = 'S', 'MFD'             # the row groups for the subsection and magnitude-bin summaries (cleaning names them)
FIXED = ('BLOCKS', 'SLIP_SECTIONS', 'RUPTURE_SET', 'CROSS_SECTIONS', 'CFM_SLIPRATES', 'CFM_RATE_LINKS', 'DOWN_THE_MIDDLE',
         'SLIP_BLOCK', 'BIN_BLOCK', 'MFD_UNCERTAINTY')
OVERRIDABLE = FIXED + ('RUN_DIR', 'RUN_PLAN')  # what run_nb.py may set through _OVERRIDES (RUN_DIR: its own folder; RUN_PLAN: a fixed
                                               # plan, dict(chains, sweeps, burn, thin, replicates), in place of RUN_MINUTES's, for tests)
PLAN_KEYS = ('chains', 'sweeps', 'burn', 'thin', 'replicates')


def derived_settings(g):
    """The derived settings, the fixed ones and the run's name, from the configuration cell's settings g (see the module's
    description). Raises ValueError for a misfit setting that is not one of MISFIT_CHOICES."""
    ov = dict(g.get('_OVERRIDES') or {})          # run_nb.py's overrides of the fixed settings
    unknown = set(ov) - set(OVERRIDABLE)
    if unknown:
        raise ValueError(f'overrides of {sorted(unknown)}: only {OVERRIDABLE} are set here')
    out = {}
    groups = (('slip', 'MISFIT_SLIP'), ('mfd', 'MISFIT_MFD'), ('paleo', 'MISFIT_PALEO'))
    for gr, nm in groups:
        if g[nm] not in MISFIT_CHOICES[gr]:
            raise ValueError(f'{nm} = {g[nm]!r}: use ' + ', '.join(repr(c_) for c_ in MISFIT_CHOICES[gr]))
    out['SLIP_MISFIT'], out['MFD_MISFIT'], out['PALEO_MISFIT'] = (MISFIT_GROUP_MODES[gr][g[nm]] for gr, nm in groups)
    out['MISFIT'] = {('linear', 'linear', 'rate'): 'linear', ('log10', 'log10', 'years'): 'log10',   # each group's mode above,
                     ('half',) * 3: 'half'}.get((g['MISFIT_SLIP'], g['MISFIT_MFD'], g['MISFIT_PALEO']), 'mixed')   # for gibbs.py
    if g['MFD_GR']:                                # MFD_GR: the bins scored by their own PDFs, whatever MISFIT_MFD says
        out['MFD_MISFIT'], out['MISFIT'] = 'pdf', 'mixed'
    fixed = dict(BLOCKS=BLOCKS, SLIP_SECTIONS=SLIP_SECTIONS, RUPTURE_SET=RUPTURE_SET, CROSS_SECTIONS=CROSS_SECTIONS,
                 CFM_SLIPRATES=CFM_SLIPRATES, CFM_RATE_LINKS=g['CFM_TEMPLATE_RHO'] is not None, DOWN_THE_MIDDLE=DOWN_THE_MIDDLE,
                 SLIP_BLOCK=SLIP_BLOCK, BIN_BLOCK=BIN_BLOCK, MFD_UNCERTAINTY=MFD_UNCERTAINTY)
    fixed.update({k: v for k, v in ov.items() if k in FIXED})
    if not (isinstance(fixed['DOWN_THE_MIDDLE'], bool) or fixed['DOWN_THE_MIDDLE'] == 'zero-model'):
        raise ValueError(f"DOWN_THE_MIDDLE = {fixed['DOWN_THE_MIDDLE']!r}: use True (the balance term), 'zero-model' (its "
                         'per-type weights) or False (the Gaussian misfit alone)')
    out.update(fixed)
    out['SD_MULTIPLIER_'] = 1 if fixed['CFM_SLIPRATES'] else 2.5   # the slip-rate sds' multiplier, no longer a setting
    # (since 2026-11-09, nor SR_pref's reading or the CFM's rounding): the CFM's slip rates (timeframe-corrected or not) with
    # each sd as given, SR_pref the mode, no rounding; the inputs' Gaussians as NZSHM22 gives them (sd 2.5 x 0.8 x SD = 2 SD),
    # scaled by SLIP_RATE_FACTOR
    out['RUN_PLAN'] = ov.get('RUN_PLAN')           # None: planned from RUN_MINUTES just before sampling (tools/run_plan.py)
    if out['RUN_PLAN'] is not None:
        if not (isinstance(out['RUN_PLAN'], dict) and set(PLAN_KEYS) <= set(out['RUN_PLAN'])):
            raise ValueError(f"RUN_PLAN = {out['RUN_PLAN']!r}: give dict({', '.join(k + '=...' for k in PLAN_KEYS)})")
        out['RUN_PLAN'] = dict(dict(hyper_every=1), **out['RUN_PLAN'])
    out['_T0'] = int(time.time())                  # the start time: the run's name and, with SEED None, its seed
    out['RUN_NAME'] = run_name(g, out)
    out['RUN_DIR'] = ov.get('RUN_DIR', os.path.join(g['DATA_DIR'], out['RUN_NAME']))   # everything the workflow writes:
    return out                                     # by default a new subfolder of DATA_DIR, so that each run has its own


def run_name(g, d):
    """The default RUN_DIR's name: the settings that tell runs apart, and the start time. g: the settings; d: the derived
    and fixed ones (derived_settings)."""
    S = dict(g)
    S.update(d)
    return (('SLIP_SECTIONS' + str(S['SLIP_SECTIONS']) + '_' if S['SLIP_SECTIONS'] != 'auto' else '')
            + (f"MFD_UNCERTAINTY{S['MFD_UNCERTAINTY']}_" if S['MFD_UNCERTAINTY'] != 'as-given' else '')
            + (f"DOWN_THE_MIDDLE_{S['DOWN_THE_MIDDLE']}_" if S['DOWN_THE_MIDDLE'] is not False else '')
            + f"{S['RUN_MINUTES']:g}min_floor{S['FLOOR']}"
            + (f"_misfit{S['MISFIT']}" if S['MISFIT'] != 'mixed' else
               f"_misfit_slip{S['MISFIT_SLIP']}_mfd{'pdf' if S['MFD_GR'] else S['MISFIT_MFD']}_paleo{S['MISFIT_PALEO']}")
            + ('_mfdmonotonic' if S['MFD_MONOTONIC'] else '')
            + ('_dsrtapered' if S['SLIP_ALONG_RUPTURE'] == 'tapered' else '')
            + ('_mfdgrb' + '-'.join(f'{x_:g}' for x_ in S['MFD_GR_B']) + 'N' + '-'.join(f'{x_:g}' for x_ in S['MFD_GR_N5'])
               + 's' + '-'.join(f'{x_:g}' for x_ in S['MFD_GR_NSCALE'])
               + ('m' + '-'.join(f'{x_:g}' for x_ in (MAGUNCERT if S['RECONSIDER_MAGUNCERT'] is True else S['RECONSIDER_MAGUNCERT']))
                  if S.get('RECONSIDER_MAGUNCERT') else '') if S['MFD_GR'] else '')
            + (f"_mfdtau{S['MFD_GR_TAU']:g}" if S['MFD_GR'] and S.get('MFD_GR_TAU') is not None else '')
            + ('_dtequal' if S.get('DATATYPES_EQUAL') else '')
            + (f"_platemax{S['SLIP_RATE_FACTOR']:g}{S['CROSS_SECTIONS']}" if S['PLATE_RATE'] else '')
            + ('_cfm' + ('tf' if S['CFM_TIMEFRAME'] else '')
               + (f"tmpl{S['CFM_TEMPLATE_RHO']:g}" + ('links' if S['CFM_RATE_LINKS'] else '')
                  if S['CFM_TEMPLATE_RHO'] is not None else '') if S['CFM_SLIPRATES'] else '')
            + (f"_srf{S['SLIP_RATE_FACTOR']:g}" if S['SLIP_RATE_FACTOR'] != 0.8 else '')
            + (f"_slipsumproj{S['CROSS_SECTIONS']}_L0{S['SLIP_SUM_L0']:g}" if S['SLIP_SUM'] else '')
            + f"_{d['_T0']}")
