"""Physical constants and model parameters.

All quantities are in SI units (m, kg, s, Pa) unless a name says otherwise.
Every value carries a source. Values without a literature source are either
exact unit conversions or explicit, documented modelling inputs (flagged
[MODEL INPUT]) -- never silent assumptions.

Sources
-------
Duanmu2019   : Duanmu et al., Front. Physiol. 10:853 (2019), doi:10.3389/fphys.2019.00853
Olufsen2000  : Olufsen et al., Ann. Biomed. Eng. 28:1281-1299 (2000), doi:10.1114/1.1326031
Wieneke2005  : Wieneke et al., J. Appl. Physiol. 98:1076-1082 (2005),
               doi:10.1152/japplphysiol.00724.2004 (global resting flow
               197 +/- 72 mL/min, hyperaemic 637 +/- 204 mL/min, n = 28)
MurrayMeta24 : Taylor et al., Am. J. Physiol. Heart Circ. Physiol. 327:H182-H190
               (2024), doi:10.1152/ajpheart.00142.2024 (pooled exponent 2.39,
               95% CI 2.24-2.54, human and mammalian coronary arteries)
ChoKensey91  : Cho and Kensey, Biorheology 28:241-262 (1991),
               doi:10.3233/BIR-1991-283-415 (Carreau parameters for blood)
"""


# --------------------------------------------------------------------------
# Unit conversions (exact)
# --------------------------------------------------------------------------
MMHG_TO_PA = 133.322387415       # 1 mmHg in Pa (exact by definition of mmHg)
PA_TO_MMHG = 1.0 / MMHG_TO_PA
ML_PER_MIN_TO_M3_PER_S = 1.0e-6 / 60.0
M3_PER_S_TO_ML_PER_MIN = 1.0 / ML_PER_MIN_TO_M3_PER_S

# --------------------------------------------------------------------------
# Blood properties
# --------------------------------------------------------------------------
RHO_BLOOD = 1055.0               # kg/m^3   -- Duanmu2019 (1.055 g/cm^3)
MU_BLOOD = 4.7e-3                # Pa.s     -- Duanmu2019 (0.0047 poise), Newtonian baseline

# Carreau shear-thinning model, mu(g) = mu_inf + (mu_0-mu_inf)[1+(lam*g)^2]^((n-1)/2)
# Human-blood parameters of ChoKensey91.
CARREAU_MU_0 = 56.0e-3           # Pa.s   zero-shear viscosity
CARREAU_MU_INF = 3.45e-3         # Pa.s   infinite-shear viscosity
CARREAU_LAMBDA = 3.313           # s      relaxation time
CARREAU_N = 0.3568               # -      power-law index

# --------------------------------------------------------------------------
# Arterial wall -- Olufsen empirical stiffness law
#   Eh/r0 = k1*exp(k2*r0) + k3        (units: Pa)
# CGS originals (Olufsen2000): k1=2.0e7 dyn/cm^2, k2=-22.53 /cm, k3=8.65e5 dyn/cm^2.
# Converted to SI (1 dyn/cm^2 = 0.1 Pa ; 1/cm = 100 /m).
# --------------------------------------------------------------------------
WALL_K1 = 2.0e6                  # Pa       -- Olufsen2000
WALL_K2 = -2253.0                # 1/m      -- Olufsen2000
WALL_K3 = 8.65e4                 # Pa       -- Olufsen2000

# --------------------------------------------------------------------------
# Cardiac cycle and driving pressures
# --------------------------------------------------------------------------
HEART_RATE_BPM = 75.0            # 1/min   [MODEL INPUT] normal resting heart rate
T_CYCLE = 60.0 / HEART_RATE_BPM  # s       cardiac period (0.8 s)

P_AO_SYS_MMHG = 120.0            # [MODEL INPUT] normal systolic aortic pressure
P_AO_DIA_MMHG = 80.0             # [MODEL INPUT] normal diastolic aortic pressure
# Mean arterial pressure, standard estimate MAP = DBP + (1/3)(SBP-DBP)
P_AO_MEAN_MMHG = P_AO_DIA_MMHG + (P_AO_SYS_MMHG - P_AO_DIA_MMHG) / 3.0

P_AO_SYS = P_AO_SYS_MMHG * MMHG_TO_PA
P_AO_DIA = P_AO_DIA_MMHG * MMHG_TO_PA
P_AO_MEAN = P_AO_MEAN_MMHG * MMHG_TO_PA

# Coronary venous (coronary sinus / right atrial) back-pressure
P_VENOUS_MMHG = 5.0              # [MODEL INPUT] typical coronary sinus pressure
P_VENOUS = P_VENOUS_MMHG * MMHG_TO_PA

# Ventricular pressures used to drive intramyocardial pressure
P_LV_SYS_MMHG = 120.0            # [MODEL INPUT] peak left-ventricular pressure
P_LV_DIA_MMHG = 8.0              # [MODEL INPUT] LV end-diastolic pressure
P_RV_SYS_MMHG = 25.0             # [MODEL INPUT] peak right-ventricular pressure
P_RV_DIA_MMHG = 4.0              # [MODEL INPUT] RV end-diastolic pressure

# Systolic fraction of the cardiac cycle (ejection timing)
SYSTOLE_FRACTION = 0.35          # [MODEL INPUT] ~280 ms systole at 75 bpm

# --------------------------------------------------------------------------
# Coronary flow and microvascular regulation
# --------------------------------------------------------------------------
# Typical adult value; within one standard deviation of Wieneke2005.
Q_CORONARY_REST_ML_MIN = 250.0   # mL/min  [MODEL INPUT] total resting coronary flow
Q_CORONARY_REST = Q_CORONARY_REST_ML_MIN * ML_PER_MIN_TO_M3_PER_S

# Morphometric flow-diameter exponent k of Q = kappa d^k, used by the mural
# outflow law and by the flow split at bifurcations (see coronary_bc).
MURRAY_EXPONENT = 2.39           # -- MurrayMeta24 (pooled human and mammalian exponent)

# Microvascular resistance reserve: at maximal hyperaemia every bed's
# resistance falls to 1/CORONARY_FLOW_RESERVE of its resting value. The
# whole-tree flow reserve is lower, because the epicardial resistance does not
# fall (3.6 for the default tree).
CORONARY_FLOW_RESERVE = 4.0      # [MODEL INPUT] microvascular resistance reserve

# Intramyocardial-pressure transmission factor: fraction of ventricular
# pressure transmitted to the intramyocardial compliant compartment. A value
# of 1.0 (P_im ~ ventricular pressure) is used by lumped coronary models and
# reproduces the near-cessation of systolic left-coronary inflow.
IM_PRESSURE_FACTOR = 1.0         # [MODEL INPUT] intramyocardial-pressure coupling

# --------------------------------------------------------------------------
# Lumped 0D coronary outlet split of total resistance
# R_total = R_a (proximal) + R_micro (microvascular) + R_v (venous)
# Fractions are a documented modelling choice consistent with coronary
# lumped-parameter models (proximal arteriolar : microvascular : venular).
# --------------------------------------------------------------------------
R_FRACTION_PROXIMAL = 0.20       # [MODEL INPUT]
R_FRACTION_MICRO = 0.70          # [MODEL INPUT] the regulated, hyperaemia-sensitive part
R_FRACTION_VENOUS = 0.10         # [MODEL INPUT]

# Compliance partition for the pulsatile 0D coronary outlet (arterial vs
# intramyocardial). Total outlet compliance is scaled per bed (see coronary_bc).
C_FRACTION_ARTERIAL = 0.30       # [MODEL INPUT]
C_FRACTION_INTRAMYO = 0.70       # [MODEL INPUT] compressed by intramyocardial pressure
