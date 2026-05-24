# compare_spectra_tokens.py
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.integrate import simps
from scipy.optimize import nnls
from scipy.interpolate import interp1d
from sklearn.metrics.pairwise import cosine_similarity

# --- Example inputs: two spectra (x in cm^-1 descending typical IR), y = absorbance
# In practice load from CSV: columns wavenumber,absorbance
x = np.linspace(4000, 400, 2000)  # example grid
# for demo, create simple synthetic spectra with 3 features
def gauss(x, A, x0, s):
    return A * np.exp(-0.5*((x-x0)/s)**2)

# create spectrum A (like 1-pentene style)
yA = gauss(x, 0.4, 3050, 15) + gauss(x, 0.25, 1600, 12) + gauss(x, 0.15, 900, 60)
# create spectrum B slightly different composition
yB = gauss(x, 0.3, 3050, 15) + gauss(x, 0.35, 1600, 12) + gauss(x, 0.10, 900, 60)

# define ranges for token slices (example)
ranges = {
    'C_sp2_H': (3100, 3000),
    'C_eq_H': (1000, 650),
    'C_eq_C': (1620, 1580)  # narrow around 1600 ±20
}

def integrate_range(x, y, low, high):
    # x array might be descending; make ascending for integration
    if x[0] > x[-1]:
        x_ = x[::-1]; y_ = y[::-1]
    else:
        x_ = x; y_ = y
    # interpolation and integration within [low, high]
    f = interp1d(x_, y_, bounds_error=False, fill_value=0.0)
    xs = np.linspace(low, high, 200)
    ys = f(xs)
    return simps(ys, xs)

def extract_areas(x,y, ranges):
    areas = {}
    for k,(hi,lo) in ranges.items():
        # ranges defined hi->lo usually; integrate accordingly
        low = min(hi, lo); high = max(hi, lo)
        areas[k] = integrate_range(x,y, low, high)
    return areas

areasA = extract_areas(x,yA, ranges)
areasB = extract_areas(x,yB, ranges)

df = pd.DataFrame([areasA, areasB], index=['SpecA','SpecB'])
print("Raw areas per slice:\n", df)

# Normalize to composition fractions
def comp_from_areas(areas):
    vals = np.array(list(areas.values()))
    if vals.sum() == 0:
        return {k:0.0 for k in areas.keys()}
    fracs = vals / vals.sum()
    return dict(zip(areas.keys(), fracs))

compA = comp_from_areas(areasA)
compB = comp_from_areas(areasB)
print("\nNormalized composition fractions:")
print("SpecA:", compA)
print("SpecB:", compB)

# Map to tokens 'C' and '=' heuristically
# Example mapping: C token = C_sp2_H + 0.5*C_eq_C + 0.2*C_eq_H  (adjust per domain)
def map_to_C_eq(comp):
    C_val = comp['C_sp2_H'] + 0.5*comp['C_eq_C'] + 0.2*comp['C_eq_H']
    eq_val = 0.8*comp['C_eq_C'] + 0.3*comp['C_eq_H']
    return {'C': C_val, '=': eq_val}

tokA = map_to_C_eq(compA)
tokB = map_to_C_eq(compB)
print("\nMapped token values (unnormalized):")
print("SpecA:", tokA)
print("SpecB:", tokB)

# Normalize token vectors so sum=1 (composition)
def normalize_tokens(tok):
    s = sum(tok.values())
    if s == 0:
        return {k:0 for k in tok}
    return {k:v/s for k,v in tok.items()}

normA = normalize_tokens(tokA)
normB = normalize_tokens(tokB)
print("\nNormalized tokens (C, =):")
print("SpecA:", normA)
print("SpecB:", normB)

# compute ratios and comparisons
ratioA = normA['C'] / (normA['=']+1e-12)
ratioB = normB['C'] / (normB['=']+1e-12)
pct_change_C = (normB['C'] - normA['C']) / (normA['C']+1e-12)
print("\nRatios and change:")
print("C/= SpecA:", ratioA)
print("C/= SpecB:", ratioB)
print("Percent change C (B vs A):", pct_change_C)

# cosine similarity between token vectors
vecA = np.array([normA['C'], normA['=']]).reshape(1,-1)
vecB = np.array([normB['C'], normB['=']]).reshape(1,-1)
cos_sim = cosine_similarity(vecA, vecB)[0,0]
print("Cosine similarity (tokens):", cos_sim)

# Simple plot
labels = list(normA.keys())
valsA = [normA[k] for k in labels]
valsB = [normB[k] for k in labels]

xpos = np.arange(len(labels))
width = 0.35
plt.figure(figsize=(6,3))
plt.bar(xpos-width/2, valsA, width, label='SpecA')
plt.bar(xpos+width/2, valsB, width, label='SpecB')
plt.xticks(xpos, labels)
plt.ylabel("Fraction")
plt.title("Token composition comparison")
plt.legend()
plt.show()
