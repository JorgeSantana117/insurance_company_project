print("Ejecutando data wrangling...")

import pandas as pd

# 1) Carga el dataset original
df = pd.read_csv('insurance_company_modified.csv', header=None)

"""
Entradas/Salidas (mismo directorio del notebook):
  Entrada  : 'insurance_company_modified.csv'  (sin encabezados)
  Salida   : 'insurance_company_cleaned_modified.csv'
"""

from pathlib import Path
import pandas as pd
import numpy as np

# =========================
# Parámetros configurables
# =========================
RARE_THRESHOLD = 0.01   # 1%: valores fuera de rango raros -> NaN + imputar; si >1% -> clip al rango
AUTO_SHIFT_MAX = 0      # 0 = desactivado; si >0, prueba shifts circulares 1..K y elige el mejor

IN_MOD  = "insurance_company_modified.csv"
OUT_MOD = "insurance_company_cleaned_modified.csv"

# Lista canónica (85 features + target)
COLS = [
"MOSTYPE","MAANTHUI","MGEMOMV","MGEMLEEF","MOSHOOFD","MGODRK","MGODPR","MGODOV","MGODGE",
"MRELGE","MRELSA","MRELOV","MFALLEEN","MFGEKIND","MFWEKIND","MOPLHOOG","MOPLMIDD","MOPLLAAG",
"MBERHOOG","MBERZELF","MBERBOER","MBERMIDD","MBERARBG","MBERARBO","MSKA","MSKB1","MSKB2","MSKC",
"MSKD","MHHUUR","MHKOOP","MAUT1","MAUT2","MAUT0","MZFONDS","MZPART","MINKM30","MINK3045","MINK4575",
"MINK7512","MINK123M","MINKGEM","MKOOPKLA","PWAPART","PWABEDR","PWALAND","PPERSAUT","PBESAUT","PMOTSCO",
"PVRAAUT","PAANHANG","PTRACTOR","PWERKT","PBROM","PLEVEN","PPERSONG","PGEZONG","PWAOREG","PBRAND",
"PZEILPL","PPLEZIER","PFIETS","PINBOED","PBYSTAND","AWAPART","AWABEDR","AWALAND","APERSAUT","ABESAUT",
"AMOTSCO","AVRAAUT","AAANHANG","ATRACTOR","AWERKT","ABROM","ALEVEN","APERSONG","AGEZONG","AWAOREG",
"ABRAND","AZEILPL","APLEZIER","AFIETS","AINBOED","ABYSTAND","CARAVAN"
]

# Rangos esperados (derivados del diccionario)
def build_expected():
    expected = {
        "MOSTYPE": (1,41),
        "MAANTHUI": (1,10),
        "MGEMOMV": (1,6),
        "MGEMLEEF": (1,6),
        "MOSHOOFD": (1,10),
        "CARAVAN": (0,1),
    }
    fields_0_9 = [
        "MGODRK","MGODPR","MGODOV","MGODGE","MRELGE","MRELSA","MRELOV","MFALLEEN","MFGEKIND","MFWEKIND",
        "MOPLHOOG","MOPLMIDD","MOPLLAAG","MBERHOOG","MBERZELF","MBERBOER","MBERMIDD","MBERARBG","MBERARBO",
        "MSKA","MSKB1","MSKB2","MSKC","MSKD","MHHUUR","MHKOOP","MAUT1","MAUT2","MAUT0","MZFONDS","MZPART",
        "MINKM30","MINK3045","MINK4575","MINK7512","MINK123M","MINKGEM","MKOOPKLA",
        "PWAPART","PWABEDR","PWALAND","PPERSAUT","PBESAUT","PMOTSCO","PVRAAUT","PAANHANG","PTRACTOR","PWERKT",
        "PBROM","PLEVEN","PPERSONG","PGEZONG","PWAOREG","PBRAND","PZEILPL","PPLEZIER","PFIETS","PINBOED","PBYSTAND",
    ]
    for f in fields_0_9:
        expected[f] = (0,9)
    a_fields = [
        "AWAPART","AWABEDR","AWALAND","APERSAUT","ABESAUT","AMOTSCO","AVRAAUT","AAANHANG",
        "ATRACTOR","AWERKT","ABROM","ALEVEN","APERSONG","AGEZONG","AWAOREG","ABRAND","AZEILPL",
        "APLEZIER","AFIETS","AINBOED","ABYSTAND"
    ]
    for a in a_fields:
        expected[a] = (1,12)
    return expected

EXPECTED = build_expected()

# =============
# Core helpers
# =============

def load_modified(path):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"No se encontró '{path}' en el directorio de trabajo.")
    df = pd.read_csv(p, header=None, dtype=str)
    # Mantener solo 86 columnas (85 features + target)
    if df.shape[1] > len(COLS):
        df = df.iloc[:, :len(COLS)]
    if df.shape[1] < len(COLS):
        for c in range(df.shape[1], len(COLS)):
            df[c] = np.nan
    df.columns = COLS
    # Forzar a numérico (no numérico -> NaN)
    df = df.apply(lambda s: pd.to_numeric(s.str.strip(), errors='coerce') if s.dtype == object else pd.to_numeric(s, errors='coerce'))
    return df


def inrange_score(df):
    total = 0
    ok = 0
    for c,(lo,hi) in EXPECTED.items():
        if c not in df.columns: continue
        s = df[c]; m = s.notna()
        total += int(m.sum())
        ok += int(((s>=lo)&(s<=hi)&m).sum())
    return ok / max(1,total)


def try_auto_shift(df, max_shift):
    if max_shift <= 0:
        return df, 0, None
    feat = [c for c in COLS if c != 'CARAVAN']
    base = df[feat].copy()
    best_df = df
    best_score = inrange_score(df)
    best_s = 0
    for s in range(1, max_shift+1):
        rolled = base.copy().iloc[:, -s:].join(base.iloc[:, :-s])
        rolled.columns = feat
        candidate = rolled.join(df[['CARAVAN']])
        score = inrange_score(candidate)
        if score > best_score:
            best_score, best_s, best_df = score, s, candidate
    return best_df, best_s, best_score


def analyze_and_repair(df, rare_threshold=0.01, verbose=True):
    report = {}
    out = df.copy()
    n = len(out)
    for col,(lo,hi) in EXPECTED.items():
        if col not in out.columns: continue
        s = out[col]
        mask_out = (~s.isna()) & ((s < lo) | (s > hi))
        cnt = int(mask_out.sum()); pct = cnt / n if n>0 else 0
        report[col] = {"min": lo, "max": hi, "bad_count": cnt, "bad_pct": float(pct)}
        if verbose and cnt>0:
            print(f"{col}: {cnt} ({pct:.2%}) fuera de rango [{lo},{hi}]")
        if cnt == 0: continue
        if pct <= rare_threshold:
            out.loc[mask_out, col] = np.nan
            if verbose: print("  -> rare -> set NaN (imputar)")
        else:
            out[col] = s.clip(lo, hi)
            if verbose: print("  -> widespread -> clip")
    return out, report


def finalize_and_save(df_repaired, out_path):
    df = df_repaired.copy()
    # Redondear a entero manteniendo NaN
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors='coerce')
        df[c] = df[c].round().where(~df[c].isna(), np.nan)

    policy = [c for c,v in EXPECTED.items() if v==(1,12)]
    binary = [c for c,v in EXPECTED.items() if v==(0,1)]
    catsm = [c for c in EXPECTED if c not in policy and c not in binary and EXPECTED[c][1] <= 9]

    # Imputaciones
    for c in catsm:
        if c in df.columns:
            mode_val = df[c].mode(dropna=True)
            fill = int(mode_val.iloc[0]) if len(mode_val)>0 else int(EXPECTED[c][0])
            df[c] = df[c].fillna(fill).astype(int)
    for c in policy:
        if c in df.columns:
            med = df[c].median(skipna=True)
            med = EXPECTED[c][0] if np.isnan(med) else med
            df[c] = df[c].fillna(int(round(med))).astype(int)
    for c in binary:
        if c in df.columns:
            mode_val = df[c].mode(dropna=True)
            fill = int(mode_val.iloc[0]) if len(mode_val)>0 else EXPECTED[c][0]
            df[c] = df[c].fillna(fill).astype(int)

    # Resto: mediana
    for c in df.columns:
        if df[c].isna().any():
            med = df[c].median(skipna=True)
            df[c] = df[c].fillna(int(round(med)) if not np.isnan(med) else 0)

    # Cast final a int
    for c in df.columns:
        df[c] = df[c].astype(int)

    df.to_csv(out_path, index=False)
    return df

# =====
# main
# =====

def main():
    print("--- Limpieza SOLO modificado (v3) ---")
    df = load_modified(IN_MOD)
    print(f"Cargado: {df.shape}")

    # Heurística de shift (opcional)
    if AUTO_SHIFT_MAX and AUTO_SHIFT_MAX > 0:
        print(f"Probando auto-shift hasta {AUTO_SHIFT_MAX} columnas...")
        shifted, s, score = try_auto_shift(df, AUTO_SHIFT_MAX)
        if s != 0:
            print(f"Shift seleccionado: +{s} (derecha) con in-range score={score:.4f}")
            df = shifted
        else:
            print("No se detectó mejora con shifts (s=0).")

    print("\n--- Analizar y reparar ---")
    repaired, report = analyze_and_repair(df, RARE_THRESHOLD, verbose=True)

    print("\n--- Imputar y guardar ---")
    cleaned = finalize_and_save(repaired, OUT_MOD)
    print(f"Guardado -> {OUT_MOD}")

    # Resumen rápido
    def quick(col):
        if col in df.columns:
            print(f"\n{col}")
            print("Antes:")
            print(df[col].value_counts(dropna=False).head(8))
            print("Después:")
            print(cleaned[col].value_counts(dropna=False).head(8))
    for c in ["MOSTYPE","MAANTHUI","AWAPART","PWAPART","CARAVAN"]:
        quick(c)

if __name__ == "__main__":
    main()
    df.to_csv('df_copy_eda.csv', index=False)
