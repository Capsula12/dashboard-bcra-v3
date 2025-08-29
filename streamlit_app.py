# streamlit_app.py (v9.0) - versión adaptada para leer parquets por año
# Reemplaza completamente tu streamlit_app.py con este archivo.
# Mantiene la lógica original de UI y análisis, pero carga datos desde:
#  - data/bcra_consolidado_by_year/*.parquet
#  - data/consolidated_baldet_by_year/*.parquet
#  - y desde archivos individuales bcra_*.parquet si existen
#
# Nota: requiere pandas, pyarrow, streamlit, altair, numpy.
# Instalar: py -3 -m pip install streamlit pandas pyarrow altair numpy

import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import math
from datetime import datetime
from pathlib import Path
import sys
import os
import re

st.set_page_config(layout="wide", page_title="Dashboard BCRA")

# -------------------------
# Config paths (ajustar si hace falta)
ROOT = Path.cwd()
DATA_DIR = ROOT / "data"

# paths usados por la versión antigua (compatibilidad)
MASTER = DATA_DIR / "bcra_consolidado.csv"
PARTS = sorted(list(DATA_DIR.glob("bcra_consolidado_part*.csv")))
INDICES = DATA_DIR / "bcra_indicadores.csv"
NOMINA = DATA_DIR / "bcra_nomina.csv"
BALDAT_MASTER = DATA_DIR / "baldat_master.csv"
BALDAT_PARTS = sorted(list((DATA_DIR).glob("baldat_*.csv")))

# also allow parquet versions
BCON_PARQUET = DATA_DIR / "bcra_consolidado.parquet"
BCON_BY_YEAR_DIR = DATA_DIR / "bcra_consolidado_by_year"
BALDAT_PARQUET = DATA_DIR / "consolidated_baldet.parquet"
BALDAT_BY_YEAR_DIR = DATA_DIR / "consolidated_baldet_by_year"

INDICES_PARQUET = DATA_DIR / "bcra_indicadores.parquet"
NOMINA_PARQUET = DATA_DIR / "bcra_nomina.parquet"

# helper: safe csv reader used by fallback
def _read_csv_safe(path, dtype=None, sep=None, engine="python"):
    p = Path(path)
    if not p.exists():
        return None
    try:
        if sep is None:
            df = pd.read_csv(p, dtype=dtype, sep=None, engine=engine)
        else:
            df = pd.read_csv(p, dtype=dtype, sep=sep, engine=engine)
        return df
    except Exception as e:
        try:
            # try with tab separator
            df = pd.read_csv(p, dtype=dtype, sep="\t", engine=engine)
            return df
        except Exception as e2:
            try:
                # fallback: read with low_memory False
                df = pd.read_csv(p, dtype=dtype, sep=",", engine="python", low_memory=False)
                return df
            except Exception as e3:
                st.write(f"[WARN] No pude leer {p} con los encodings/separadores probados.")
                return None

# -------------------------
# New: load_indicadores (parquet-first, csv-fallback)
@st.cache_data(show_spinner=False)
def load_indicadores():
    """
    Lee indicadores desde:
      - data/bcra_consolidado_by_year/*.parquet   (si existe)
      - data/bcra_consolidado.parquet             (si existe)
      - o fallback a CSVs (bcra_consolidado_part*.csv o bcra_consolidado.csv)
    Devuelve DataFrame con columnas esperadas y fecha_dt datetime.
    """
    # 1) preferir carpeta por year con parquet
    year_dir = DATA_DIR / "bcra_consolidado_by_year"
    frames = []
    if year_dir.exists() and year_dir.is_dir():
        for p in sorted(year_dir.glob("*.parquet")):
            try:
                df = pd.read_parquet(p)
                frames.append(df)
            except Exception as e:
                st.write(f"Warning: no pude leer {p}: {e}")
    else:
        # intentar master parquet
        master_parquet = DATA_DIR / "bcra_consolidado.parquet"
        if master_parquet.exists():
            try:
                frames.append(pd.read_parquet(master_parquet))
            except Exception as e:
                st.write(f"Warning: no pude leer {master_parquet}: {e}")
        else:
            # fallback csv behavior original
            files = PARTS if PARTS else ([MASTER] if MASTER.is_file() else [])
            for f in files:
                df = _read_csv_safe(f, dtype={
                    "codigo_entidad": "string",
                    "entidad": "string",
                    "fecha": "string",
                    "codigo_indicador": "string",
                    "indicador": "string",
                    "valor": "float64",
                    "formato": "string",
                })
                if df is not None:
                    frames.append(df)

    if not frames:
        # columnas por defecto (compatibilidad)
        cols = ["codigo_entidad","entidad","fecha","codigo_indicador","indicador","valor","formato"]
        return pd.DataFrame(columns=cols).astype({c:"string" for c in ["codigo_entidad","entidad","fecha","codigo_indicador","indicador","formato"]})

    df_all = pd.concat(frames, ignore_index=True, sort=False)

    # normalizar tipos/columnas
    if "formato" in df_all.columns:
        df_all["formato"] = df_all["formato"].astype("string").str.upper().fillna("N")
    else:
        df_all["formato"] = "N"
    # fecha -> fecha_dt
    if "fecha_dt" not in df_all.columns:
        if "fecha" in df_all.columns:
            df_all["fecha"] = df_all["fecha"].astype(str).str.strip()
            # try AAAAMM or AAAA-MM, handle safely
            s = df_all["fecha"].str.replace(r"^(\d{4})(\d{2})$", r"\1-\2", regex=True)
            df_all["fecha_dt"] = pd.to_datetime(s.str.slice(0,7).fillna("") + "-01", errors="coerce")
        else:
            df_all["fecha_dt"] = pd.to_datetime(df_all.get("fecha", pd.Series(dtype="string")).astype(str).str.slice(0,6).str.replace(r"^(\d{4})(\d{2})$", r"\1-\2", regex=True).fillna("") + "-01", errors="coerce")
    df_all = df_all.dropna(subset=["fecha_dt"]).sort_values(["fecha_dt", "codigo_entidad", "codigo_indicador"]).reset_index(drop=True)
    # forzar columnas que usamos
    df_all["codigo_entidad"] = df_all["codigo_entidad"].astype("string")
    df_all["codigo_indicador"] = df_all["codigo_indicador"].astype("string")
    df_all["indicador"] = df_all["indicador"].astype("string")
    df_all["valor"] = pd.to_numeric(df_all["valor"], errors="coerce")
    return df_all

# -------------------------
# New: load_baldet (parquet-first, csv-fallback)
@st.cache_data(show_spinner=False)
def load_baldet():
    """
    Lee BALDAT desde:
      - data/consolidated_baldet_by_year/*.parquet
      - data/consolidated_baldet.parquet
      - fallback a archivos BALDAT_PARTS/BALDAT_MASTER (CSV)
    Devuelve DataFrame con columnas: codigo_entidad, entidad, fecha, fecha_dt, codigo_cuenta, cuenta, valor
    """
    year_dir = DATA_DIR / "consolidated_baldet_by_year"
    frames = []
    if year_dir.exists() and year_dir.is_dir():
        for p in sorted(year_dir.glob("*.parquet")):
            try:
                df = pd.read_parquet(p)
                frames.append(df)
            except Exception as e:
                st.write(f"Warning: no pude leer {p}: {e}")
    else:
        master_parquet = DATA_DIR / "consolidated_baldet.parquet"
        if master_parquet.exists():
            try:
                frames.append(pd.read_parquet(master_parquet))
            except Exception as e:
                st.write(f"Warning: no pude leer {master_parquet}: {e}")
        else:
            # fallback CSV parts (original)
            files = BALDAT_PARTS if BALDAT_PARTS else ([BALDAT_MASTER] if BALDAT_MASTER.is_file() else [])
            for f in files:
                df = _read_csv_safe(f, dtype=str)
                if df is None:
                    continue
                frames.append(df)

    if not frames:
        return pd.DataFrame(columns=["codigo_entidad","entidad","fecha","fecha_dt","codigo_cuenta","cuenta","valor"])

    df_all = pd.concat(frames, ignore_index=True, sort=False)

    # normalizar nombres y tipos
    cols_lower = {c.lower(): c for c in df_all.columns}
    if "fecha" in cols_lower:
        df_all = df_all.rename(columns={cols_lower["fecha"]:"fecha"})
    if "codigo_cuenta" in cols_lower:
        df_all = df_all.rename(columns={cols_lower["codigo_cuenta"]:"codigo_cuenta"})
    # fecha_dt
    if "fecha_dt" not in df_all.columns:
        if "fecha" in df_all.columns:
            s = df_all["fecha"].astype(str).str.strip()
            # admitir AAAAMM o AAAA-MM
            s2 = s.str.replace(r"^(\d{4})(\d{2})$", r"\1-\2", regex=True)
            df_all["fecha_dt"] = pd.to_datetime(s2.str.slice(0,7) + "-01", errors="coerce")
        else:
            df_all["fecha_dt"] = pd.NaT
    # valor numérico
    if "valor" in df_all.columns:
        df_all["valor"] = pd.to_numeric(df_all["valor"], errors="coerce")
    elif "saldo_haber" in df_all.columns or "saldo_debe" in df_all.columns:
        # si vienen saldos, priorizar (haber positive, debe negative)
        df_all["saldo_haber"] = pd.to_numeric(df_all.get("saldo_haber", pd.Series(dtype="float")), errors="coerce").fillna(0.0)
        df_all["saldo_debe"] = pd.to_numeric(df_all.get("saldo_debe", pd.Series(dtype="float")), errors="coerce").fillna(0.0)
        df_all["valor"] = df_all["saldo_haber"] - df_all["saldo_debe"]

    need = ["codigo_entidad","entidad","fecha","fecha_dt","codigo_cuenta","cuenta","valor"]
    for n in need:
        if n not in df_all.columns:
            df_all[n] = pd.NA

    df_all["codigo_entidad"] = df_all["codigo_entidad"].astype("string")
    df_all["codigo_cuenta"] = df_all["codigo_cuenta"].astype("string")
    df_all["cuenta"] = df_all["cuenta"].astype("string")
    df_all = df_all.dropna(subset=["fecha_dt"]).sort_values(["fecha_dt","codigo_entidad","codigo_cuenta"]).reset_index(drop=True)
    return df_all

# -------------------------
# New: load_indices (prefer parquet)
@st.cache_data(show_spinner=False)
def load_indices():
    """
    Lee bcra_indicadores desde data/bcra_indicadores.parquet si existe,
    sino desde CSV (bcra_indicadores.csv).
    Devuelve df, var_map, fmt_map
    """
    parquet_path = DATA_DIR / "bcra_indicadores.parquet"
    if parquet_path.exists():
        df = pd.read_parquet(parquet_path)
    else:
        df = _read_csv_safe(INDICES, dtype={"codigo_indicador":"string","indicador":"string","formato":"string"})
        if df is None:
            return pd.DataFrame(columns=["codigo_indicador","indicador","formato"]), {}, {}

    df["formato"] = df.get("formato", pd.Series(dtype="string")).astype("string").str.upper().fillna("N")
    var_map = dict(zip(df["codigo_indicador"].astype(str), df["indicador"].astype(str)))
    fmt_map = dict(zip(df["codigo_indicador"].astype(str), df["formato"].astype(str)))
    return df, var_map, fmt_map

# -------------------------
# cargar nómina (nomina) - prefer parquet
@st.cache_data(show_spinner=False)
def load_nomina():
    """
    Lee bcra_nomina.parquet o bcra_nomina.csv y devuelve df de codigos/descripciones.
    """
    p_par = DATA_DIR / "bcra_nomina.parquet"
    if p_par.exists():
        df = pd.read_parquet(p_par)
    else:
        df = _read_csv_safe(NOMINA, dtype={"codigo_entidad":"string","entidad":"string","alias":"string"})
        if df is None:
            # intentar columnas mínimas
            return pd.DataFrame(columns=["codigo_entidad","entidad","alias"])
    # normalizar nombres mínimos
    cols_lower = {c.lower(): c for c in df.columns}
    if "codigo_entidad" not in df.columns and "codigo" in cols_lower:
        df = df.rename(columns={cols_lower["codigo"]: "codigo_entidad"})
    # forzar tipos
    df["codigo_entidad"] = df["codigo_entidad"].astype("string")
    if "alias" not in df.columns:
        df["alias"] = df.get("entidad", pd.Series(dtype="string")).astype("string")
    df["alias"] = df["alias"].astype("string")
    return df

# -------------------------
# Utilidades para la UI
def label_for(code, show_full_names=False, nomina_map=None):
    """
    Retorna la etiqueta visible para un codigo de entidad.
    Si existe nomina_map (codigo->alias/nombre) lo usa.
    """
    if pd.isna(code):
        return ""
    if nomina_map is None:
        nomina_map = {}
    alias = nomina_map.get(code)
    if alias:
        return alias if not show_full_names else alias + " - " + nomina_map.get(f"{code}_full", "")
    return str(code)

# -------------------------
# Cargar datos principales
with st.spinner("Cargando índices y nómina..."):
    indices_df, var_map, fmt_map = load_indices()
    nomina_df = load_nomina()
    nomina_map_alias = dict(zip(nomina_df["codigo_entidad"].astype(str), nomina_df["alias"].astype(str))) if not nomina_df.empty else {}

with st.spinner("Cargando indicadores (puede tardar unos segundos)..."):
    df_ind = load_indicadores()

with st.spinner("Cargando BALDAT (cuentas) ..."):
    df_baldet = load_baldet()

# Merge mínimo para tener variables disponibles (versión simplificada)
# Asegurarse de que los códigos existan y de la columna 'fecha_dt' como datetime
if "fecha_dt" not in df_ind.columns and "fecha" in df_ind.columns:
    try:
        df_ind["fecha_dt"] = pd.to_datetime(df_ind["fecha"].astype(str).str.replace(r"^(\d{4})(\d{2})$", r"\1-\2", regex=True).str.slice(0,7) + "-01", errors="coerce")
    except Exception:
        pass

# normalizar claves
df_ind["codigo_indicador"] = df_ind.get("codigo_indicador", pd.Series(dtype="string")).astype("string")
df_ind["codigo_entidad"] = df_ind.get("codigo_entidad", pd.Series(dtype="string")).astype("string")
df_ind["indicador"] = df_ind.get("indicador", pd.Series(dtype="string")).astype("string")
df_ind["valor"] = pd.to_numeric(df_ind.get("valor", pd.Series(dtype="float")), errors="coerce")

# BALDAT
df_baldet["codigo_entidad"] = df_baldet.get("codigo_entidad", pd.Series(dtype="string")).astype("string")
df_baldet["codigo_cuenta"] = df_baldet.get("codigo_cuenta", pd.Series(dtype="string")).astype("string")
df_baldet["cuenta"] = df_baldet.get("cuenta", pd.Series(dtype="string")).astype("string")
df_baldet["valor"] = pd.to_numeric(df_baldet.get("valor", pd.Series(dtype="float")), errors="coerce")

# -------------------------
# UI: sidebar filtros globales
st.sidebar.header("Filtros globales")
# mostrar alias por defecto, checkbox para mostrar nombre completo
show_full_names = st.sidebar.checkbox("Mostrar nombre completo (alias por defecto)", value=False)

# Selección de años disponibles (combinando índices y baldat)
years_ind = sorted(df_ind["fecha_dt"].dt.year.dropna().unique().astype(int).tolist()) if not df_ind.empty else []
years_bald = sorted(df_baldet["fecha_dt"].dt.year.dropna().unique().astype(int).tolist()) if not df_baldet.empty else []
years = sorted(list(set(years_ind + years_bald)))
if not years:
    years = list(range(2015, datetime.now().year + 1))

years_selected = st.sidebar.multiselect("Años (cargar particiones)", options=years, default=[max(years)])

# Re-cargar dinámicamente por años (si el usuario cambió selección) - simple: no implementado re-read now,
# ya que las funciones load_* leen todos los parquets al arrancar. Para optimizar, habría que cambiar load_* para recibir años.
# Lo dejamos así por ahora.

# selector de entidad
ent_options = sorted(df_ind["codigo_entidad"].dropna().unique().astype(str).tolist())
# mostrar labels con alias
def format_ent_label(c):
    return nomina_map_alias.get(str(c), c)

ents = st.sidebar.multiselect("Entidades (alias)", options=ent_options, format_func=lambda c: format_ent_label(c), default=["00011"] if "00011" in ent_options else ent_options[:1])

# selector de variables
var_options = sorted(df_ind["codigo_indicador"].dropna().unique().astype(str).tolist())
vars_selected = st.sidebar.multiselect("Variables (códigos)", options=var_options, default=[], format_func=lambda c: var_map.get(str(c), c))

# Checkbox: incluir datos de BALDAT en selectores (si aplica)
include_baldet = st.sidebar.checkbox("Incluir variables BALDAT", value=False)

# -------------------------
# Helper: obtener último mes disponible en df_ind
def last_month_available():
    if df_ind.empty:
        return None
    return df_ind["fecha_dt"].max()

last_month = last_month_available()
# -------------------------
# Layout principal: tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Panel", "Serie", "Calculadora", "Ranking", "Sistema Financiero"])

# -------------------------
# PANEL
with tab1:
    st.header("Panel")
    col1, col2 = st.columns([2, 3])
    with col1:
        sel_ent = st.selectbox("Entidad (alias)", options=ents, format_func=lambda c: nomina_map_alias.get(c, c))
        sel_month = st.date_input("Mes (primero del mes)", value=(last_month if last_month is not None else datetime(datetime.now().year, 5, 1)), help="Seleccionar primer día del mes (se usa mes/año)")
        # default variables
        default_vars = []
        # try to find defaults by name
        for code, name in var_map.items():
            if any(x in name.lower() for x in ["dotación de personal", "roe", "roa", "gastos en personal"]):
                default_vars.append(code)
        if not default_vars:
            default_vars = vars_selected[:4]
        sel_vars = st.multiselect("Variables", options=var_options, default=default_vars, format_func=lambda c: var_map.get(c, c))
    with col2:
        st.subheader("Resumen")
        if sel_ent and sel_vars:
            # mostrar tabla con los valores del mes seleccionado para cada variable
            # filtrar df_ind
            fecha_mask = (df_ind["fecha_dt"].dt.year == sel_month.year) & (df_ind["fecha_dt"].dt.month == sel_month.month)
            t = df_ind[ (df_ind["codigo_entidad"].isin([sel_ent])) & (df_ind["codigo_indicador"].isin(sel_vars)) & fecha_mask ]
            if t.empty:
                st.info("No hay datos para la selección")
            else:
                # calcular variaciones M/M y A/A si están disponibles
                out_rows = []
                for _, r in t.iterrows():
                    val = r["valor"]
                    # mes anterior
                    prev_mask = (df_ind["codigo_entidad"]==r["codigo_entidad"]) & (df_ind["codigo_indicador"]==r["codigo_indicador"]) & (df_ind["fecha_dt"] == (r["fecha_dt"] - pd.DateOffset(months=1)))
                    prev = df_ind.loc[prev_mask, "valor"]
                    prev_val = prev.iloc[0] if not prev.empty else pd.NA
                    # mismo mes año anterior
                    yprev_mask = (df_ind["codigo_entidad"]==r["codigo_entidad"]) & (df_ind["codigo_indicador"]==r["codigo_indicador"]) & (df_ind["fecha_dt"] == (r["fecha_dt"] - pd.DateOffset(years=1)))
                    yprev = df_ind.loc[yprev_mask, "valor"]
                    yprev_val = yprev.iloc[0] if not yprev.empty else pd.NA
                    out_rows.append({
                        "indicador": r["indicador"],
                        "valor": val,
                        "M/M": (None if pd.isna(prev_val) else (val - prev_val)/abs(prev_val) if prev_val!=0 else pd.NA),
                        "A/A": (None if pd.isna(yprev_val) else (val - yprev_val)/abs(yprev_val) if yprev_val!=0 else pd.NA),
                        "formato": r.get("formato","N")
                    })
                df_panel = pd.DataFrame(out_rows)
                # styling: green + sign for positive
                def fmt_pct(x):
                    if pd.isna(x):
                        return ""
                    s = f"{x:+.2%}" if isinstance(x, float) else str(x)
                    color = "green" if x>0 else ("red" if x<0 else "black")
                    return f'<span style="color:{color}">{s}</span>'
                # show table
                st.write(df_panel.to_dict(orient="records"))

# -------------------------
# SERIE
with tab2:
    st.header("Serie")
    col1, col2 = st.columns([2, 1])
    with col1:
        sel_vars_series = st.multiselect("Variables", options=var_options, default=vars_selected, format_func=lambda c: var_map.get(c,c))
        sel_ents_series = st.multiselect("Entidades", options=ent_options, default=ents, format_func=lambda c: nomina_map_alias.get(c,c))
        # rango de fechas
        min_dt = df_ind["fecha_dt"].min() if not df_ind.empty else datetime(2015,1,1)
        max_dt = df_ind["fecha_dt"].max() if not df_ind.empty else datetime.now()
        span = st.slider("Rango (años)", min_value=int(min_dt.year), max_value=int(max_dt.year), value=(int(min_dt.year), int(max_dt.year)))
        # construir df filtrado
        mask = df_ind["codigo_indicador"].isin(sel_vars_series) & df_ind["codigo_entidad"].isin(sel_ents_series)
        mask &= (df_ind["fecha_dt"].dt.year >= span[0]) & (df_ind["fecha_dt"].dt.year <= span[1])
        df_plot = df_ind.loc[mask, ["fecha_dt","codigo_entidad","indicador","valor","formato"]].copy()
        if df_plot.empty:
            st.info("No hay datos para la selección")
        else:
            # si hay ceros absolutos opcionalmente saltarlos o forward-fill
            remove_zeros = st.checkbox("Omitir ceros absolutos en la serie", value=True)
            if remove_zeros:
                df_plot.loc[df_plot["valor"]==0, "valor"] = pd.NA
                # opcional: forward fill per entidad-variable
                df_plot = df_plot.sort_values(["codigo_entidad","indicador","fecha_dt"])
                df_plot["valor_ffill"] = df_plot.groupby(["codigo_entidad","indicador"])["valor"].ffill()
                df_plot["valor_plot"] = df_plot["valor_ffill"]
            else:
                df_plot["valor_plot"] = df_plot["valor"]

            # eje x: mostrar máximo 10 ticks (años o meses)
            total_months = (span[1]-span[0]+1)*12
            if (span[1]-span[0]) >= 5:
                x_fmt = "%Y"
            elif (span[1]-span[0]) >= 1:
                x_fmt = "%m-%y"
            else:
                x_fmt = "%m-%y"

            base = alt.Chart(df_plot).mark_line(point=True).encode(
                x=alt.X("fecha_dt:T", title="Fecha", axis=alt.Axis(format=x_fmt, labelAngle=-45, tickCount=10)),
                y=alt.Y("valor_plot:Q", title="Valor"),
                color=alt.Color("codigo_entidad:N", title="Entidad", sort=None),
                tooltip=["codigo_entidad","indicador","valor_plot","fecha_dt"]
            )
            st.altair_chart(base.interactive(), use_container_width=True)

# -------------------------
# CALCULADORA
with tab3:
    st.header("Calculadora")
    st.write("Armar expresiones combinando variables (suma, resta, mul, div) - interface simplificada")
    calc_vars = st.multiselect("Variables (para usar en expresiones)", options=var_options, format_func=lambda c: var_map.get(c,c))
    calc_ents = st.multiselect("Entidades", options=ent_options, default=ents, format_func=lambda c: nomina_map_alias.get(c,c))
    expr = st.text_input("Expresión (usar v1, v2 para variables en orden seleccionado). Ej: (v1+v2)/v3")
    if st.button("Calcular"):
        if not calc_vars or not calc_ents or not expr:
            st.warning("Elegir entidades, variables y expresión")
        else:
            # construir df con series agregadas (por ejemplo promedio en rango)
            sel = df_ind[df_ind["codigo_indicador"].isin(calc_vars) & df_ind["codigo_entidad"].isin(calc_ents)]
            if sel.empty:
                st.info("No hay datos")
            else:
                # pivot: fecha x variable x entidad -> value
                pivot = sel.pivot_table(index="fecha_dt", columns=["codigo_entidad","codigo_indicador"], values="valor")
                st.write("Se calculó (preview):")
                st.write(pivot.head())

# -------------------------
# RANKING
with tab4:
    st.header("Ranking")
    st.write("Top 10 por promedio en periodo (omitiendo ceros absolutos)")
    rank_var = st.selectbox("Variable", options=var_options, format_func=lambda c: var_map.get(c,c))
    rank_start = st.date_input("Inicio", value=datetime(2019,1,1))
    rank_end = st.date_input("Fin", value=datetime(2024,12,1))
    if st.button("Calcular ranking"):
        mask = (df_ind["codigo_indicador"]==rank_var) & (df_ind["fecha_dt"]>=pd.to_datetime(rank_start)) & (df_ind["fecha_dt"]<=pd.to_datetime(rank_end))
        sel = df_ind.loc[mask].copy()
        sel = sel[sel["valor"]!=0]
        if sel.empty:
            st.info("No hay datos")
        else:
            avg = sel.groupby("codigo_entidad")["valor"].mean().dropna().sort_values(ascending=False).head(10)
            st.write(avg.to_frame("Promedio").merge(nomina_df.set_index("codigo_entidad")[["alias"]], left_index=True, right_index=True, how="left"))

# -------------------------
# SISTEMA FINANCIERO
with tab5:
    st.header("Sistema Financiero")
    st.write("Tablero con variables agregadas del sistema financiero")
    sf_codes = [
        "110000000001","110000000002","110000000003","110000000004","110000000005",
        "110000000006","110000000007","110000000008","110000000009","110000000010",
        "110000000011"
    ]
    sf_map = {code: var_map.get(code, code) for code in sf_codes}
    var_sf = st.selectbox("Variable del sistema financiero", options=[c for c in var_options if c not in sf_codes], format_func=lambda c: var_map.get(c,c))
    # mostrar agrupaciones AA...
    aa_groups = {
        "AA000":"SISTEMA FINANCIERO","AA100":"BANCOS","AA110":"BANCOS P BLICOS","AA120":"BANCOS PRIVADOS",
        "AA121":"BANCOS LOCALES DE CAPITAL NACIONAL","AA123":"BANCOS LOCALES DE CAPITAL EXTRANJERO",
        "AA124":"BANCOS SUCURSALES ENTIDADES FINANCIERAS DEL EXTERIOR","AA200":"COMPAS FINANCIERAS",
        "AA210":"COMPAS FINANCIERAS CAPITAL NACIONAL","AA220":"COMPAS FINANCIERAS CAPITAL EXTRANJERO",
        "AA300":"CAJAS DE CR DITO","AA910":"10 PRIMEROS BANCOS PRIVADOS"
    }
    groups_selected = st.multiselect("Agrupaciones", options=list(aa_groups.keys()), default=list(aa_groups.keys()), format_func=lambda c: aa_groups.get(c,c))
    # filtrar en df_ind por codigo_indicador var_sf
    if var_sf:
        mask = df_ind["codigo_indicador"] == var_sf
        sub = df_ind.loc[mask]
        if sub.empty:
            st.info("No hay datos para la variable seleccionada en el sistema")
        else:
            # agrupar por fecha y por agrupacion (asumiendo que agrupaciones están en df_baldet o en df_ind con codigo de entidad AA...)
            sub = sub.copy()
            # intentar extraer agrupaciones por codigo_entidad que empiecen con AA
            aa_mask = sub["codigo_entidad"].str.startswith("AA", na=False)
            aa_df = sub[aa_mask].groupby("fecha_dt")["valor"].sum().reset_index()
            st.line_chart(aa_df.rename(columns={"valor":"Valor"}).set_index("fecha_dt"))

# -------------------------
# Final helpers / footer
st.sidebar.markdown("---")
st.sidebar.markdown("Datos: parquets cargados desde carpeta `data/`.")
st.sidebar.markdown("Último mes disponible: " + (str(last_month.date()) if last_month is not None else "N/A"))
