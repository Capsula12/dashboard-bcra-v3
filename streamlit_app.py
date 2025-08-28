# streamlit_app.py (v9.0)  -- incorpora BALDAT + filtros por fuente
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from pathlib import Path
import glob
from datetime import date, datetime
import math
import io

st.set_page_config(page_title="Indicadores BCRA (v9.0)", layout="wide")

DATA_DIR = Path("./data")
# indicadores
MASTER = DATA_DIR / "bcra_consolidado.csv"
PARTS = sorted([Path(p) for p in glob.glob(str(DATA_DIR / "bcra_consolidado_part*.csv"))])
INDICES = DATA_DIR / "bcra_indicadores.csv"
NOMINA = DATA_DIR / "bcra_nomina.csv"
# baldet (nuevo)
BALDAT_MASTER = DATA_DIR / "consolidated_baldet.csv"
BALDAT_PARTS = sorted([Path(p) for p in glob.glob(str(DATA_DIR / "consolidated_baldet_part*.csv"))])
BCRA_CUENTAS = DATA_DIR / "bcra_cuentas.csv"

DEFAULT_ENTITY_CODE = "00011"
DEFAULT_MONTH_STR = "2025-05"
DEFAULT_VAR_HINTS = ["Dotación de personal", "ROE", "ROA", "Gastos en personal"]

# variables a ocultar en selectores (sistema financiero)
HIDE_VARS = {
    "110000000001","110000000002","110000000003","110000000004","110000000005","110000000006",
    "110000000007","110000000008","110000000009","110000000010","110000000011"
}

AA_GROUPS = {
    "AA000": "SISTEMA FINANCIERO",
    "AA100": "BANCOS",
    "AA110": "BANCOS PÚBLICOS",
    "AA120": "BANCOS PRIVADOS",
    "AA121": "BANCOS LOCALES DE CAPITAL NACIONAL",
    "AA123": "BANCOS LOCALES DE CAPITAL EXTRANJERO",
    "AA124": "BANCOS SUCURSALES ENTIDADES FINANCIERAS DEL EXTERIOR",
    "AA200": "COMPAÑÍAS FINANCIERAS",
    "AA210": "COMPAÑÍAS FINANCIERAS DE CAPITAL NACIONAL",
    "AA220": "COMPAÑÍAS FINANCIERAS  DE CAPITAL EXTRANJERO",
    "AA300": "CAJAS DE CRÉDITO",
    "AA910": "10 PRIMEROS BANCOS PRIVADOS",
}

def _read_csv_safe(path, **kwargs):
    if not Path(path).is_file():
        return None
    try:
        return pd.read_csv(path, encoding="utf-8-sig", **kwargs)
    except Exception:
        return pd.read_csv(path, **kwargs)

@st.cache_data(show_spinner=False)
def load_indicadores():
    # carga indicadores (consolidado)
    dtypes = {
        "codigo_entidad": "string",
        "entidad": "string",
        "fecha": "string",
        "codigo_indicador": "string",
        "indicador": "string",
        "valor": "float64",
        "formato": "string",
    }
    files = PARTS if PARTS else ([MASTER] if MASTER.is_file() else [])
    if not files:
        return pd.DataFrame(columns=list(dtypes.keys())).astype(dtypes)
    frames = []
    for f in files:
        df = _read_csv_safe(f, dtype=dtypes)
        if df is not None:
            frames.append(df)
    if not frames:
        return pd.DataFrame(columns=list(dtypes.keys())).astype(dtypes)
    df_all = pd.concat(frames, ignore_index=True)
    df_all["formato"] = df_all["formato"].astype("string").str.upper().fillna("N")
    df_all["fecha"] = df_all["fecha"].astype("string").str.strip()
    df_all["fecha_dt"] = pd.to_datetime(df_all["fecha"] + "-01", format="%Y-%m-%d", errors="coerce")
    df_all = df_all.dropna(subset=["fecha_dt"]).sort_values(["fecha_dt", "codigo_entidad", "codigo_indicador"]).reset_index(drop=True)
    return df_all

@st.cache_data(show_spinner=False)
def load_baldet():
    # carga baldat consolidado (ya generado por el script consolida_baldet)
    cols = None
    files = BALDAT_PARTS if BALDAT_PARTS else ([BALDAT_MASTER] if BALDAT_MASTER.is_file() else [])
    if not files:
        return pd.DataFrame(columns=["codigo_entidad","entidad","fecha","fecha_dt","codigo_cuenta","cuenta","valor","source_file"]).astype(object)
    frames = []
    for f in files:
        df = _read_csv_safe(f, dtype=str)
        if df is None: 
            continue
        # normalizar columnas esperadas si vienen con otros nombres
        lower = {c.lower(): c for c in df.columns}
        # try to detect typical names
        if "codigo_cuenta" in lower:
            df = df.rename(columns={lower["codigo_cuenta"]:"codigo_cuenta"})
        if "fecha" in lower:
            df = df.rename(columns={lower["fecha"]:"fecha"})
        # convert types
        if "fecha" in df.columns:
            df["fecha"] = df["fecha"].astype(str).str.strip()
            df["fecha_dt"] = pd.to_datetime(df["fecha"].str.slice(0,4)+"-"+df["fecha"].str.slice(4,6)+"-01", format="%Y-%m-%d", errors="coerce")
        if "valor" in df.columns:
            df["valor"] = pd.to_numeric(df["valor"], errors="coerce")
        # keep only needed columns, fill missing
        need = ["codigo_entidad","entidad","fecha","fecha_dt","codigo_cuenta","cuenta","valor"]
        for n in need:
            if n not in df.columns:
                df[n] = np.nan
        frames.append(df[need].copy())
    if not frames:
        return pd.DataFrame(columns=["codigo_entidad","entidad","fecha","fecha_dt","codigo_cuenta","cuenta","valor"])
    df_all = pd.concat(frames, ignore_index=True)
    df_all = df_all.dropna(subset=["fecha_dt"]).sort_values(["fecha_dt","codigo_entidad","codigo_cuenta"]).reset_index(drop=True)
    return df_all

@st.cache_data(show_spinner=False)
def load_nomina_maps():
    df = _read_csv_safe(NOMINA)
    if df is None or df.empty:
        return {}, {}
    cols = {c.lower(): c for c in df.columns}
    code_col = cols.get("codigo_entidad", "codigo_entidad")
    name_col = cols.get("entidad", "entidad")
    alias_col = cols.get("alias", None)
    def to_clean_str_series(s):
        if s is None:
            return pd.Series([], dtype="string")
        s = s.astype("string").fillna("").astype(str).str.strip()
        s = s.replace({"<NA>": ""})
        return s
    codes = to_clean_str_series(df[code_col])
    names = to_clean_str_series(df[name_col])
    aliases = to_clean_str_series(df[alias_col]) if alias_col else names.copy()
    alias_use = [a if a else n for a,n in zip(aliases,names)]
    alias_map = dict(zip(codes.tolist(), alias_use))
    ent_map = dict(zip(codes.tolist(), names.tolist()))
    return ent_map, alias_map

@st.cache_data(show_spinner=False)
def load_indices():
    dtypes = {"codigo_indicador":"string","indicador":"string","formato":"string"}
    df = _read_csv_safe(INDICES, dtype=dtypes)
    if df is None or df.empty:
        return pd.DataFrame(columns=list(dtypes.keys())), {}, {}
    df["formato"] = df["formato"].astype("string").str.upper().fillna("N")
    var_map = dict(zip(df["codigo_indicador"], df["indicador"]))
    fmt_map = dict(zip(df["codigo_indicador"], df["formato"]))
    return df, var_map, fmt_map

@st.cache_data(show_spinner=False)
def load_cuentas_map():
    df = _read_csv_safe(BCRA_CUENTAS)
    if df is None or df.empty:
        return {}
    cols = {c.lower(): c for c in df.columns}
    code_col = cols.get("codigo_cuenta", list(df.columns)[0])
    name_col = cols.get("cuenta", list(df.columns)[1] if len(df.columns)>1 else list(df.columns)[0])
    m = dict(zip(df[code_col].astype(str), df[name_col].astype(str)))
    return m

# cargar datasets
df_ind = load_indicadores()
df_bald = load_baldet()
ent_map, alias_map = load_nomina_maps()
idx_df, var_map_ind, fmt_map_ind = load_indices()
cuentas_map = load_cuentas_map()

# construir mapas combinados (indicadores + baldat)
# Para BALDAT prefix usamos "B_{codigo_cuenta}"
bald_var_map = {}
bald_fmt_map = {}
if not df_bald.empty:
    # única lista de cuentas
    codes = df_bald["codigo_cuenta"].astype(str).fillna("").unique().tolist()
    for c in codes:
        if not c:
            continue
        key = f"B_{c}"
        desc = cuentas_map.get(c) or df_bald.loc[df_bald["codigo_cuenta"]==c, "cuenta"].iloc[0]
        bald_var_map[key] = f"{desc} (Baldet {c})"
        bald_fmt_map[key] = "N"

# combinado
combined_var_map = {**var_map_ind, **bald_var_map}
combined_fmt_map = {**fmt_map_ind, **bald_fmt_map}

# helpers
def percent_change(curr, prev):
    if curr is None or prev in (None,0) or pd.isna(curr) or pd.isna(prev) or prev==0:
        return None
    return (curr/prev)-1.0

def format_value(val, fmt="N", decimals=2):
    if pd.isna(val):
        return "—"
    if (fmt or "").upper() == "P":
        return f"{val:.{decimals}f}%"
    return f"{val:,.{decimals}f}".replace(",", "X").replace(".", ",").replace("X", ".")

def to_plot_series(series_vals, fmt_code):
    return series_vals/100.0 if str(fmt_code).upper()=="P" else series_vals

def tick_values_for_range(start_dt, end_dt):
    start = pd.Timestamp(start_dt.year, start_dt.month, 1)
    end = pd.Timestamp(end_dt.year, end_dt.month, 1)
    span_months = (end.year - start.year)*12 + (end.month - start.month)+1
    if span_months > 60:
        ticks = pd.date_range(start, end, freq="YS")
        fmt = "%Y"
    elif span_months > 12:
        months = pd.date_range(start, end, freq="MS")
        ticks = [d for d in months if d.month in (1,7)]
        fmt = "%m-%y"
    else:
        ticks = pd.date_range(start, end, freq="MS")
        fmt = "%m-%y"
    if len(ticks) > 10:
        idx = np.linspace(0, len(ticks)-1, 10).round().astype(int)
        ticks = [ticks[i] for i in idx]
    return ticks, fmt

def label_for(code, show_full):
    s = "" if code is None else str(code)
    if not s or s=="<NA>":
        return ""
    if show_full:
        name = ent_map.get(s) or alias_map.get(s) or s
    else:
        name = alias_map.get(s) or ent_map.get(s) or s
    try:
        if name is None or (isinstance(name,float) and np.isnan(name)) or str(name) in ("","<NA>"):
            name = s
    except Exception:
        name = s
    return str(name)

def display_name(code, show_full):
    s = "" if code is None else str(code)
    lbl = label_for(s, show_full)
    if (not lbl or lbl==s) and s.startswith("AA"):
        return AA_GROUPS.get(s, s)
    return lbl

def find_variable_codes_by_hint(hints, top_k=4, sources=("indicadores","baldet")):
    codes = []
    low_map = {k:v for k,v in combined_var_map.items()}
    for hint in hints:
        hl = hint.lower()
        for k,v in low_map.items():
            # filtrar por fuente si corresponde
            is_bald = k.startswith("B_")
            if ("baldet" not in sources) and is_bald:
                continue
            if ("indicadores" not in sources) and (not is_bald):
                continue
            if hl in str(v).lower():
                if k not in codes and k not in HIDE_VARS:
                    codes.append(k)
            if len(codes)>=top_k:
                break
        if len(codes)>=top_k:
            break
    # fallback popular indicators
    if len(codes)<top_k and not df_ind.empty:
        for c in df_ind["codigo_indicador"].value_counts().index.tolist():
            if c not in codes and c not in HIDE_VARS and "indicadores" in sources:
                codes.append(c)
            if len(codes)>=top_k:
                break
    return codes[:top_k]

def get_unified_data_for_vars(var_codes, entidades=None, start_dt=None, end_dt=None):
    """
    Devuelve un DataFrame unificado con columnas:
    codigo_entidad, entidad, fecha_dt, codigo_indicador (prefijo B_ cuando es baldat),
    indicador, formato, valor
    """
    frames = []
    if not var_codes:
        return pd.DataFrame([], columns=["codigo_entidad","entidad","fecha_dt","codigo_indicador","indicador","formato","valor"])
    for vc in var_codes:
        if vc.startswith("B_"):
            # baldat
            codigo_cuenta = vc.split("B_",1)[1]
            sub = df_bald[df_bald["codigo_cuenta"].astype(str)==codigo_cuenta].copy()
            if entidades:
                sub = sub[sub["codigo_entidad"].isin(entidades)]
            if start_dt is not None:
                sub = sub[sub["fecha_dt"]>=start_dt]
            if end_dt is not None:
                sub = sub[sub["fecha_dt"]<=end_dt]
            if sub.empty:
                continue
            sub_u = sub[["codigo_entidad","entidad","fecha_dt","codigo_cuenta","cuenta","valor"]].copy()
            sub_u = sub_u.rename(columns={"codigo_cuenta":"codigo_indicador","cuenta":"indicador"})
            sub_u["codigo_indicador"] = "B_" + sub_u["codigo_indicador"].astype(str)
            sub_u["formato"] = "N"
            frames.append(sub_u[["codigo_entidad","entidad","fecha_dt","codigo_indicador","indicador","formato","valor"]])
        else:
            # indicador normal
            sub = df_ind[df_ind["codigo_indicador"]==vc].copy()
            if entidades:
                sub = sub[sub["codigo_entidad"].isin(entidades)]
            if start_dt is not None:
                sub = sub[sub["fecha_dt"]>=start_dt]
            if end_dt is not None:
                sub = sub[sub["fecha_dt"]<=end_dt]
            if sub.empty:
                continue
            sub_u = sub[["codigo_entidad","entidad","fecha_dt","codigo_indicador","indicador","formato","valor"]].copy()
            frames.append(sub_u)
    if not frames:
        return pd.DataFrame([], columns=["codigo_entidad","entidad","fecha_dt","codigo_indicador","indicador","formato","valor"])
    out = pd.concat(frames, ignore_index=True)
    out["formato"] = out["formato"].astype("string").str.upper().fillna("N")
    return out

# load datasets into memory for use
df_ind = load_indicadores()
df_bald = load_baldet()

# ---------- UI ----------
st.title("📊 Indicadores BCRA (v9.0) — indicadores + baldat")

show_full_names = st.checkbox("Mostrar nombre completo", value=False)
zero_mode = st.selectbox(
    "Tratamiento de valores 0.00 (posibles faltantes)",
    options=["Mantener", "Ignorar (saltear meses)", "Reemplazar por dato previo (ffill)"],
    index=0
)

if df_ind.empty and df_bald.empty:
    st.warning("No se encontraron datos en ./data. Subí los CSV requeridos.")
    st.stop()

# pestañas
tab_panel, tab_serie, tab_calc, tab_rank, tab_sys = st.tabs(["Panel","Serie","Calculadora","Rankings","Sistema financiero"])

# ---------------- PANEL ----------------
with tab_panel:
    st.subheader("Panel")
    c1,c2,c3 = st.columns([1,1,1])
    with c1:
        ent_codes = [str(c) for c in pd.concat([df_ind["codigo_entidad"], df_bald["codigo_entidad"]]).dropna().unique()]
        ent_codes_sorted = sorted(set(ent_codes), key=lambda c: display_name(c, show_full_names))
        default_ent_idx = ent_codes_sorted.index(DEFAULT_ENTITY_CODE) if DEFAULT_ENTITY_CODE in ent_codes_sorted else 0
        ent_code = st.selectbox("Entidad", options=ent_codes_sorted, index=default_ent_idx, format_func=lambda code: display_name(code, show_full_names))
    with c2:
        # checkboxes de fuentes para variables
        cols_src = st.columns([1,1])
        with cols_src[0]:
            include_ind = st.checkbox("Indicadores", value=True, help="Incluir variables provenientes de bcra_indicadores")
        with cols_src[1]:
            include_bald = st.checkbox("Baldet (cuentas)", value=False, help="Incluir variables provenientes de baldet (cuentas patrimoniales)")
        sources = []
        if include_ind: sources.append("indicadores")
        if include_bald: sources.append("baldet")
        # construir lista de variables según fuentes elegidas
        var_options = []
        if include_ind:
            var_options += [c for c in var_map_ind.keys() if c not in HIDE_VARS]
        if include_bald:
            var_options += list(bald_var_map.keys())
        var_options = sorted(set(var_options), key=lambda c: combined_var_map.get(c, c))
        default_var_codes = [c for c in find_variable_codes_by_hint(DEFAULT_VAR_HINTS, top_k=4, sources=sources) if c in var_options]
        var_sel = st.multiselect("Variables", options=var_options, default=default_var_codes, format_func=lambda code: combined_var_map.get(code, code))
    with c3:
        all_months = sorted(pd.concat([df_ind["fecha_dt"], df_bald["fecha_dt"]]).dropna().unique().tolist())
        def_month = pd.to_datetime(DEFAULT_MONTH_STR + "-01", format="%Y-%m-%d", errors="coerce")
        if def_month not in all_months and len(all_months)>0:
            def_month = all_months[-1]
        month = st.selectbox("Mes", options=all_months, index=(all_months.index(def_month) if def_month in all_months else len(all_months)-1), format_func=lambda d: d.strftime("%Y-%m"))

    if not var_sel:
        st.info("Seleccioná al menos una variable.")
    else:
        rows = []
        for v in var_sel:
            desc = combined_var_map.get(v, v)
            fmt = combined_fmt_map.get(v, "N")
            # tomar valor actual (para la entidad y mes)
            if v.startswith("B_"):
                codigo_cuenta = v.split("B_",1)[1]
                row_now = df_bald[(df_bald["codigo_entidad"]==ent_code) & (df_bald["codigo_cuenta"].astype(str)==codigo_cuenta) & (df_bald["fecha_dt"]==month)]
                val_now = row_now["valor"].iloc[0] if not row_now.empty else np.nan
                row_prev = df_bald[(df_bald["codigo_entidad"]==ent_code) & (df_bald["codigo_cuenta"].astype(str)==codigo_cuenta) & (df_bald["fecha_dt"]== (month - pd.offsets.MonthBegin(1)))]
                val_prev = row_prev["valor"].iloc[0] if not row_prev.empty else np.nan
                row_yoy = df_bald[(df_bald["codigo_entidad"]==ent_code) & (df_bald["codigo_cuenta"].astype(str)==codigo_cuenta) & (df_bald["fecha_dt"]== (month - pd.DateOffset(years=1)))]
                val_yoy = row_yoy["valor"].iloc[0] if not row_yoy.empty else np.nan
            else:
                row_now = df_ind[(df_ind["codigo_entidad"]==ent_code) & (df_ind["codigo_indicador"]==v) & (df_ind["fecha_dt"]==month)]
                val_now = row_now["valor"].iloc[0] if not row_now.empty else np.nan
                row_prev = df_ind[(df_ind["codigo_entidad"]==ent_code) & (df_ind["codigo_indicador"]==v) & (df_ind["fecha_dt"]== (month - pd.offsets.MonthBegin(1)))]
                val_prev = row_prev["valor"].iloc[0] if not row_prev.empty else np.nan
                row_yoy = df_ind[(df_ind["codigo_entidad"]==ent_code) & (df_ind["codigo_indicador"]==v) & (df_ind["fecha_dt"]== (month - pd.DateOffset(years=1)))]
                val_yoy = row_yoy["valor"].iloc[0] if not row_yoy.empty else np.nan
            rows.append({"Variable":desc,"Código":v,"Formato":fmt,"Actual":val_now,"Previo":val_prev,"YoY":val_yoy})
        panel_df = pd.DataFrame(rows)
        ncols = 2 if len(panel_df)<=2 else 4
        cols = st.columns(ncols)
        for i,(_,r) in enumerate(panel_df.iterrows()):
            with cols[i % ncols]:
                st.markdown(f"**{r['Variable']}**")
                st.markdown(f"### {format_value(r['Actual'], r['Formato'])}")
                mm = percent_change(r["Actual"], r["Previo"])
                yy = percent_change(r["Actual"], r["YoY"])
                sign_mm = "+" if (mm is not None and mm>=0) else "−"
                color_mm = "#2e7d32" if (mm is not None and mm>=0) else "#c62828"
                sign_yy = "+" if (yy is not None and yy>=0) else "−"
                color_yy = "#2e7d32" if (yy is not None and yy>=0) else "#c62828"
                mm_txt = f'<span style="color:{color_mm}">{(sign_mm + f"{abs(mm)*100:.1f}%") if mm is not None else "—"}</span>'
                yy_txt = f'<span style="color:{color_yy}">{(sign_yy + f"{abs(yy)*100:.1f}%") if yy is not None else "—"}</span>'
                st.markdown(f'<span style="font-size:0.9rem;color:#666">M/M-1: {mm_txt} · A/A-1: {yy_txt}</span>', unsafe_allow_html=True)

# ---------------- SERIE ----------------
with tab_serie:
    st.subheader("Serie")
    left, right = st.columns([1,2])
    with left:
        # fuentes
        src1, src2 = st.columns([1,1])
        with src1:
            include_ind_s = st.checkbox("Indicadores", value=True, key="serie_src_ind")
        with src2:
            include_bald_s = st.checkbox("Baldet", value=False, key="serie_src_bald")
        sources = []
        if include_ind_s: sources.append("indicadores")
        if include_bald_s: sources.append("baldet")

        all_ent = [str(c) for c in pd.concat([df_ind["codigo_entidad"], df_bald["codigo_entidad"]]).dropna().unique()]
        ents = st.multiselect("Entidades", options=sorted(set(all_ent), key=lambda c: display_name(c, show_full_names)),
                              default=[DEFAULT_ENTITY_CODE] if DEFAULT_ENTITY_CODE in all_ent else [],
                              format_func=lambda c: display_name(c, show_full_names))
        # variables según fuentes
        var_options = []
        if "indicadores" in sources:
            var_options += [c for c in var_map_ind.keys() if c not in HIDE_VARS]
        if "baldet" in sources:
            var_options += list(bald_var_map.keys())
        var_options = sorted(set(var_options), key=lambda c: combined_var_map.get(c,c))
        vars_series = st.multiselect("Variables", options=var_options,
                                    default=find_variable_codes_by_hint(DEFAULT_VAR_HINTS, top_k=2, sources=sources),
                                    format_func=lambda v: combined_var_map.get(v,v))
        min_dt = pd.to_datetime(min(pd.concat([df_ind["fecha_dt"].dropna(), df_bald["fecha_dt"].dropna()]))).date()
        max_dt = pd.to_datetime(max(pd.concat([df_ind["fecha_dt"].dropna(), df_bald["fecha_dt"].dropna()]))).date()
        default_start = date(max_dt.year-5, max_dt.month, 1) if (max_dt.year-min_dt.year)>=5 else min_dt
        date_range = st.slider("Rango temporal", min_value=min_dt, max_value=max_dt, value=(default_start, max_dt))
    with right:
        if not ents or not vars_series:
            st.info("Seleccioná al menos una entidad y una variable.")
        else:
            start_dt = pd.to_datetime(date_range[0])
            end_dt = pd.to_datetime(date_range[1])
            final_plot = get_unified_data_for_vars(vars_series, entidades=ents, start_dt=start_dt, end_dt=end_dt)
            if final_plot.empty:
                st.info("No hay datos para graficar en el rango seleccionado.")
            else:
                tvals, tfmt = tick_values_for_range(final_plot["fecha_dt"].min(), final_plot["fecha_dt"].max())
                base = alt.Chart(final_plot).mark_line(point=True).encode(
                    x=alt.X("fecha_dt:T", title="Mes", axis=alt.Axis(values=tvals, format=tfmt, labelOverlap=True, tickCount=len(tvals))),
                    y=alt.Y("valor:Q", title="Valor"),
                    color=alt.Color("entidad:N", title="Entidad", legend=alt.Legend(orient="bottom"))
                ).properties(height=320)
                chart = base.facet(facet=alt.Facet("indicador:N", title=None), columns=1).resolve_scale(y="independent")
                st.altair_chart(chart, use_container_width=True)

# ---------------- CALCULADORA ----------------
with tab_calc:
    st.subheader("Calculadora (puede mezclar indicadores y baldet)")
    # Fuentes
    c1,c2 = st.columns([1,2])
    with c1:
        ci = st.checkbox("Indicadores", value=True, key="calc_src_ind")
        cb = st.checkbox("Baldet", value=False, key="calc_src_bald")
        sources = []
        if ci: sources.append("indicadores")
        if cb: sources.append("baldet")
        ents2 = st.multiselect("Entidades", options=sorted(set([str(c) for c in pd.concat([df_ind["codigo_entidad"], df_bald["codigo_entidad"]]).dropna().unique()]), key=lambda c: display_name(c, show_full_names)),
                               default=[DEFAULT_ENTITY_CODE] if DEFAULT_ENTITY_CODE in df_ind["codigo_entidad"].astype(str).tolist() else [])
        # variables pool
        var_pool=[]
        if "indicadores" in sources: var_pool += [c for c in var_map_ind.keys() if c not in HIDE_VARS]
        if "baldet" in sources: var_pool += list(bald_var_map.keys())
        var_pool = sorted(set(var_pool), key=lambda c: combined_var_map.get(c,c))
        ops = ["+","-","*","/"]
        term_vars=[]; term_ops=[]
        for i in range(1,6):
            v = st.selectbox(f"Variable {i}", options=["—"]+var_pool, index=0, format_func=lambda x: combined_var_map.get(x,"—") if x!="—" else "—", key=f"calc_var_{i}")
            term_vars.append(None if v=="—" else v)
            if i<5:
                op = st.selectbox(f"Operación {i}→{i+1}", options=["—"]+ops, index=0, key=f"calc_op_{i}")
                term_ops.append(None if op=="—" else op)
        min_dt = pd.to_datetime(min(pd.concat([df_ind["fecha_dt"].dropna(), df_bald["fecha_dt"].dropna()]))).date()
        max_dt = pd.to_datetime(max(pd.concat([df_ind["fecha_dt"].dropna(), df_bald["fecha_dt"].dropna()]))).date()
        range_calc = st.slider("Rango temporal", min_value=min_dt, max_value=max_dt, value=(min_dt,max_dt))
    with c2:
        active_terms = [t for t in term_vars if t]
        if not ents2 or not active_terms:
            st.info("Elegí al menos 1 variable y una entidad.")
        else:
            vars_sel = active_terms
            pivot_df = get_unified_data_for_vars(vars_sel, entidades=ents2, start_dt=pd.to_datetime(range_calc[0]), end_dt=pd.to_datetime(range_calc[1]))
            if pivot_df.empty:
                st.info("No hay datos en ese rango.")
            else:
                # pivot wide: codigo_entidad x fecha_dt
                wide = pivot_df.pivot_table(index=["codigo_entidad","fecha_dt"], columns="codigo_indicador", values="valor", aggfunc="last").reset_index()
                wide = wide.sort_values(["codigo_entidad","fecha_dt"])
                # apply zero mode
                if zero_mode.startswith("Ignorar"):
                    for v in vars_sel:
                        if v in wide.columns:
                            wide.loc[wide[v]==0, v] = np.nan
                elif zero_mode.startswith("Reemplazar"):
                    for v in vars_sel:
                        if v in wide.columns:
                            wide.loc[wide[v]==0, v] = np.nan
                            wide[v] = wide.groupby("codigo_entidad")[v].ffill()
                def apply_expr(row):
                    nums=[]
                    for v in term_vars:
                        if v:
                            nums.append(row.get(v, np.nan))
                    if not nums:
                        return np.nan
                    res = nums[0]
                    idx=0
                    for j in range(1,len(nums)):
                        op = term_ops[idx] if idx < len(term_ops) else None
                        val = nums[j]; idx+=1
                        if op is None or pd.isna(res) or pd.isna(val):
                            continue
                        try:
                            if op=="+": res = res + val
                            elif op=="-": res = res - val
                            elif op=="*": res = res * val
                            elif op=="/": res = np.nan if val==0 else res / val
                        except Exception:
                            res = np.nan
                    return res
                wide["Resultado"] = wide.apply(apply_expr, axis=1)
                wide["Entidad"] = wide["codigo_entidad"].apply(lambda c: display_name(c, show_full_names))
                tvals, tfmt = tick_values_for_range(wide["fecha_dt"].min(), wide["fecha_dt"].max())
                chart = alt.Chart(wide.dropna(subset=["Resultado"])).mark_line(point=True).encode(
                    x=alt.X("fecha_dt:T", title="Mes", axis=alt.Axis(values=tvals, format=tfmt, tickCount=len(tvals))),
                    y=alt.Y("Resultado:Q", title="Resultado"),
                    color=alt.Color("Entidad:N", title="Entidad", legend=alt.Legend(orient="bottom"))
                ).properties(height=420)
                st.altair_chart(chart, use_container_width=True)
                with st.expander("Ver datos"):
                    st.dataframe(wide[["Entidad","fecha_dt","Resultado"]].sort_values(["Entidad","fecha_dt"]), use_container_width=True)

# ---------------- RANKINGS ----------------
with tab_rank:
    st.subheader("Rankings")
    left, right = st.columns([1,2])
    with left:
        ind_cb = st.checkbox("Indicadores", value=True, key="rank_ind")
        bald_cb = st.checkbox("Baldet", value=False, key="rank_bald")
        sources=[]
        if ind_cb: sources.append("indicadores")
        if bald_cb: sources.append("baldet")
        var_rank = st.selectbox("Variable", options=sorted([v for v in combined_var_map.keys() if ((v.startswith("B_") and "baldet" in sources) or (not v.startswith("B_") and "indicadores" in sources)) and v not in HIDE_VARS], key=lambda v: combined_var_map.get(v,v)), format_func=lambda v: combined_var_map.get(v,v))
        topn = st.number_input("Top N", min_value=3, max_value=50, value=10, step=1)
        min_dt = pd.to_datetime(min(pd.concat([df_ind["fecha_dt"].dropna(), df_bald["fecha_dt"].dropna()]))).date()
        max_dt = pd.to_datetime(max(pd.concat([df_ind["fecha_dt"].dropna(), df_bald["fecha_dt"].dropna()]))).date()
        default_start = date(max_dt.year-1, max_dt.month, 1)
        range_rank = st.slider("Rango temporal", min_value=min_dt, max_value=max_dt, value=(default_start,max_dt))
    with right:
        # preparar df combinado para la variable
        partir = get_unified_data_for_vars([var_rank], start_dt=pd.to_datetime(range_rank[0]), end_dt=pd.to_datetime(range_rank[1]))
        if partir.empty:
            st.info("No hay datos para ese rango/variable.")
        else:
            # excluir AA* del ranking (solo entidades)
            df_rank = partir[~partir["codigo_entidad"].astype(str).str.startswith("AA", na=False)].copy()
            if df_rank.empty:
                st.info("No hay datos para ranking.")
            else:
                # promedio sin ceros
                df_nz = df_rank[df_rank["valor"]!=0].copy()
                if df_nz.empty:
                    st.info("No hay datos no-cero para calcular promedios.")
                else:
                    avg = df_nz.groupby("codigo_entidad", as_index=False)["valor"].mean().rename(columns={"valor":"metric"})
                    avg["Entidad"] = avg["codigo_entidad"].apply(lambda c: display_name(c, show_full_names))
                    top_tot = avg.sort_values("metric", ascending=False).head(int(topn))
                    st.markdown("### 🏆 Top valores totales (promedio sin 0.00)")
                    chart_tot = alt.Chart(top_tot).mark_bar().encode(
                        x=alt.X("metric:Q", title="Promedio"),
                        y=alt.Y("Entidad:N", sort="-x", title=None),
                        tooltip=[alt.Tooltip("Entidad:N"), alt.Tooltip("metric:Q", title="Promedio", format=",.2f")]
                    ).properties(height=28*len(top_tot)+20)
                    text_tot = chart_tot.mark_text(align="left", dx=3).encode(text=alt.Text("metric:Q", format=",.2f"))
                    st.altair_chart(chart_tot + text_tot, use_container_width=True)
                # variación % (ultimo/primero)
                s = df_rank.sort_values(["codigo_entidad","fecha_dt"])
                first = s.groupby("codigo_entidad", as_index=False).first()[["codigo_entidad","valor"]].rename(columns={"valor":"first"})
                last = s.groupby("codigo_entidad", as_index=False).last()[["codigo_entidad","valor"]].rename(columns={"valor":"last"})
                ch = pd.merge(first,last,on="codigo_entidad",how="inner")
                ch = ch[(~ch["first"].isna()) & (~ch["last"].isna()) & (ch["first"]!=0)]
                ch["delta"] = ch["last"]/ch["first"] - 1.0
                ch["Entidad"] = ch["codigo_entidad"].apply(lambda c: display_name(c, show_full_names))
                st.markdown("### 📈 Mayores subas / 📉 Mayores bajas")
                up = ch.sort_values("delta", ascending=False).head(int(topn))
                dn = ch.sort_values("delta", ascending=True).head(int(topn))
                if not up.empty:
                    chart_up = alt.Chart(up).mark_bar().encode(x=alt.X("delta:Q", axis=alt.Axis(format=".1%")), y=alt.Y("Entidad:N", sort="-x", title=None),
                                                               tooltip=[alt.Tooltip("Entidad:N"), alt.Tooltip("delta:Q",format=".1%")]).properties(height=28*len(up)+20)
                    st.altair_chart(chart_up, use_container_width=True)
                if not dn.empty:
                    chart_dn = alt.Chart(dn).mark_bar().encode(x=alt.X("delta:Q", axis=alt.Axis(format=".1%")), y=alt.Y("Entidad:N", title=None),
                                                               tooltip=[alt.Tooltip("Entidad:N"), alt.Tooltip("delta:Q",format=".1%")]).properties(height=28*len(dn)+20)
                    st.altair_chart(chart_dn, use_container_width=True)

# ---------------- SISTEMA FINANCIERO ----------------
with tab_sys:
    st.subheader("Sistema financiero")
    st.markdown("Evolución de cantidades y comparador por agrupaciones (AA*)")
    # A) Evolución de cantidades (tomamos indicadores HIDE_VARS)
    sys_df = df_ind[df_ind["codigo_indicador"].isin(HIDE_VARS)].copy()
    if sys_df.empty:
        st.info("No se encontraron series del tablero de sistema.")
    else:
        sys_df = sys_df.groupby(["fecha_dt","codigo_indicador"], as_index=False).agg(indicador=("indicador","last"), formato=("formato","last"), valor=("valor","last"))
        sys_df = sys_df.sort_values("fecha_dt")
        sys_df["ValorPlot"] = to_plot_series(sys_df["valor"], "N")
        tvals, tfmt = tick_values_for_range(sys_df["fecha_dt"].min(), sys_df["fecha_dt"].max())
        chart_sys = alt.Chart(sys_df).mark_line(point=True).encode(
            x=alt.X("fecha_dt:T", title="Mes", axis=alt.Axis(values=tvals, format=tfmt, tickCount=len(tvals))),
            y=alt.Y("ValorPlot:Q", title="Cantidad"),
            color=alt.Color("indicador:N", title="Indicador", legend=alt.Legend(orient="bottom", columns=2)),
            tooltip=[alt.Tooltip("indicador:N"), alt.Tooltip("fecha_dt:T"), alt.Tooltip("valor:Q", format=",.0f")]
        ).properties(height=360)
        st.altair_chart(chart_sys, use_container_width=True)
    st.divider()
    # B) Comparador por AA* (en baldet la agrupaciones AA* vienen como codigo_entidad AAxxx)
    cL,cR = st.columns([1,2])
    with cL:
        var_sel_sys = st.selectbox("Variable (no-sistema)", options=sorted([v for v in combined_var_map.keys() if v not in HIDE_VARS], key=lambda v: combined_var_map.get(v,v)), format_func=lambda v: combined_var_map.get(v,v))
        aa_pick = st.multiselect("Agrupaciones (AA*)", options=list(AA_GROUPS.keys()), default=list(AA_GROUPS.keys()), format_func=lambda c: display_name(c, show_full_names))
        range_sys = st.slider("Rango temporal", min_value=pd.to_datetime(min(pd.concat([df_ind["fecha_dt"].dropna(), df_bald["fecha_dt"].dropna()]))).date(), max_value=pd.to_datetime(max(pd.concat([df_ind["fecha_dt"].dropna(), df_bald["fecha_dt"].dropna()]))).date(), value=(pd.to_datetime(min(pd.concat([df_ind["fecha_dt"].dropna(), df_bald["fecha_dt"].dropna()]))).date(), pd.to_datetime(max(pd.concat([df_ind["fecha_dt"].dropna(), df_bald["fecha_dt"].dropna()]))).date()))
    with cR:
        sub = get_unified_data_for_vars([var_sel_sys], entidades=aa_pick, start_dt=pd.to_datetime(range_sys[0]), end_dt=pd.to_datetime(range_sys[1]))
        if sub.empty:
            st.info("No hay datos para la selección.")
        else:
            sub = sub.sort_values("fecha_dt")
            sub["ValorPlot"] = to_plot_series(sub["valor"], sub["formato"].iloc[0] if "formato" in sub.columns else "N")
            tvals, tfmt = tick_values_for_range(sub["fecha_dt"].min(), sub["fecha_dt"].max())
            y_axis = alt.Axis(title="Valor", format=(".1%" if (sub["formato"].iloc[0]=="P" if "formato" in sub.columns else False) else None))
            chart_cmp = alt.Chart(sub).mark_line(point=True).encode(
                x=alt.X("fecha_dt:T", title="Mes", axis=alt.Axis(values=tvals, format=tfmt, tickCount=len(tvals))),
                y=alt.Y("ValorPlot:Q", axis=y_axis),
                color=alt.Color("entidad:N", title="Agrupación", legend=alt.Legend(orient="bottom")),
                tooltip=[alt.Tooltip("entidad:N"), alt.Tooltip("fecha_dt:T"), alt.Tooltip("valor:Q")]
            ).properties(height=360)
            st.altair_chart(chart_cmp, use_container_width=True)

st.caption("Fuente: TXT BCRA (Entfin/Tec_Cont) + BALDAT. v9.0 — Usa checkboxes para elegir fuentes de variables.")
