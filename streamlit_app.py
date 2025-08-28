# streamlit_app.py (v8.1 -> v8.1.2)
# Requiere: streamlit pandas altair numpy
# CSVs esperados en ./data:
#   - bcra_consolidado.csv (o bcra_consolidado_part*.csv)
#   - bcra_nomina.csv  (con columnas: codigo_entidad, entidad, alias)
#   - bcra_indicadores.csv
#   - (opcional) bcra_agregados.csv con columnas:
#       fecha (YYYY-MM), codigo_indicador, indicador, formato,
#       sistema_financiero, banca_publica, banca_privada,
#       banca_nacional, banca_extranjera, companias_financieras

import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from pathlib import Path
import glob
from datetime import date

st.set_page_config(page_title="Indicadores BCRA (v8.1.2)", layout="wide")

DATA_DIR = Path("./data")
MASTER = DATA_DIR / "bcra_consolidado.csv"
PARTS = sorted([Path(p) for p in glob.glob(str(DATA_DIR / "bcra_consolidado_part*.csv"))])
NOMINA = DATA_DIR / "bcra_nomina.csv"
INDICES = DATA_DIR / "bcra_indicadores.csv"
AGREGADOS = DATA_DIR / "bcra_agregados.csv"  # opcional

DEFAULT_ENTITY_CODE = "00011"  # Banco Nación
DEFAULT_MONTH_STR = "2025-05"
DEFAULT_VAR_HINTS = ["Dotación de personal", "ROE", "ROA", "Gastos en personal"]

# Indicadores del tablero de Sistema (a ocultar en otras páginas)
HIDE_VARS = {
    "110000000001","110000000002","110000000003","110000000004","110000000005","110000000006",
    "110000000007","110000000008","110000000009","110000000010","110000000011"
}

# Agrupaciones AA (para comparador del Sistema)
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
def load_consolidado() -> pd.DataFrame:
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
        return pd.DataFrame(columns=dtypes.keys()).astype(dtypes)
    frames = []
    for f in files:
        df = _read_csv_safe(f, dtype=dtypes)
        if df is not None:
            frames.append(df)
    if not frames:
        return pd.DataFrame(columns=dtypes.keys()).astype(dtypes)
    df_all = pd.concat(frames, ignore_index=True)
    df_all["formato"] = df_all["formato"].astype("string").str.upper().fillna("N")
    df_all["fecha"] = df_all["fecha"].astype("string").str.strip()
    df_all["fecha_dt"] = pd.to_datetime(df_all["fecha"] + "-01", format="%Y-%m-%d", errors="coerce")
    df_all = df_all.dropna(subset=["fecha_dt"]).sort_values(
        ["fecha_dt", "codigo_entidad", "codigo_indicador"]
    ).reset_index(drop=True)
    return df_all

@st.cache_data(show_spinner=False)
def load_nomina_maps():
    df = _read_csv_safe(NOMINA)
    if df is None or df.empty:
        return {}, {}
    # normalizar nombres de columnas (tolerante a mayúsculas)
    cols = {c.lower(): c for c in df.columns}
    code_col = cols.get("codigo_entidad", "codigo_entidad")
    name_col = cols.get("entidad", "entidad")
    alias_col = cols.get("alias", None)

    # convertir a strings "limpios" (sin <NA>, sin None)
    def to_clean_str_series(s):
        if s is None:
            return pd.Series([], dtype="string")
        s = s.astype("string").fillna("").astype(str).str.strip()
        s = s.replace({"<NA>": ""})
        return s

    codes = to_clean_str_series(df[code_col])
    names = to_clean_str_series(df[name_col])
    aliases = to_clean_str_series(df[alias_col]) if alias_col else names.copy()

    # si alias está vacío, caer al nombre largo
    alias_use = [a if a else n for a, n in zip(aliases, names)]

    # construir mapas con claves/valores str
    alias_map = dict(zip(codes.tolist(), alias_use))
    ent_map = dict(zip(codes.tolist(), names.tolist()))
    return ent_map, alias_map

@st.cache_data(show_spinner=False)
def load_indices():
    dtypes = {"codigo_indicador": "string", "indicador": "string", "formato": "string"}
    df = _read_csv_safe(INDICES, dtype=dtypes)
    if df is None or df.empty:
        return pd.DataFrame(columns=dtypes.keys()).astype(dtypes), {}, {}
    df["formato"] = df["formato"].astype("string").str.upper().fillna("N")
    var_map = dict(zip(df["codigo_indicador"], df["indicador"]))
    fmt_map = dict(zip(df["codigo_indicador"], df["formato"]))
    return df, var_map, fmt_map

@st.cache_data(show_spinner=False)
def load_agregados():
    if not AGREGADOS.is_file():
        return None
    df = _read_csv_safe(AGREGADOS)
    if df is None or df.empty:
        return None
    df["fecha"] = df["fecha"].astype(str)
    df["fecha_dt"] = pd.to_datetime(df["fecha"].str.strip() + "-01", format="%Y-%m-%d", errors="coerce")
    df = df.dropna(subset=["fecha_dt"]).sort_values(["fecha_dt", "codigo_indicador"]).reset_index(drop=True)
    if "formato" in df.columns:
        df["formato"] = df["formato"].astype("string").str.upper().fillna("N")
    return df

df = load_consolidado()
ent_map, alias_map = load_nomina_maps()
idx_df, var_map, fmt_map = load_indices()
agg_df = load_agregados()

# ---------- helpers ----------
def percent_change(curr, prev):
    if curr is None or prev in (None, 0) or pd.isna(curr) or pd.isna(prev) or prev == 0:
        return None
    return (curr / prev) - 1.0

def format_value(val: float, fmt: str, decimals=2) -> str:
    if pd.isna(val):
        return "—"
    if (fmt or "").upper() == "P":
        return f"{val:.{decimals}f}%"
    return f"{val:,.{decimals}f}".replace(",", "X").replace(".", ",").replace("X", ".")

def to_plot_series(series_vals: pd.Series, fmt_code: str) -> pd.Series:
    return series_vals / 100.0 if str(fmt_code).upper() == "P" else series_vals

def find_variable_codes_by_hint(hints, top_k=4):
    codes = []
    if var_map:
        for hint in hints:
            hint_low = hint.lower()
            matches = [c for c, d in var_map.items() if hint_low in str(d).lower()]
            for m in matches:
                if m not in codes and m not in HIDE_VARS:
                    codes.append(m)
            if len(codes) >= top_k:
                break
    if len(codes) < top_k and not df.empty:
        for c in df["codigo_indicador"].value_counts().index.tolist():
            if c not in codes and c not in HIDE_VARS:
                codes.append(c)
            if len(codes) >= top_k:
                break
    return codes[:top_k]

def default_month_value():
    candidate = pd.to_datetime(DEFAULT_MONTH_STR + "-01", format="%Y-%m-%d", errors="coerce")
    if candidate is not pd.NaT and not df.empty and candidate in df["fecha_dt"].unique():
        return candidate
    return df["fecha_dt"].max() if not df.empty else None

def tick_values_for_range(start_dt: pd.Timestamp, end_dt: pd.Timestamp):
    """Genera ticks (fechas) para el eje X sin agrupar datos; ≤10 etiquetas."""
    start = pd.Timestamp(start_dt.year, start_dt.month, 1)
    end = pd.Timestamp(end_dt.year, end_dt.month, 1)
    span_months = (end.year - start.year) * 12 + (end.month - start.month) + 1

    if span_months > 60:
        ticks = pd.date_range(start, end, freq="YS")  # 1 enero de cada año
        fmt = "%Y"
    elif span_months > 12:
        months = pd.date_range(start, end, freq="MS")
        ticks = [d for d in months if d.month in (1, 7)]  # ene/jul
        fmt = "%m-%y"
    else:
        ticks = pd.date_range(start, end, freq="MS")
        fmt = "%m-%y"

    if len(ticks) > 10:
        idx = np.linspace(0, len(ticks) - 1, 10).round().astype(int)
        ticks = [ticks[i] for i in idx]
    return ticks, fmt

def treat_zeros_long(df_long: pd.DataFrame, value_col: str, group_cols: list, mode: str):
    """Aplica el tratamiento de ceros a un DF largo (de series)."""
    if df_long.empty:
        return df_long
    if mode.startswith("Mantener"):
        return df_long
    out = df_long.copy()
    if mode.startswith("Ignorar"):
        return out.loc[out[value_col] != 0].copy()
    # ffill
    out.loc[out[value_col] == 0, value_col] = np.nan
    sort_cols = [c for c in group_cols if c in out.columns] + (["fecha_dt"] if "fecha_dt" in out.columns else [])
    if sort_cols:
        out = out.sort_values(sort_cols)
    if group_cols:
        out[value_col] = out.groupby([c for c in group_cols if c in out.columns], group_keys=False)[value_col].ffill()
    else:
        out[value_col] = out[value_col].ffill()
    return out

def label_for(code: str, show_full: bool) -> str:
    """Devuelve SIEMPRE un str: ALIAS (default) o nombre completo si show_full=True."""
    s = "" if code is None else str(code)
    if not s or s == "<NA>":
        return ""
    if show_full:
        name = ent_map.get(s) or alias_map.get(s) or s
    else:
        name = alias_map.get(s) or ent_map.get(s) or s
    # blindaje contra NaN/NA
    try:
        if name is None or (isinstance(name, float) and np.isnan(name)) or str(name) in ("", "<NA>"):
            name = s
    except Exception:
        name = s
    return str(name)

def display_name(code: str, show_full: bool) -> str:
    """Etiqueta preferida para entidades y AA*: alias/nombre o fallback a AA_GROUPS."""
    s = "" if code is None else str(code)
    lbl = label_for(s, show_full)
    if (not lbl or lbl == s) and s.startswith("AA"):
        return AA_GROUPS.get(s, s)
    return lbl

# ---------- UI global ----------
st.title("📊 Indicadores del BCRA (v8.1.2)")

# Control alias/nombre
show_full_names = st.checkbox("Mostrar nombre completo", value=False, help="Si está desactivado, se muestran ALIAS.")

# Control global de tratamiento de ceros
zero_mode = st.selectbox(
    "Tratamiento de valores 0.00 (posibles faltantes)",
    options=["Mantener", "Ignorar (saltear meses)", "Reemplazar por dato previo (ffill)"],
    index=0,
    help="Si hay meses en 0.00 por falta de dato, podés saltearlos o rellenar con el valor del mes anterior."
)

if df.empty:
    st.warning("No se encontraron CSV en `./data`. Subí `bcra_consolidado.csv` (o partes), `bcra_nomina.csv` y `bcra_indicadores.csv`.")
    st.stop()

# Tabs (Panel, Serie, Calculadora, Rankings, Sistema)
tab_panel, tab_serie, tab_calc, tab_rank, tab_sys = st.tabs(["Panel", "Serie", "Calculadora", "Rankings", "Sistema financiero"])

# ===================== PANEL =====================
with tab_panel:
    st.subheader("Panel")
    colA, colB, colC = st.columns([1, 1, 1])
    with colA:
        ent_codes = [str(c) for c in df["codigo_entidad"].astype("string").tolist() if pd.notna(c) and str(c) != "<NA>" and str(c)]
        ent_codes_sorted = sorted(set(ent_codes), key=lambda c: display_name(c, show_full_names))
        default_ent_idx = ent_codes_sorted.index(DEFAULT_ENTITY_CODE) if DEFAULT_ENTITY_CODE in ent_codes_sorted else 0
        ent_code = st.selectbox(
            "Entidad", options=ent_codes_sorted, index=default_ent_idx,
            format_func=lambda code: display_name(code, show_full_names),
            key="panel_ent"
        )
    with colB:
        all_vars = sorted([v for v in df["codigo_indicador"].dropna().astype(str).unique().tolist() if v not in HIDE_VARS],
                          key=lambda v: var_map.get(v, v))
        default_var_codes = [c for c in find_variable_codes_by_hint(DEFAULT_VAR_HINTS, top_k=4) if c in all_vars]
        var_sel = st.multiselect(
            "Variables", options=all_vars,
            default=default_var_codes,
            format_func=lambda code: var_map.get(code, code),
            key="panel_vars"
        )
    with colC:
        all_months = sorted(df["fecha_dt"].unique().tolist())
        def_month = default_month_value()
        month = st.selectbox(
            "Mes", options=all_months,
            index=(all_months.index(def_month) if def_month in all_months else len(all_months) - 1),
            format_func=lambda d: d.strftime("%Y-%m"),
            key="panel_month"
        )

    if not var_sel:
        st.info("Seleccioná al menos una variable.")
        st.stop()

    rows = []
    for vcode in var_sel:
        vdesc = var_map.get(vcode, vcode)
        fmt = str(fmt_map.get(vcode, "N")).upper()
        # actual
        row_now = df[(df["codigo_entidad"] == ent_code) & (df["codigo_indicador"] == vcode) & (df["fecha_dt"] == month)]
        val_now = row_now["valor"].iloc[0] if not row_now.empty else np.nan
        # m-1
        month_prev = month - pd.offsets.MonthBegin(1)
        row_prev = df[(df["codigo_entidad"] == ent_code) & (df["codigo_indicador"] == vcode) & (df["fecha_dt"] == month_prev)]
        val_prev = row_prev["valor"].iloc[0] if not row_prev.empty else np.nan
        # yoy
        month_yoy = month - pd.DateOffset(years=1)
        row_yoy = df[(df["codigo_entidad"] == ent_code) & (df["codigo_indicador"] == vcode) & (df["fecha_dt"] == month_yoy)]
        val_yoy = row_yoy["valor"].iloc[0] if not row_yoy.empty else np.nan

        rows.append({
            "Variable": vdesc, "Código": vcode, "Formato": fmt,
            "Actual": val_now, "Previo": val_prev, "YoY": val_yoy
        })
    panel_df = pd.DataFrame(rows)

    ncols = 2 if len(panel_df) <= 2 else 4
    cols = st.columns(ncols)
    for i, (_, r) in enumerate(panel_df.iterrows()):
        with cols[i % ncols]:
            st.markdown(f"**{r['Variable']}**")
            st.markdown(f"### {format_value(r['Actual'], r['Formato'])}")
            mm = percent_change(r["Actual"], r["Previo"])
            yy = percent_change(r["Actual"], r["YoY"])
            sign_mm = "+" if (mm is not None and mm >= 0) else "−"
            color_mm = "#2e7d32" if (mm is not None and mm >= 0) else "#c62828"
            sign_yy = "+" if (yy is not None and yy >= 0) else "−"
            color_yy = "#2e7d32" if (yy is not None and yy >= 0) else "#c62828"
            mm_txt = f'<span style="color:{color_mm}">{(sign_mm + f"{abs(mm)*100:.1f}%") if mm is not None else "—"}</span>'
            yy_txt = f'<span style="color:{color_yy}">{(sign_yy + f"{abs(yy)*100:.1f}%") if yy is not None else "—"}</span>'
            st.markdown(
                f'<span style="font-size:0.9rem;color:#666">M/M-1: {mm_txt} · A/A-1: {yy_txt}</span>',
                unsafe_allow_html=True
            )

            # Mini-plot (últimos 18 meses)
            series = df[(df["codigo_entidad"] == ent_code) & (df["codigo_indicador"] == r["Código"])][["fecha_dt", "valor"]].copy()
            series["serie"] = display_name(ent_code, show_full_names)
            plot_df = series.rename(columns={"valor": "value"}).copy()

            # tratar ceros por serie
            plot_df = treat_zeros_long(plot_df, "value", ["serie"], zero_mode)

            plot_df["value_plot"] = to_plot_series(plot_df["value"], r["Formato"])
            cutoff = month - pd.DateOffset(months=18)
            plot_df = plot_df[plot_df["fecha_dt"] >= cutoff]

            if not plot_df.empty:
                legend = alt.Legend(orient="bottom", direction="horizontal", title=None, labelLimit=160)
                tvals, tfmt = tick_values_for_range(plot_df["fecha_dt"].min(), plot_df["fecha_dt"].max())
                # construir axis Y sólo si corresponde (evita pasar format=None)
                fmt_axis = ".1%" if str(r["Formato"]).upper() == "P" else None
                axis_kwargs = {"title": ""}
                if isinstance(fmt_axis, str) and fmt_axis:
                    axis_kwargs["format"] = fmt_axis
                axis_y = alt.Axis(**axis_kwargs)

                base = alt.Chart(plot_df).mark_line().encode(
                    x=alt.X("fecha_dt:T", title="", axis=alt.Axis(values=tvals, format=tfmt, tickCount=len(tvals))),
                    y=alt.Y("value_plot:Q", title="", axis=axis_y),
                    color=alt.Color("serie:N", legend=legend, scale=alt.Scale(scheme="tableau10"))
                ).properties(height=140)
                st.altair_chart(base, use_container_width=True)
            else:
                st.caption("Sin datos recientes para esta variable.")

# ===================== SERIE =====================
with tab_serie:
    st.subheader("Serie")
    c1, c2 = st.columns([1, 2])
    with c1:
        ent_options = [str(c) for c in df["codigo_entidad"].astype("string").tolist() if pd.notna(c) and str(c) != "<NA>" and str(c)]
        ents = st.multiselect(
            "Entidades",
            options=sorted(set(ent_options), key=lambda c: display_name(c, show_full_names)),
            default=[DEFAULT_ENTITY_CODE] if DEFAULT_ENTITY_CODE in ent_options else [],
            format_func=lambda c: display_name(c, show_full_names),
            key="serie_ents"
        )
        vars_series = st.multiselect(
            "Variables",
            options=sorted([v for v in df["codigo_indicador"].astype("string").dropna().unique().tolist() if v not in HIDE_VARS],
                           key=lambda v: var_map.get(v, v)),
            default=find_variable_codes_by_hint(DEFAULT_VAR_HINTS, top_k=2),
            format_func=lambda v: var_map.get(v, v),
            key="serie_vars"
        )
        # Selector de rango temporal
        min_dt = df["fecha_dt"].min().date()
        max_dt = df["fecha_dt"].max().date()
        default_start = date(max_dt.year - 5, max_dt.month, 1) if (max_dt.year - min_dt.year) >= 5 else min_dt
        date_range = st.slider(
            "Rango temporal",
            min_value=min_dt,
            max_value=max_dt,
            value=(default_start, max_dt),
            key="serie_range"
        )

    with c2:
        if not ents or not vars_series:
            st.info("Seleccioná al menos una entidad y una variable.")
        else:
            start_dt = pd.to_datetime(date_range[0])
            end_dt = pd.to_datetime(date_range[1])

            plot_frames = []
            for e in ents:
                sub = df[(df["codigo_entidad"] == e) & (df["codigo_indicador"].isin(vars_series))].copy()
                sub = sub[(sub["fecha_dt"] >= start_dt) & (sub["fecha_dt"] <= end_dt)]
                if sub.empty:
                    continue
                sub["Entidad"] = display_name(e, show_full_names)
                sub["Variable"] = sub["codigo_indicador"].map(lambda c: var_map.get(c, c))
                sub["Formato"] = sub["codigo_indicador"].map(lambda c: fmt_map.get(c, "N"))
                sub["Formato"] = sub["Formato"].astype("string").str.upper().fillna("N")

                # tratar ceros por serie (Entidad, Variable)
                sub = treat_zeros_long(sub, "valor", ["Entidad", "Variable"], zero_mode)

                sub["ValorPlot"] = np.where(sub["Formato"] == "P", sub["valor"] / 100.0, sub["valor"])
                plot_frames.append(sub)

            if plot_frames:
                final_plot = pd.concat(plot_frames, ignore_index=True)
                tvals, tfmt = tick_values_for_range(final_plot["fecha_dt"].min(), final_plot["fecha_dt"].max())

                base = (
                    alt.Chart(final_plot)
                    .mark_line(point=True)
                    .encode(
                        x=alt.X("fecha_dt:T", title="Mes", axis=alt.Axis(values=tvals, format=tfmt, labelOverlap=True, tickCount=len(tvals))),
                        y=alt.Y("ValorPlot:Q", title="Valor"),
                        color=alt.Color("Entidad:N", title="Serie", legend=alt.Legend(orient="bottom"))
                    )
                ).properties(height=240)

                chart = base.facet(
                    facet=alt.Facet("Variable:N", title=None),
                    columns=1
                ).resolve_scale(
                    y="independent"
                )

                st.altair_chart(chart, use_container_width=True)
            else:
                st.info("No hay datos para graficar en el rango seleccionado.")

# ===================== CALCULADORA =====================
with tab_calc:
    st.subheader("Calculadora de variables")
    c1, c2 = st.columns([1, 2])
    with c1:
        ent_options2 = [str(c) for c in df["codigo_entidad"].astype("string").tolist() if pd.notna(c) and str(c) != "<NA>" and str(c)]
        ents2 = st.multiselect(
            "Entidades",
            options=sorted(set(ent_options2), key=lambda c: display_name(c, show_full_names)),
            default=[DEFAULT_ENTITY_CODE] if DEFAULT_ENTITY_CODE in ent_options2 else [],
            format_func=lambda c: display_name(c, show_full_names),
            key="calc_ents"
        )
        st.caption("Construí una fórmula: Termo1 (op) Termo2 (op) Termo3 ...")

        ops = ["+", "-", "*", "/"]
        term_vars = []
        term_ops = []
        for i in range(1, 6):
            v = st.selectbox(
                f"Variable {i}",
                options=["—"] + sorted([vv for vv in df["codigo_indicador"].astype("string").dropna().unique().tolist() if vv not in HIDE_VARS],
                                       key=lambda x: var_map.get(x, x)),
                index=0,
                format_func=lambda x: var_map.get(x, x) if x != "—" else "—",
                key=f"calc_var_{i}"
            )
            term_vars.append(None if v == "—" else v)
            if i < 5:
                op = st.selectbox(f"Operación {i}→{i+1}", options=["—"] + ops, index=0, key=f"calc_op_{i}")
                term_ops.append(None if op == "—" else op)

        # Rango de fechas
        min_dt = df["fecha_dt"].min().date()
        max_dt = df["fecha_dt"].max().date()
        range_calc = st.slider(
            "Rango temporal",
            min_value=min_dt, max_value=max_dt, value=(min_dt, max_dt),
            key="calc_range"
        )

    with c2:
        active_terms = [tv for tv in term_vars if tv]
        if not ents2 or len(active_terms) < 1:
            st.info("Elegí al menos 1 variable y una entidad.")
        else:
            vars_sel = [v for v in term_vars if v]
            sub = df[df["codigo_indicador"].isin(vars_sel)].copy()
            sub = sub[(sub["fecha_dt"] >= pd.to_datetime(range_calc[0])) & (sub["fecha_dt"] <= pd.to_datetime(range_calc[1]))]
            if sub.empty:
                st.info("No hay datos en ese rango.")
            else:
                # pivot ancho por entidad-fecha
                pivot = sub.pivot_table(index=["codigo_entidad", "fecha_dt"], columns="codigo_indicador", values="valor", aggfunc="last").reset_index()
                pivot = pivot[pivot["codigo_entidad"].isin(ents2)].copy()
                pivot = pivot.sort_values(["codigo_entidad", "fecha_dt"])

                # tratamiento de ceros en inputs
                if zero_mode.startswith("Ignorar"):
                    for v in vars_sel:
                        if v in pivot.columns:
                            pivot.loc[pivot[v] == 0, v] = np.nan
                elif zero_mode.startswith("Reemplazar"):
                    for v in vars_sel:
                        if v in pivot.columns:
                            pivot.loc[pivot[v] == 0, v] = np.nan
                            pivot[v] = pivot.groupby("codigo_entidad", group_keys=False)[v].ffill()

                def apply_expr(row):
                    nums = []
                    for v in term_vars:
                        if v:
                            nums.append(row.get(v, np.nan))
                    if not nums:
                        return np.nan
                    res = nums[0]
                    idx_op = 0
                    for j in range(1, len(nums)):
                        op = term_ops[idx_op] if idx_op < len(term_ops) else None
                        val = nums[j]
                        idx_op += 1
                        if op is None or pd.isna(res) or pd.isna(val):
                            continue
                        try:
                            if op == "+": res = res + val
                            elif op == "-": res = res - val
                            elif op == "*": res = res * val
                            elif op == "/": res = np.nan if val == 0 else res / val
                        except Exception:
                            res = np.nan
                    return res

                pivot["Resultado"] = pivot.apply(apply_expr, axis=1)
                pivot["Entidad"] = pivot["codigo_entidad"].apply(lambda c: display_name(c, show_full_names))

                # ticks legibles sin agrupar datos
                tvals, tfmt = tick_values_for_range(pivot["fecha_dt"].min(), pivot["fecha_dt"].max())

                chart = (
                    alt.Chart(pivot.dropna(subset=["Resultado"]))
                    .mark_line(point=True)
                    .encode(
                        x=alt.X("fecha_dt:T", title="Mes", axis=alt.Axis(values=tvals, format=tfmt, tickCount=len(tvals))),
                        y=alt.Y("Resultado:Q", title="Resultado"),
                        color=alt.Color("Entidad:N", title="Entidad", legend=alt.Legend(orient="bottom"))
                    )
                    .properties(height=420)
                )
                st.altair_chart(chart, use_container_width=True)

                with st.expander("Ver datos"):
                    st.dataframe(pivot[["Entidad", "fecha_dt", "Resultado"]].sort_values(["Entidad", "fecha_dt"]), use_container_width=True)

# ===================== RANKINGS =====================
with tab_rank:
    st.subheader("Rankings")

    c1, c2 = st.columns([1, 2])
    with c1:
        var_rank = st.selectbox(
            "Variable",
            options=sorted([v for v in df["codigo_indicador"].astype("string").dropna().unique().tolist() if v not in HIDE_VARS],
                           key=lambda v: var_map.get(v, v)),
            index=0,
            format_func=lambda v: var_map.get(v, v),
            key="rank_var",
        )
        min_dt = df["fecha_dt"].min().date()
        max_dt = df["fecha_dt"].max().date()
        default_start = date(max_dt.year - 1, max_dt.month, 1) if (max_dt.year - min_dt.year) >= 1 else min_dt
        range_rank = st.slider(
            "Rango temporal",
            min_value=min_dt, max_value=max_dt,
            value=(default_start, max_dt),
            key="rank_range"
        )
        topn = st.number_input("Top N", min_value=3, max_value=50, value=10, step=1, key="rank_topn")

        fmt_rank = str(fmt_map.get(var_rank, "N")).upper()
        st.caption("📌 En “Top valores totales” se usa **promedio** del período, omitiendo 0.00.")

    with c2:
        # Excluir agrupadas (AA*) en rankings
        df_rank = df[~df["codigo_entidad"].astype("string").str.startswith("AA", na=False)].copy()
        sub = df_rank[df_rank["codigo_indicador"] == var_rank].copy()
        sub = sub[
            (sub["fecha_dt"] >= pd.to_datetime(range_rank[0])) &
            (sub["fecha_dt"] <= pd.to_datetime(range_rank[1]))
        ]

        if sub.empty:
            st.info("No hay datos para ese rango/variable.")
        else:
            # Tratamiento de ceros para variaciones
            sub = treat_zeros_long(sub, "valor", ["codigo_entidad"], zero_mode)

            # Top valores totales (promedio sin 0.00)
            sub_nz = sub[sub["valor"] != 0].copy()
            if sub_nz.empty:
                st.info("No hay datos (no nulos) para calcular promedios en este rango.")
            else:
                totals = (
                    sub_nz.groupby("codigo_entidad", as_index=False)["valor"]
                    .mean()
                    .rename(columns={"valor": "metric"})
                )
                metric_label = "Promedio en período (sin 0.00)"
                totals["Entidad"] = totals["codigo_entidad"].apply(lambda c: display_name(c, show_full_names))

                if fmt_rank == "P":
                    totals["metric_plot"] = totals["metric"] / 100.0
                    x_axis_tot = alt.Axis(title=metric_label, format=".1%")
                    text_fmt_tot = ".1%"
                else:
                    totals["metric_plot"] = totals["metric"]
                    x_axis_tot = alt.Axis(title=metric_label, format=",.2f")
                    text_fmt_tot = ",.2f"

                top_tot = totals.sort_values("metric_plot", ascending=False).head(int(topn))

                st.markdown("### 🏆 Top valores totales (promedio)")
                if top_tot.empty:
                    st.info("No hay datos suficientes para calcular promedios.")
                else:
                    chart_tot = (
                        alt.Chart(top_tot)
                        .mark_bar()
                        .encode(
                            x=alt.X("metric_plot:Q", axis=x_axis_tot),
                            y=alt.Y("Entidad:N", sort="-x", title=None),
                            tooltip=[
                                alt.Tooltip("Entidad:N"),
                                alt.Tooltip("metric:Q", title=metric_label, format=text_fmt_tot),
                            ],
                        )
                        .properties(height=28 * len(top_tot) + 20)
                    )
                    text_tot = chart_tot.mark_text(align="left", dx=3).encode(
                        text=alt.Text("metric_plot:Q", format=text_fmt_tot)
                    )
                    st.altair_chart(chart_tot + text_tot, use_container_width=True)

            # Top variación % (último vs primero)
            s = sub.sort_values(["codigo_entidad", "fecha_dt"])
            first = s.groupby("codigo_entidad", as_index=False).first()[["codigo_entidad", "valor"]].rename(columns={"valor": "first"})
            last = s.groupby("codigo_entidad", as_index=False).last()[["codigo_entidad", "valor"]].rename(columns={"valor": "last"})
            ch = pd.merge(first, last, on="codigo_entidad", how="inner")
            ch = ch[(~ch["first"].isna()) & (~ch["last"].isna()) & (ch["first"] != 0)]
            ch["delta"] = ch["last"] / ch["first"] - 1.0
            ch["Entidad"] = ch["codigo_entidad"].apply(lambda c: display_name(c, show_full_names))

            col_up, col_dn = st.columns(2)
            with col_up:
                st.markdown("### 📈 Mayores subas (% cambio)")
                top_up = ch.sort_values("delta", ascending=False).head(int(topn))
                if top_up.empty:
                    st.info("No hay suficientes datos para calcular subas.")
                else:
                    chart_up = (
                        alt.Chart(top_up)
                        .mark_bar()
                        .encode(
                            x=alt.X("delta:Q", axis=alt.Axis(title="% cambio", format=".1%")),
                            y=alt.Y("Entidad:N", sort="-x", title=None),
                            tooltip=[
                                alt.Tooltip("Entidad:N"),
                                alt.Tooltip("delta:Q", title="% cambio", format=".1%"),
                            ],
                        )
                        .properties(height=28 * len(top_up) + 20)
                    )
                    text_up = chart_up.mark_text(align="left", dx=3).encode(text=alt.Text("delta:Q", format=".1%"))
                    st.altair_chart(chart_up + text_up, use_container_width=True)

            with col_dn:
                st.markdown("### 📉 Mayores bajas (% cambio)")
                top_dn = ch.sort_values("delta", ascending=True).head(int(topn))
                if top_dn.empty:
                    st.info("No hay suficientes datos para calcular bajas.")
                else:
                    chart_dn = (
                        alt.Chart(top_dn)
                        .mark_bar()
                        .encode(
                            x=alt.X("delta:Q", axis=alt.Axis(title="% cambio", format=".1%")),
                            y=alt.Y("Entidad:N", sort=None, title=None),
                            tooltip=[
                                alt.Tooltip("Entidad:N"),
                                alt.Tooltip("delta:Q", title="% cambio", format=".1%"),
                            ],
                        )
                        .properties(height=28 * len(top_dn) + 20)
                    )
                    text_dn = chart_dn.mark_text(align="left", dx=3).encode(text=alt.Text("delta:Q", format=".1%"))
                    st.altair_chart(chart_dn + text_dn, use_container_width=True)

# ===================== SISTEMA FINANCIERO =====================
with tab_sys:
    st.subheader("Sistema financiero")

    # -------- A) Evolución de cantidades (11 indicadores) --------
    st.markdown("#### Evolución de cantidades (desde 2015)")
    sys_codes = list(HIDE_VARS)  # los 11 códigos
    sys_df = df[df["codigo_indicador"].isin(sys_codes)].copy()

    if sys_df.empty:
        st.info("No se encontraron series para los indicadores de sistema. Verificá que existan en los CSV.")
    else:
        # Un valor por fecha e indicador
        sys_df = sys_df.sort_values(["fecha_dt"])
        sys_df = sys_df.groupby(["fecha_dt", "codigo_indicador"], as_index=False).agg(
            indicador=("indicador", "last"),
            formato=("formato", "last"),
            valor=("valor", "last"),
        )
        sys_df = treat_zeros_long(sys_df, "valor", [], zero_mode)
        sys_df["ValorPlot"] = to_plot_series(sys_df["valor"], "N")
        tvals, tfmt = tick_values_for_range(sys_df["fecha_dt"].min(), sys_df["fecha_dt"].max())

        chart_sys = (
            alt.Chart(sys_df)
            .mark_line(point=True)
            .encode(
                x=alt.X("fecha_dt:T", title="Mes", axis=alt.Axis(values=tvals, format=tfmt, tickCount=len(tvals))),
                y=alt.Y("ValorPlot:Q", title="Cantidad"),
                color=alt.Color("indicador:N", title="Indicador", legend=alt.Legend(orient="bottom", columns=2)),
                tooltip=[alt.Tooltip("indicador:N"), alt.Tooltip("fecha_dt:T"), alt.Tooltip("valor:Q", format=",.0f")]
            )
            .properties(height=360)
        )
        st.altair_chart(chart_sys, use_container_width=True)

    st.divider()

    # -------- B) Comparador por agrupaciones AA* para una variable (no-sistema) --------
    st.markdown("#### Comparador por agrupaciones (AA*)")
    colL, colR = st.columns([1,2])
    with colL:
        allowed_vars = sorted([v for v in df["codigo_indicador"].astype("string").dropna().unique().tolist() if v not in HIDE_VARS],
                              key=lambda v: var_map.get(v, v))
        var_sel_sys = st.selectbox(
            "Variable",
            options=allowed_vars,
            format_func=lambda v: var_map.get(v, v),
            key="sys_var"
        )
        aa_opts = list(AA_GROUPS.keys())
        aa_pick = st.multiselect(
            "Agrupaciones",
            options=aa_opts,
            default=aa_opts,
            format_func=lambda c: display_name(c, show_full_names),
            key="sys_aa"
        )
        min_dt = df["fecha_dt"].min().date()
        max_dt = df["fecha_dt"].max().date()
        range_sys = st.slider(
            "Rango temporal",
            min_value=min_dt, max_value=max_dt, value=(min_dt, max_dt),
            key="sys_range"
        )

    with colR:
        if not aa_pick:
            st.info("Elegí al menos una agrupación.")
        else:
            start_dt = pd.to_datetime(range_sys[0])
            end_dt = pd.to_datetime(range_sys[1])
            sub = df[(df["codigo_entidad"].isin(aa_pick)) & (df["codigo_indicador"] == var_sel_sys)].copy()
            sub = sub[(sub["fecha_dt"] >= start_dt) & (sub["fecha_dt"] <= end_dt)]

            if sub.empty:
                st.info("No hay datos para esa variable/agrupaciones en el rango.")
            else:
                sub["Entidad"] = sub["codigo_entidad"].apply(lambda c: display_name(c, show_full_names))
                sub["Variable"] = var_map.get(var_sel_sys, var_sel_sys)

                # ---------- robust patch: asegurar formato escalar, numericidad y axis válido ----------
                fmt_var = str(fmt_map.get(var_sel_sys, "N") or "N").upper()
                sub["Formato"] = fmt_var

                # asegurar que 'valor' sea numérico y no tenga valores raros
                sub["valor"] = pd.to_numeric(sub["valor"], errors="coerce")

                # tratar ceros por agrupación
                sub = treat_zeros_long(sub, "valor", ["Entidad"], zero_mode)

                # calcular ValorPlot según formato escalar
                if fmt_var == "P":
                    sub["ValorPlot"] = sub["valor"] / 100.0
                else:
                    sub["ValorPlot"] = sub["valor"]

                # limpiar filas inválidas antes de graficar
                sub = sub.dropna(subset=["fecha_dt", "ValorPlot"]).copy()
                if sub.empty:
                    st.info("No hay datos válidos para graficar después del tratamiento (fechas/valores faltantes).")
                else:
                    # preparar ticks
                    tvals, tfmt = tick_values_for_range(sub["fecha_dt"].min(), sub["fecha_dt"].max())

                    # construir axis kwargs SOLO si el formato es válido (cadena)
                    fmt_axis = ".1%" if fmt_var == "P" else None
                    axis_kwargs = {"title": "Valor"}
                    if isinstance(fmt_axis, str) and fmt_axis:
                        axis_kwargs["format"] = fmt_axis
                    # crear el objeto Axis (Altair validará aquí)
                    y_axis = alt.Axis(**axis_kwargs)

                    # intentar graficar y, en caso de error, mostrar info útil de depuración
                    try:
                        chart_cmp = (
                            alt.Chart(sub)
                            .mark_line(point=True)
                            .encode(
                                x=alt.X("fecha_dt:T", title="Mes", axis=alt.Axis(values=tvals, format=tfmt, tickCount=len(tvals))),
                                y=alt.Y("ValorPlot:Q", axis=y_axis),
                                color=alt.Color("Entidad:N", title="Agrupación", legend=alt.Legend(orient="bottom")),
                                tooltip=[alt.Tooltip("Entidad:N"), alt.Tooltip("fecha_dt:T", title="Fecha"), alt.Tooltip("valor:Q", title="Valor")]
                            )
                            .properties(height=360)
                        )
                        st.altair_chart(chart_cmp, use_container_width=True)

                    except Exception as e:
                        # mostrar el error y datos para diagnosticar
                        st.error(f"Error al renderizar gráfico: {type(e).__name__}: {e}")
                        st.markdown("**Depuración rápida:**")
                        st.write("fmt_var (valor y tipo):", repr(fmt_var), " — tipo:", type(fmt_var))
                        st.write("Primeras filas de `sub` (fecha_dt, valor, ValorPlot, Formato):")
                        st.dataframe(sub[["fecha_dt", "valor", "ValorPlot", "Formato"]].head(20))
                        st.write("Valores únicos en 'Formato':", sub["Formato"].unique().tolist())
                        st.info("Si ves formatos raros o strings en 'valor', revisá bcra_indicadores.csv (columna 'formato') o la limpieza de 'valor' en el consolidado.")

# Footer
st.caption("Fuente: TXT BCRA (Entfin/Tec_Cont). ALIAS habilitados; nombres largos opcionales. v8.1.2")
