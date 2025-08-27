# streamlit_app.py (v6)
# Requiere: streamlit pandas altair numpy
# CSVs esperados en ./data:
#   - bcra_consolidado.csv (o bcra_consolidado_part*.csv)
#   - bcra_nomina.csv
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

st.set_page_config(page_title="Indicadores BCRA", layout="wide")

DATA_DIR = Path("./data")
MASTER = DATA_DIR / "bcra_consolidado.csv"
PARTS = sorted([Path(p) for p in glob.glob(str(DATA_DIR / "bcra_consolidado_part*.csv"))])
NOMINA = DATA_DIR / "bcra_nomina.csv"
INDICES = DATA_DIR / "bcra_indicadores.csv"
AGREGADOS = DATA_DIR / "bcra_agregados.csv"  # opcional

DEFAULT_ENTITY_CODE = "00011"  # Banco Nación
DEFAULT_MONTH_STR = "2025-05"
DEFAULT_VAR_HINTS = ["Dotación de personal", "ROE", "ROA", "Gastos en personal"]

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
        ["fecha_dt","codigo_entidad","codigo_indicador"]
    ).reset_index(drop=True)
    return df_all

@st.cache_data(show_spinner=False)
def load_nomina_map():
    dtypes = {"codigo_entidad": "string", "entidad": "string"}
    df = _read_csv_safe(NOMINA, dtype=dtypes)
    if df is None or df.empty:
        return {}
    return dict(zip(df["codigo_entidad"], df["entidad"]))

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
    df = df.dropna(subset=["fecha_dt"]).sort_values(["fecha_dt","codigo_indicador"]).reset_index(drop=True)
    if "formato" in df.columns:
        df["formato"] = df["formato"].astype("string").str.upper().fillna("N")
    return df

df = load_consolidado()
ent_map = load_nomina_map()
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
    return series_vals/100.0 if str(fmt_code).upper()=="P" else series_vals

def find_variable_codes_by_hint(hints, top_k=4):
    codes = []
    if var_map:
        for hint in hints:
            hint_low = hint.lower()
            matches = [c for c, d in var_map.items() if hint_low in str(d).lower()]
            for m in matches:
                if m not in codes:
                    codes.append(m)
            if len(codes) >= top_k:
                break
    if len(codes) < top_k and not df.empty:
        for c in df["codigo_indicador"].value_counts().index.tolist():
            if c not in codes:
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
    span_months = (end.year - start.year)*12 + (end.month - start.month) + 1

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
        idx = np.linspace(0, len(ticks)-1, 10).round().astype(int)
        ticks = [ticks[i] for i in idx]
    return ticks, fmt

def treat_zeros_long(df_long: pd.DataFrame, value_col: str, group_cols: list, mode: str):
    """Aplica el tratamiento de ceros a un DF largo (de series).
       - 'Mantener': no cambia nada
       - 'Ignorar (saltear meses)': elimina filas con valor==0
       - 'Reemplazar por dato previo (ffill)': 0 -> NaN y ffill por grupo
    """
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

# ---------- UI ----------
st.title("📊 Indicadores del BCRA (v6)")

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

# Tabs (sin “Porcentaje del total”)
tab_panel, tab_serie, tab_calc, tab_rank = st.tabs(["Panel", "Serie", "Calculadora", "Rankings"])

# ===================== PANEL =====================
with tab_panel:
    st.subheader("Panel")
    colA, colB, colC = st.columns([1,1,1])
    with colA:
        ent_codes_sorted = sorted(df["codigo_entidad"].dropna().unique().tolist(), key=lambda c: ent_map.get(c, c))
        default_ent_idx = ent_codes_sorted.index(DEFAULT_ENTITY_CODE) if DEFAULT_ENTITY_CODE in ent_codes_sorted else 0
        ent_code = st.selectbox(
            "Entidad", options=ent_codes_sorted, index=default_ent_idx,
            format_func=lambda code: ent_map.get(code, code),
            key="panel_ent"
        )
    with colB:
        default_var_codes = find_variable_codes_by_hint(DEFAULT_VAR_HINTS, top_k=4)
        var_codes_sorted = sorted(df["codigo_indicador"].dropna().unique().tolist(), key=lambda v: var_map.get(v, v))
        var_sel = st.multiselect(
            "Variables", options=var_codes_sorted,
            default=[c for c in default_var_codes if c in var_codes_sorted][:4],
            format_func=lambda code: var_map.get(code, code),
            key="panel_vars"
        )
    with colC:
        all_months = sorted(df["fecha_dt"].unique().tolist())
        def_month = default_month_value()
        month = st.selectbox(
            "Mes", options=all_months,
            index=(all_months.index(def_month) if def_month in all_months else len(all_months)-1),
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
            # color en el delta
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

            # Mini-plot (últimos 18 meses) con leyenda abajo y tratamiento de ceros
            series = df[(df["codigo_entidad"] == ent_code) & (df["codigo_indicador"] == r["Código"])][["fecha_dt","valor"]].copy()
            series["serie"] = ent_map.get(ent_code, ent_code)
            plot_frames = [series.rename(columns={"valor":"value"})]
            if agg_df is not None:
                ag = agg_df[agg_df["codigo_indicador"] == r["Código"]].copy()
                cmps = {
                    "Sistema financiero": "sistema_financiero",
                    "Banca pública": "banca_publica",
                    "Banca privada": "banca_privada",
                    "Capital nacional": "banca_nacional",
                    "Capital extranjero": "banca_extranjera",
                    "Compañías financieras": "companias_financieras",
                }
                for label, colname in cmps.items():
                    if colname in ag.columns:
                        tmp = ag[["fecha_dt", colname]].rename(columns={colname:"value"})
                        tmp["serie"] = label
                        plot_frames.append(tmp[["fecha_dt","value","serie"]])
            plot_df = pd.concat(plot_frames, ignore_index=True) if len(plot_frames)>1 else plot_frames[0]

            # tratar ceros por serie
            plot_df = treat_zeros_long(plot_df, "value", ["serie"], zero_mode)

            # preparar valores y recorte temporal
            plot_df["value_plot"] = to_plot_series(plot_df["value"], r["Formato"])
            cutoff = month - pd.DateOffset(months=18)
            plot_df = plot_df[plot_df["fecha_dt"] >= cutoff]

            legend = alt.Legend(orient="bottom", direction="horizontal", title=None, labelLimit=160)
            tvals, tfmt = tick_values_for_range(plot_df["fecha_dt"].min(), plot_df["fecha_dt"].max())
            axis_y = alt.Axis(format=(".1%" if r["Formato"]=="P" else None))
            base = alt.Chart(plot_df).mark_line().encode(
                x=alt.X("fecha_dt:T", title="", axis=alt.Axis(values=tvals, format=tfmt, tickCount=len(tvals))),
                y=alt.Y("value_plot:Q", title="", axis=axis_y),
                color=alt.Color("serie:N", legend=legend, scale=alt.Scale(scheme="tableau10"))
            ).properties(height=140)
            st.altair_chart(base, use_container_width=True)

    if agg_df is None:
        st.info("Para comparadores (Sistema, Pública/Privada, etc.), agregá `data/bcra_agregados.csv`.")

# ===================== SERIE =====================
with tab_serie:
    st.subheader("Serie")
    c1, c2 = st.columns([1,2])
    with c1:
        ents = st.multiselect(
            "Entidades",
            options=sorted(df["codigo_entidad"].unique().tolist(), key=lambda c: ent_map.get(c, c)),
            default=[DEFAULT_ENTITY_CODE] if DEFAULT_ENTITY_CODE in df["codigo_entidad"].unique() else [],
            format_func=lambda c: ent_map.get(c, c),
            key="serie_ents"
        )
        vars_series = st.multiselect(
            "Variables",
            options=sorted(df["codigo_indicador"].unique().tolist(), key=lambda v: var_map.get(v, v)),
            default=find_variable_codes_by_hint(DEFAULT_VAR_HINTS, top_k=2),
            format_func=lambda v: var_map.get(v, v),
            key="serie_vars"
        )
        show_agg = st.checkbox("Mostrar comparadores (si hay agregados)", value=True, key="serie_show_agg")

        # Selector de rango temporal
        min_dt = df["fecha_dt"].min().date()
        max_dt = df["fecha_dt"].max().date()
        default_start = date(max_dt.year-5, max_dt.month, 1) if (max_dt.year - min_dt.year) >= 5 else min_dt
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
                sub = df[(df["codigo_entidad"]==e) & (df["codigo_indicador"].isin(vars_series))].copy()
                sub = sub[(sub["fecha_dt"]>=start_dt) & (sub["fecha_dt"]<=end_dt)]
                if sub.empty:
                    continue
                sub["Entidad"] = ent_map.get(e, e)
                sub["Variable"] = sub["codigo_indicador"].map(lambda c: var_map.get(c, c))
                sub["Formato"] = sub["codigo_indicador"].map(lambda c: fmt_map.get(c, "N"))
                sub["Formato"] = sub["Formato"].astype("string").str.upper().fillna("N")

                # tratar ceros por serie (Entidad, Variable)
                sub = treat_zeros_long(sub, "valor", ["Entidad","Variable"], zero_mode)

                sub["ValorPlot"] = np.where(sub["Formato"]=="P", sub["valor"]/100.0, sub["valor"])
                plot_frames.append(sub)

            if show_agg and agg_df is not None:
                ag = agg_df[agg_df["codigo_indicador"].isin(vars_series)].copy()
                ag = ag[(ag["fecha_dt"]>=start_dt) & (ag["fecha_dt"]<=end_dt)]
                long_list = []
                for name, col in {
                    "Sistema financiero":"sistema_financiero",
                    "Banca pública":"banca_publica",
                    "Banca privada":"banca_privada",
                    "Capital nacional":"banca_nacional",
                    "Capital extranjero":"banca_extranjera",
                    "Compañías financieras":"companias_financieras",
                }.items():
                    if col in ag.columns:
                        t = ag[["fecha_dt","codigo_indicador","indicador","formato",col]].rename(columns={col:"valor"})
                        t["Entidad"] = name
                        t["Variable"] = t["codigo_indicador"].map(lambda c: var_map.get(c, c))
                        t["Formato"] = t["codigo_indicador"].map(lambda c: fmt_map.get(c, "N"))
                        t["Formato"] = t["Formato"].astype("string").str.upper().fillna("N")

                        # tratar ceros por serie agregada (Entidad, Variable)
                        t = treat_zeros_long(t, "valor", ["Entidad","Variable"], zero_mode)

                        t["ValorPlot"] = np.where(t["Formato"]=="P", t["valor"]/100.0, t["valor"])
                        long_list.append(t)
                if long_list:
                    plot_frames.append(pd.concat(long_list, ignore_index=True))

            if plot_frames:
                final_plot = pd.concat(plot_frames, ignore_index=True)

                # ticks legibles sin agrupar datos
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
    c1, c2 = st.columns([1,2])
    with c1:
        ents2 = st.multiselect(
            "Entidades",
            options=sorted(df["codigo_entidad"].unique().tolist(), key=lambda c: ent_map.get(c, c)),
            default=[DEFAULT_ENTITY_CODE] if DEFAULT_ENTITY_CODE in df["codigo_entidad"].unique() else [],
            format_func=lambda c: ent_map.get(c, c),
            key="calc_ents"
        )
        st.caption("Construí una fórmula: Termo1 (op) Termo2 (op) Termo3 ...")

        ops = ["+", "-", "*", "/"]
        term_vars = []
        term_ops = []
        for i in range(1,6):
            v = st.selectbox(
                f"Variable {i}",
                options=["—"] + sorted(df["codigo_indicador"].unique().tolist(), key=lambda x: var_map.get(x, x)),
                index=0,
                format_func=lambda x: var_map.get(x, x) if x!="—" else "—",
                key=f"calc_var_{i}"
            )
            term_vars.append(None if v=="—" else v)
            if i < 5:
                op = st.selectbox(f"Operación {i}→{i+1}", options=["—"]+ops, index=0, key=f"calc_op_{i}")
                term_ops.append(None if op=="—" else op)

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
            sub = sub[(sub["fecha_dt"]>=pd.to_datetime(range_calc[0])) & (sub["fecha_dt"]<=pd.to_datetime(range_calc[1]))]
            if sub.empty:
                st.info("No hay datos en ese rango.")
            else:
                # pivot ancho por entidad-fecha
                pivot = sub.pivot_table(index=["codigo_entidad","fecha_dt"], columns="codigo_indicador", values="valor", aggfunc="last").reset_index()
                pivot = pivot[pivot["codigo_entidad"].isin(ents2)].copy()
                pivot = pivot.sort_values(["codigo_entidad","fecha_dt"])

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
                pivot["Entidad"] = pivot["codigo_entidad"].map(lambda c: ent_map.get(c, c))

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
                    st.dataframe(pivot[["Entidad","fecha_dt","Resultado"]].sort_values(["Entidad","fecha_dt"]), use_container_width=True)


# ===================== RANKINGS =====================
with tab_rank:
    st.subheader("Rankings")

    c1, c2 = st.columns([1, 2])
    with c1:
        # Variable a rankear
        var_rank = st.selectbox(
            "Variable",
            options=sorted(df["codigo_indicador"].unique().tolist(), key=lambda v: var_map.get(v, v)),
            index=0,
            format_func=lambda v: var_map.get(v, v),
            key="rank_var",
        )

        # Rango temporal
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
        if fmt_rank == "P":
            st.caption("📌 La variable es porcentual: en “Valores totales” se usa **promedio** del período (no suma).")

    with c2:
        # Filtrar: excluir entidades agrupadas (codigos que empiezan con 'AA')
        df_rank = df[~df["codigo_entidad"].str.startswith("AA", na=False)].copy()

        sub = df_rank[df_rank["codigo_indicador"] == var_rank].copy()
        sub = sub[
            (sub["fecha_dt"] >= pd.to_datetime(range_rank[0])) &
            (sub["fecha_dt"] <= pd.to_datetime(range_rank[1]))
        ]

        if sub.empty:
            st.info("No hay datos para ese rango/variable.")
        else:
            # Tratamiento de ceros según selector global
            sub = treat_zeros_long(sub, "valor", ["codigo_entidad"], zero_mode)

            # ---------- Top valores totales ----------
            if fmt_rank == "P":
                totals = sub.groupby("codigo_entidad", as_index=False)["valor"].mean().rename(columns={"valor": "metric"})
                metric_label = "Promedio en período"
                totals["metric_plot"] = totals["metric"] / 100.0
                x_axis_tot = alt.Axis(title=metric_label, format=".1%")
                text_fmt_tot = ".1%"
            else:
                totals = sub.groupby("codigo_entidad", as_index=False)["valor"].sum().rename(columns={"valor": "metric"})
                metric_label = "Suma en período"
                totals["metric_plot"] = totals["metric"]
                x_axis_tot = alt.Axis(title=metric_label, format=",.2f")
                text_fmt_tot = ",.2f"

            totals["Entidad"] = totals["codigo_entidad"].map(lambda c: ent_map.get(c, c))
            top_tot = totals.sort_values("metric_plot", ascending=False).head(int(topn))

            st.markdown("### 🏆 Top valores totales")
            if top_tot.empty:
                st.info("No hay datos suficientes para calcular totales.")
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
                text_tot = chart_tot.mark_text(align="left", dx=3).encode(text=alt.Text("metric_plot:Q", format=text_fmt_tot))
                st.altair_chart(chart_tot + text_tot, use_container_width=True)

            # ---------- Top variación % (último vs primero del rango) ----------
            s = sub.sort_values(["codigo_entidad", "fecha_dt"])
            first = s.groupby("codigo_entidad", as_index=False).first()[["codigo_entidad", "valor"]].rename(columns={"valor": "first"})
            last = s.groupby("codigo_entidad", as_index=False).last()[["codigo_entidad", "valor"]].rename(columns={"valor": "last"})
            ch = pd.merge(first, last, on="codigo_entidad", how="inner")
            # filtrar entidades sin base válida
            ch = ch[(~ch["first"].isna()) & (~ch["last"].isna()) & (ch["first"] != 0)]
            ch["delta"] = ch["last"] / ch["first"] - 1.0
            ch["Entidad"] = ch["codigo_entidad"].map(lambda c: ent_map.get(c, c))

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

# Footer
st.caption("Fuente: TXT BCRA (Entfin/Tec_Cont). UI oculta códigos; trabaja con descripciones. v6")
