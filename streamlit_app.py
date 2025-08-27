# streamlit_app.py
# Requiere: streamlit pandas altair numpy
# Estructura esperada en el repo:
#   /data/bcra_consolidado.csv  (o varias partes: bcra_consolidado_part*.csv)
#   /data/bcra_nomina.csv
#   /data/bcra_indicadores.csv
#   (opcional) /data/bcra_agregados.csv  -> columnas: 
#     fecha (YYYY-MM), codigo_indicador, indicador, formato,
#     sistema_financiero, banca_publica, banca_privada, banca_nacional, banca_extranjera, companias_financieras
#
# Pestañas:
#  - Panel
#  - Serie
#  - Calculadora
#  - Porcentaje del total

import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from pathlib import Path
import glob
from functools import lru_cache

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
    # utf-8-sig para BOM si lo hay
    try:
        return pd.read_csv(path, encoding="utf-8-sig", **kwargs)
    except Exception:
        # Fallback
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
        if df is None:
            continue
        frames.append(df)
    if not frames:
        return pd.DataFrame(columns=dtypes.keys()).astype(dtypes)
    df_all = pd.concat(frames, ignore_index=True)
    # fecha_dt: primer día del mes
    df_all["fecha"] = df_all["fecha"].str.strip()
    df_all["fecha_dt"] = pd.to_datetime(df_all["fecha"] + "-01", format="%Y-%m-%d", errors="coerce")
    df_all = df_all.dropna(subset=["fecha_dt"]).sort_values(["fecha_dt","codigo_entidad","codigo_indicador"]).reset_index(drop=True)
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
    df["formato"] = df["formato"].str.upper().fillna("N")
    var_map = dict(zip(df["codigo_indicador"], df["indicador"]))
    fmt_map = dict(zip(df["codigo_indicador"], df["formato"]))
    return df, var_map, fmt_map

@st.cache_data(show_spinner=False)
def load_agregados():
    # Opcional. Si existe, lo usamos como “segundo plano”/comparadores.
    if not AGREGADOS.is_file():
        return None
    df = _read_csv_safe(AGREGADOS)
    if df is None or df.empty:
        return None
    # Esperamos columnas: fecha (YYYY-MM), codigo_indicador, indicador, formato, 
    # sistema_financiero, banca_publica, banca_privada, banca_nacional, banca_extranjera, companias_financieras
    df["fecha"] = df["fecha"].astype(str)
    df["fecha_dt"] = pd.to_datetime(df["fecha"].str.strip() + "-01", format="%Y-%m-%d", errors="coerce")
    df = df.dropna(subset=["fecha_dt"]).sort_values(["fecha_dt","codigo_indicador"]).reset_index(drop=True)
    df["formato"] = df["formato"].astype(str).str.upper().replace({"": "N"})
    return df

df = load_consolidado()
ent_map = load_nomina_map()
idx_df, var_map, fmt_map = load_indices()
agg_df = load_agregados()

def entity_label_from_code(code: str) -> str:
    return ent_map.get(code, code)

def format_value(val: float, fmt: str, decimals=2) -> str:
    if pd.isna(val):
        return "—"
    if (fmt or "").upper() == "P":
        return f"{val:.{decimals}f}%"
    return f"{val:,.{decimals}f}".replace(",", "X").replace(".", ",").replace("X", ".")

def percent_change(curr, prev):
    if curr is None or prev in (None, 0) or pd.isna(curr) or pd.isna(prev) or prev == 0:
        return None
    return (curr / prev) - 1.0

def find_variable_codes_by_hint(hints, top_k=4):
    # Devuelve hasta top_k códigos de indicador según "contiene" case-insensitive en descripción
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
    # Si no encontró suficiente, completa con variables más frecuentes
    if len(codes) < top_k and not df.empty:
        freq = df["codigo_indicador"].value_counts().index.tolist()
        for c in freq:
            if c not in codes:
                codes.append(c)
            if len(codes) >= top_k:
                break
    return codes[:top_k]

def default_month_value():
    # intenta usar DEFAULT_MONTH_STR si existe; si no, usa el max disponible
    candidate = pd.to_datetime(DEFAULT_MONTH_STR + "-01", format="%Y-%m-%d", errors="coerce")
    if candidate and not df.empty and candidate in df["fecha_dt"].unique():
        return candidate
    return df["fecha_dt"].max() if not df.empty else None

def to_plot_series(series_vals: pd.Series, fmt: str) -> pd.Series:
    # Para gráficos: si el formato es porcentaje, convertir de “12.3” a “0.123”
    if (fmt or "").upper() == "P":
        return series_vals / 100.0
    return series_vals

def render_delta(curr, prev, fmt):
    rel = percent_change(curr, prev)
    if rel is None:
        return "—"
    sign = "▲" if rel >= 0 else "▼"
    return f"{sign} {abs(rel)*100:.1f}%"

# ───────────────────────── UI ─────────────────────────
st.title("📊 Indicadores del BCRA")

if df.empty:
    st.warning("No se encontraron CSV en `./data`. Subí `bcra_consolidado.csv` (o sus partes), `bcra_nomina.csv` y `bcra_indicadores.csv`.")
    st.stop()

tab_panel, tab_serie, tab_calc, tab_share = st.tabs(["Panel", "Serie", "Calculadora", "Porcentaje del total"])

# =============== PANEL =================
with tab_panel:
    st.subheader("Panel")

    # Selectores
    colA, colB, colC = st.columns([1,1,1])
    with colA:
        # Entidad (mostrar descripciones, mantener códigos por debajo)
        ent_codes_sorted = sorted(df["codigo_entidad"].dropna().unique().tolist(), key=lambda c: ent_map.get(c, c))
        default_ent_idx = 0
        if DEFAULT_ENTITY_CODE in ent_codes_sorted:
            default_ent_idx = ent_codes_sorted.index(DEFAULT_ENTITY_CODE)
        ent_code = st.selectbox(
            "Entidad",
            options=ent_codes_sorted,
            index=default_ent_idx if ent_codes_sorted else 0,
            format_func=lambda code: ent_map.get(code, code)
        )

    with colB:
        # Variables
        default_var_codes = find_variable_codes_by_hint(DEFAULT_VAR_HINTS, top_k=4)
        var_codes_sorted = sorted(df["codigo_indicador"].dropna().unique().tolist(), key=lambda v: var_map.get(v, v))
        var_sel = st.multiselect(
            "Variables",
            options=var_codes_sorted,
            default=[c for c in default_var_codes if c in var_codes_sorted][:4],
            format_func=lambda code: var_map.get(code, code)
        )

    with colC:
        # Mes
        min_date = df["fecha_dt"].min()
        max_date = df["fecha_dt"].max()
        def_month = default_month_value()
        month = st.selectbox(
            "Mes",
            options=sorted(df["fecha_dt"].unique().tolist()),
            index=(sorted(df["fecha_dt"].unique().tolist()).index(def_month) if def_month is not None else 0),
            format_func=lambda d: d.strftime("%Y-%m")
        )

    # Datos seleccionados
    if not var_sel:
        st.info("Seleccioná al menos una variable.")
        st.stop()

    # Panel de métricas
    rows = []
    for vcode in var_sel:
        vdesc = var_map.get(vcode, vcode)
        fmt = fmt_map.get(vcode, "N")

        # Valor del mes seleccionado para la entidad
        row_now = df[(df["codigo_entidad"] == ent_code) & (df["codigo_indicador"] == vcode) & (df["fecha_dt"] == month)]
        val_now = row_now["valor"].iloc[0] if not row_now.empty else np.nan

        # Mes anterior
        prev_month = (month + pd.offsets.MonthEnd(0) - pd.offsets.MonthBegin(1)) - pd.offsets.MonthBegin(0)  # no la necesitamos, usamos shift directo
        month_prev = (month - pd.offsets.MonthBegin(1))  # primer día del mes anterior
        row_prev = df[(df["codigo_entidad"] == ent_code) & (df["codigo_indicador"] == vcode) & (df["fecha_dt"] == month_prev)]
        val_prev = row_prev["valor"].iloc[0] if not row_prev.empty else np.nan

        # Mismo mes año anterior
        month_yoy = (month - pd.DateOffset(years=1))
        row_yoy = df[(df["codigo_entidad"] == ent_code) & (df["codigo_indicador"] == vcode) & (df["fecha_dt"] == month_yoy)]
        val_yoy = row_yoy["valor"].iloc[0] if not row_yoy.empty else np.nan

        rows.append({
            "Variable": vdesc,
            "Código": vcode,
            "Formato": fmt,
            "Actual": val_now,
            "Previo": val_prev,
            "YoY": val_yoy
        })

    panel_df = pd.DataFrame(rows)

    # Métricas (valor + variaciones)
    ncols = 2 if len(panel_df) == 1 else (2 if len(panel_df)==2 else 4)
    cols = st.columns(ncols) if len(panel_df) >= ncols else st.columns(len(panel_df))
    for i, (_, r) in enumerate(panel_df.iterrows()):
        with cols[i % ncols]:
            st.markdown(f"**{r['Variable']}**")
            st.markdown(f"### {format_value(r['Actual'], r['Formato'])}")
            st.caption(
                f"m/m-1: {render_delta(r['Actual'], r['Previo'], r['Formato'])} · "
                f"a/a-1: {render_delta(r['Actual'], r['YoY'], r['Formato'])}"
            )

            # Mini serie con comparadores (si hay agregados)
            series = df[(df["codigo_entidad"] == ent_code) & (df["codigo_indicador"] == r["Código"])][["fecha_dt","valor"]].copy()
            series["serie"] = ent_map.get(ent_code, ent_code)

            plot_frames = [series.rename(columns={"valor":"value"})]
            if agg_df is not None:
                ag = agg_df[agg_df["codigo_indicador"] == r["Código"]].copy()
                # Crear largas
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
                        tmp = ag[["fecha_dt", colname, "formato"]].rename(columns={colname:"value"})
                        tmp["serie"] = label
                        plot_frames.append(tmp[["fecha_dt","value","serie"]])

            plot_df = pd.concat(plot_frames, ignore_index=True) if len(plot_frames)>1 else plot_frames[0]
            # Convertir a ratio si es porcentaje
            fmt = r["Formato"]
            plot_df["value_plot"] = to_plot_series(plot_df["value"], fmt)

            # Graficamos últimos 18 meses
            cutoff = month - pd.DateOffset(months=18)
            plot_df = plot_df[plot_df["fecha_dt"] >= cutoff]

            base = alt.Chart(plot_df).mark_line().encode(
                x=alt.X("fecha_dt:T", title=""),
                y=alt.Y("value_plot:Q", title=""),
                color=alt.Color("serie:N", title="", scale=alt.Scale(scheme="tableau10")),
                strokeDash=alt.StrokeDash("serie:N", legend=None)
                    .scale(domain=[ent_map.get(ent_code, ent_code)], range=[[1,0]])  # la entidad sólida
            ).properties(height=120)

            # Punto resaltado del mes elegido
            highlight = alt.Chart(plot_df[plot_df["fecha_dt"]==month]).mark_point(size=60).encode(
                x="fecha_dt:T", y="value_plot:Q", color="serie:N"
            )

            # Eje formato
            if fmt.upper()=="P":
                y_axis = alt.Axis(format=".1%")
            else:
                y_axis = alt.Axis()

            st.altair_chart(base.encode(y=alt.Y("value_plot:Q", axis=y_axis)) + highlight, use_container_width=True)

    if agg_df is None:
        st.info("Para ver comparadores (Sistema financiero, Banca pública/privada, etc.), agregá `data/bcra_agregados.csv`.")

# =============== SERIE =================
with tab_serie:
    st.subheader("Serie")
    c1, c2 = st.columns([1,2])
    with c1:
        ents = st.multiselect(
            "Entidades",
            options=sorted(df["codigo_entidad"].unique().tolist(), key=lambda c: ent_map.get(c, c)),
            default=[DEFAULT_ENTITY_CODE] if DEFAULT_ENTITY_CODE in df["codigo_entidad"].unique() else [],
            format_func=lambda c: ent_map.get(c, c)
        )
        vars_series = st.multiselect(
            "Variables",
            options=sorted(df["codigo_indicador"].unique().tolist(), key=lambda v: var_map.get(v, v)),
            default=find_variable_codes_by_hint(DEFAULT_VAR_HINTS, top_k=2),
            format_func=lambda v: var_map.get(v, v)
        )
        show_agg = st.checkbox("Mostrar comparadores (si hay agregados)", value=True)

    with c2:
        if not ents or not vars_series:
            st.info("Seleccioná al menos una entidad y una variable.")
        else:
            # Armamos DF para graficar
            plot_frames = []
            for e in ents:
                sub = df[(df["codigo_entidad"]==e) & (df["codigo_indicador"].isin(vars_series))].copy()
                sub["Entidad"] = ent_map.get(e, e)
                sub["Variable"] = sub["codigo_indicador"].map(lambda c: var_map.get(c, c))
                sub["Formato"] = sub["codigo_indicador"].map(lambda c: fmt_map.get(c, "N"))
                # Para graficar: dividir % por 100
                sub["ValorPlot"] = np.where(sub["Formato"].str.upper()=="P", sub["valor"]/100.0, sub["valor"])
                plot_frames.append(sub)

            if show_agg and agg_df is not None:
                ag = agg_df[agg_df["codigo_indicador"].isin(vars_series)].copy()
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
                        t["ValorPlot"] = np.where(t["Formato"].str.upper()=="P", t["valor"]/100.0, t["valor"])
                        long_list.append(t)
                if long_list:
                    plot_frames.append(pd.concat(long_list, ignore_index=True))

            if plot_frames:
                final_plot = pd.concat(plot_frames, ignore_index=True)
                y_axis = alt.Axis(title="Valor")
                chart = (
                    alt.Chart(final_plot)
                    .mark_line(point=True)
                    .encode(
                        x=alt.X("fecha_dt:T", title="Mes"),
                        y=alt.Y("ValorPlot:Q", title="Valor", axis=y_axis),
                        color=alt.Color("Entidad:N", title="Serie"),
                        facet=alt.Facet("Variable:N", title=None, columns=1)
                    )
                    .properties(height=260)
                )
                # Ajuste de formato de eje por variable (si alguna es %)
                # (Mantenemos una única escala por panel; si querés eje por faceta, se puede complicar con resolve_scale)
                st.altair_chart(chart, use_container_width=True)
            else:
                st.info("No hay datos para graficar.")

# =============== CALCULADORA =================
with tab_calc:
    st.subheader("Calculadora de variables")

    c1, c2 = st.columns([1,2])
    with c1:
        ents2 = st.multiselect(
            "Entidades",
            options=sorted(df["codigo_entidad"].unique().tolist(), key=lambda c: ent_map.get(c, c)),
            default=[DEFAULT_ENTITY_CODE] if DEFAULT_ENTITY_CODE in df["codigo_entidad"].unique() else [],
            format_func=lambda c: ent_map.get(c, c)
        )
        st.caption("Construí una fórmula simple: Termo1 (op) Termo2 (op) Termo3 ...")

        # Constructor simple: hasta 5 términos
        ops = ["+", "-", "*", "/"]
        term_vars = []
        term_ops = []
        for i in range(1,6):
            v = st.selectbox(
                f"Variable {i}",
                options=["—"] + sorted(df["codigo_indicador"].unique().tolist(), key=lambda x: var_map.get(x, x)),
                index=0,
                format_func=lambda x: var_map.get(x, x) if x!="—" else "—"
            )
            term_vars.append(None if v=="—" else v)
            if i < 5:
                op = st.selectbox(f"Operación {i}→{i+1}", options=["—"]+ops, index=0, key=f"op_{i}")
                term_ops.append(None if op=="—" else op)

    with c2:
        # Construir expresión
        active_terms = [tv for tv in term_vars if tv]
        if not ents2 or len(active_terms) < 1:
            st.info("Elegí al menos 1 variable y una entidad.")
        else:
            # Generar DataFrame por entidad con columnas = variables seleccionadas
            vars_sel = [v for v in term_vars if v]
            sub = df[df["codigo_indicador"].isin(vars_sel)].copy()
            # Armonizar por entidad/mes
            sub = sub.pivot_table(index=["codigo_entidad","fecha_dt"], columns="codigo_indicador", values="valor", aggfunc="last").reset_index()

            # Evaluar fórmula por filas (sin precedencia compleja; se aplica secuencialmente)
            def apply_expr(row):
                # arranca con primer término
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
                    if pd.isna(res) or pd.isna(val) or op is None:
                        idx_op += 1
                        continue
                    try:
                        if op == "+":
                            res = res + val
                        elif op == "-":
                            res = res - val
                        elif op == "*":
                            res = res * val
                        elif op == "/":
                            res = np.nan if val == 0 else res / val
                    except Exception:
                        res = np.nan
                    idx_op += 1
                return res

            # Filtramos por entidades seleccionadas
            sub = sub[sub["codigo_entidad"].isin(ents2)].copy()
            if sub.empty:
                st.info("No hay intersección de esas variables para las entidades seleccionadas.")
            else:
                sub["Resultado"] = sub.apply(apply_expr, axis=1)
                sub["Entidad"] = sub["codigo_entidad"].map(lambda c: ent_map.get(c, c))

                chart = (
                    alt.Chart(sub.dropna(subset=["Resultado"]))
                    .mark_line(point=True)
                    .encode(
                        x=alt.X("fecha_dt:T", title="Mes"),
                        y=alt.Y("Resultado:Q", title="Resultado"),
                        color=alt.Color("Entidad:N", title="Entidad")
                    )
                    .properties(height=420)
                )
                st.altair_chart(chart, use_container_width=True)

                with st.expander("Ver datos"):
                    st.dataframe(sub[["Entidad","fecha_dt","Resultado"]].sort_values(["Entidad","fecha_dt"]), use_container_width=True)

# =============== PORCENTAJE DEL TOTAL =================
with tab_share:
    st.subheader("Participación sobre el total")

    c1, c2 = st.columns([1,2])
    with c1:
        ent_share = st.selectbox(
            "Entidad",
            options=sorted(df["codigo_entidad"].unique().tolist(), key=lambda c: ent_map.get(c, c)),
            index=(sorted(df["codigo_entidad"].unique().tolist()).index(DEFAULT_ENTITY_CODE) if DEFAULT_ENTITY_CODE in df["codigo_entidad"].unique() else 0),
            format_func=lambda c: ent_map.get(c, c)
        )
        var_share = st.selectbox(
            "Variable",
            options=sorted(df["codigo_indicador"].unique().tolist(), key=lambda v: var_map.get(v, v)),
            index=0,
            format_func=lambda v: var_map.get(v, v)
        )
        st.caption("Nota: tiene sentido sobre **variables aditivas** (cantidades, montos). En porcentajes / ratios puede no ser interpretable.")

    with c2:
        # Serie de la entidad
        a = df[(df["codigo_entidad"]==ent_share) & (df["codigo_indicador"]==var_share)][["fecha_dt","valor"]].copy()
        a = a.rename(columns={"valor":"ent_val"})

        # Serie total: si hay agregados, usamos "sistema_financiero"; si no, sumamos todas las entidades (aditivo)
        if agg_df is not None and "sistema_financiero" in agg_df.columns:
            tot = agg_df[agg_df["codigo_indicador"]==var_share][["fecha_dt","sistema_financiero","formato"]].rename(columns={"sistema_financiero":"tot_val"})
        else:
            # Total por mes = suma de entidades disponibles (advertencia en ratios)
            tot = df[df["codigo_indicador"]==var_share].groupby("fecha_dt", as_index=False)["valor"].sum().rename(columns={"valor":"tot_val"})
            # formato:
            fmt = fmt_map.get(var_share, "N")
            tot["formato"] = fmt

        merged = pd.merge(a, tot, on="fecha_dt", how="inner")
        merged["share"] = np.where(merged["tot_val"]==0, np.nan, merged["ent_val"]/merged["tot_val"])
        merged["Entidad"] = ent_map.get(ent_share, ent_share)
        fmt = fmt_map.get(var_share, "N")

        if merged.empty:
            st.info("No hay datos para calcular participación.")
        else:
            chart = (
                alt.Chart(merged)
                .mark_line(point=True)
                .encode(
                    x=alt.X("fecha_dt:T", title="Mes"),
                    y=alt.Y("share:Q", title="% del total", axis=alt.Axis(format=".1%")),
                    color=alt.Color("Entidad:N", legend=None)
                )
                .properties(height=420)
            )
            st.altair_chart(chart, use_container_width=True)

            # Último valor
            last = merged.sort_values("fecha_dt").tail(1)["share"].iloc[0]
            st.metric("Último % del total", f"{last*100:.2f}%")

# Footer
st.caption("Fuente: TXT BCRA (Entfin/Tec_Cont). UI oculta códigos; trabaja con descripciones.")
