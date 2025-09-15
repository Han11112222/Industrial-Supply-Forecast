# app.py — 도시가스 산업용 업종별 예측 (연도 범위 예측 · 다년 표시)
# • 좌측: 학습 연도(멀티), 예측 구간(시작연~종료연) — 월 제거
# • 학습: 선택 연도만 사용해 3가지 방법(OLS/CAGR/Holt)으로 다년 예측
# • 표/그래프: 예측연도 전부(예: 2026, 2027)를 표시
# • 엑셀: 시트1 전체표(모든 예측연도 열), 시트2~4 각 방법(표+Top20 막대+총합 라인)

from pathlib import Path
from io import BytesIO
import numpy as np
import pandas as pd
import streamlit as st

# statsmodels (Holt)
try:
    from statsmodels.tsa.holtwinters import Holt
except Exception:
    Holt = None

# openpyxl charts (엑셀 내 차트)
from openpyxl import Workbook
from openpyxl.utils.dataframe import dataframe_to_rows
from openpyxl.chart import BarChart, Reference, LineChart

import altair as alt

st.set_page_config(page_title="도시가스 공급량·판매량 예측 (업종별)", layout="wide")
st.title("도시가스 공급량·판매량 예측 (업종별)")
st.caption("RAW 엑셀 업로드 → 학습연도 선택 → 추세 예측(3종) → 정렬/총합/그래프/다운로드")

# ───────────────── 사이드바: 파일/방법 ─────────────────
with st.sidebar:
    st.header("① 데이터 불러오기")
    up = st.file_uploader("원본 엑셀 업로드 (.xlsx)", type=["xlsx"])
    sample_path = Path("산업용_업종별.xlsx")
    use_sample = st.checkbox(f"Repo 내 파일 사용: {sample_path.name}", value=sample_path.exists())

    st.divider()
    st.header("② 예측 방법")
    method_choices = ["선형추세(OLS)", "CAGR", "Holt(지수평활)"]
    methods = st.multiselect("방법 선택(정렬 기준은 첫 번째 선택 값)", method_choices, default=method_choices)

# ───────────────── 유틸 ─────────────────
METHOD_DESC = {
    "선형추세(OLS)": "연도(t)와 사용량(y)의 직선관계 y_t = a + b t 을 최소제곱으로 적합.",
    "CAGR": "시점1→시점2 복리성장률 g로 종료연도 예측 = y_end × (1+g)^(Δ연도).",
    "Holt(지수평활)": "수준 l_t, 추세 b_t 를 지수 가중으로 갱신; 예측 = l_T + h·b_T (계절성 미포함).",
}

@st.cache_data
def read_excel_to_long(file) -> pd.DataFrame:
    """엑셀을 ‘업종, 연도, 사용량’ Long 형태로 변환. 모든 4자리 연도 헤더 감지."""
    df = pd.read_excel(file, engine="openpyxl")
    obj_cols = [c for c in df.columns if df[c].dtype == "object"]
    cat_col = obj_cols[0] if obj_cols else df.columns[0]

    # ‘연도’로 보이는 열 자동 감지
    year_cols = []
    for c in df.columns:
        s = str(c)
        # 헤더 안의 4자리 연도 추출
        y = pd.Series([s]).str.extract(r"((?:19|20)\d{2})")[0][0]
        if pd.to_numeric(y, errors="coerce") is not None:
            year_cols.append(c)
    if not year_cols:
        # 헤더에 연도표기가 없으면 숫자형 열을 후보로
        year_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and c != cat_col]

    m = df[[cat_col] + year_cols].copy().melt(id_vars=[cat_col], var_name="연도열", value_name="사용량")
    years_extracted = m["연도열"].astype(str).str.extract(r"((?:19|20)\d{2})")[0]
    m["연도"] = pd.to_numeric(years_extracted, errors="coerce").astype("Int64")
    m = m.dropna(subset=["연도"])

    m["연도"] = m["연도"].astype(int)
    m = m.rename(columns={cat_col: "업종"})
    m["사용량"] = pd.to_numeric(m["사용량"], errors="coerce")
    m = m.dropna(subset=["업종", "사용량"])
    return m[["업종", "연도", "사용량"]]

def _linear_forecast(x_years, y_vals, targets):
    coef = np.polyfit(x_years, y_vals, 1)
    fitted = np.polyval(coef, x_years)
    preds = [float(np.polyval(coef, t)) for t in targets]
    return preds, fitted

def _cagr_forecast(x_years, y_vals, targets):
    y_start, y_end = float(y_vals[0]), float(y_vals[-1]); n = int(x_years[-1] - x_years[0])
    if y_start <= 0 or y_end <= 0 or n <= 0:
        return _linear_forecast(x_years, y_vals, targets)
    g = (y_end / y_start) ** (1.0 / n) - 1.0
    last = x_years[-1]
    preds = [float(y_end * (1.0 + g) ** (t - last)) for t in targets]
    return preds, np.array(y_vals, dtype=float)

def _holt_forecast(y_vals, last_train_year, targets):
    """Holt로 다년 예측. targets는 오름차순 가정. 실패/비정상 시 OLS로 대체."""
    x_years = list(range(last_train_year - len(y_vals) + 1, last_train_year + 1))
    if Holt is None or len(y_vals) < 2 or any(t <= last_train_year for t in targets):
        return _linear_forecast(x_years, y_vals, targets)
    try:
        model = Holt(np.asarray(y_vals), exponential=False, damped_trend=False, initialization_method="estimated")
        fit = model.fit(optimized=True)
        max_h = max(t - last_train_year for t in targets)
        fc = fit.forecast(max_h)  # 1-step부터
        preds = [float(fc[h - 1]) for h in [t - last_train_year for t in targets]]
        return preds, np.array(fit.fittedvalues, dtype=float)
    except Exception:
        return _linear_forecast(x_years, y_vals, targets)

def fmt_int_with_comma(x):
    if pd.isna(x): return ""
    try: return f"{int(round(float(x))):,}"
    except Exception: return x

# ───────────────── 본문 실행 ─────────────────
df_long_for_ui = None
if up is not None:
    df_long_for_ui = read_excel_to_long(up)
elif use_sample and sample_path.exists():
    df_long_for_ui = read_excel_to_long(sample_path)

TRAIN_YEARS = None
FORECAST_YEARS = []
if df_long_for_ui is not None:
    all_years_sorted = sorted(df_long_for_ui["연도"].unique().tolist())
    default_train = all_years_sorted[-5:] if len(all_years_sorted) >= 5 else all_years_sorted

    with st.sidebar:
        st.divider()
        st.header("③ 학습 데이터 연도 선택")
        TRAIN_YEARS = st.multiselect("연도 선택", all_years_sorted, default=default_train)
        if not TRAIN_YEARS:
            st.info("학습 연도를 1개 이상 선택하세요.")
        TRAIN_YEARS = sorted(TRAIN_YEARS)

        st.divider()
        st.header("④ 예측 설정 (연도만)")
        future_years = list(range(all_years_sorted[-1], all_years_sorted[-1] + 10))
        yr_options = sorted(set(all_years_sorted + future_years))
        start_year = st.selectbox("예측 시작(연)", yr_options, index=yr_options.index(all_years_sorted[-1]))
        end_year   = st.selectbox("예측 종료(연)", yr_options, index=min(len(yr_options)-1, yr_options.index(all_years_sorted[-1])+1))
        if end_year < start_year:
            st.warning("예측 종료(연)는 예측 시작(연)보다 크거나 같아야 합니다.")
        FORECAST_YEARS = list(range(min(start_year, end_year), max(start_year, end_year) + 1))

        st.divider()
        run = st.button("예측 시작")
else:
    run = False
    st.info("좌측에서 엑셀 업로드 또는 ‘Repo 내 파일 사용’을 선택하세요.")

if run:
    if df_long_for_ui is None:
        st.error("데이터를 먼저 불러오세요."); st.stop()
    if not TRAIN_YEARS:
        st.error("학습 연도를 선택하세요."); st.stop()
    if not methods:
        st.error("예측 방법을 1개 이상 선택하세요."); st.stop()
    if not FORECAST_YEARS:
        st.error("예측 연도를 설정하세요."); st.stop()

    # 실제 사용 데이터
    df_long = df_long_for_ui.copy()
    st.success(f"로드 완료: 업종 {df_long['업종'].nunique()}개, 전체 연도 {df_long['연도'].min()}–{df_long['연도'].max()}")

    with st.expander("예측 방법 설명", expanded=False):
        for k, v in METHOD_DESC.items():
            st.markdown(f"**{k}** — {v}")

    # 피벗(학습 연도만 사용)
    pv_all = df_long.pivot_table(index="업종", columns="연도", values="사용량", aggfunc="sum")
    missing = [y for y in TRAIN_YEARS if y not in pv_all.columns]
    if missing:
        st.error(f"선택한 학습 연도에 데이터가 없습니다: {missing}"); st.stop()
    pv = pv_all.reindex(columns=TRAIN_YEARS).fillna(0)

    # ── 업종별 다년 예측 ─────────────────────────────────────
    result_df = pv.copy()
    result_df.columns = [f"{c} 실적" for c in result_df.columns]

    fit_store = {}  # (industry, method) -> (x_years, fitted_values)

    for industry, row in pv.iterrows():
        y = row.values.astype(float).tolist()
        x = TRAIN_YEARS
        last_train = x[-1]

        for m in methods:
            if m == "선형추세(OLS)":
                preds, fitted = _linear_forecast(x, y, FORECAST_YEARS)
            elif m == "CAGR":
                preds, fitted = _cagr_forecast(x, y, FORECAST_YEARS)
            elif m == "Holt(지수평활)":
                preds, fitted = _holt_forecast(y, last_train, FORECAST_YEARS)
            else:
                continue

            fit_store[(industry, m)] = (x, fitted)
            for yy, pred in zip(FORECAST_YEARS, preds):
                col = f"{m}({yy})"
                if col not in result_df.columns:
                    result_df[col] = np.nan
                result_df.loc[industry, col] = pred

    # 정렬 기준: 첫 번째로 선택한 방법의 "종료연도" 예측열
    sort_method = methods[0]
    sort_col = f"{sort_method}({FORECAST_YEARS[-1]})"
    if sort_col not in result_df.columns:
        fallback_cols = [c for c in result_df.columns if c.endswith(f"({FORECAST_YEARS[-1]})")]
        sort_col = fallback_cols[0] if fallback_cols else result_df.columns[-1]

    final_sorted = result_df.sort_values(by=sort_col, ascending=False)

    # ── 총합 행 계산(실적/예측 모두) — KeyError 수정 포인트 ──
    # 각 열을 직접 합산(numeric_only=True)하여 자료형 불일치 문제 제거
    totals_series = final_sorted.sum(axis=0, numeric_only=True)
    totals = {col: float(totals_series.get(col, 0.0)) for col in final_sorted.columns}
    total_row = pd.DataFrame([totals], index=["총합"])
    final_sorted_with_total = pd.concat([final_sorted, total_row], axis=0)

    # 표시용(콤마)
    display_df = final_sorted_with_total.copy()
    display_df.insert(0, "업종", display_df.index)
    for c in display_df.columns[1:]:
        display_df[c] = display_df[c].apply(fmt_int_with_comma)

    st.subheader(
        f"업종별 예측 표 — 학습연도 {TRAIN_YEARS[0]}–{TRAIN_YEARS[-1]} / 예측연도 {FORECAST_YEARS[0]}–{FORECAST_YEARS[-1]}"
    )
    st.dataframe(display_df.reset_index(drop=True), use_container_width=True)

    # ───────────────── 그래프: 연도별 총합(실적 + 예측 포인트) ─────────────────
    tot_actual = pv.sum(axis=0).reset_index()
    tot_actual.columns = ["연도", "합계"]

    pts_rows = []
    for m in methods:
        for yy in FORECAST_YEARS:
            col = f"{m}({yy})"
            if col in final_sorted.columns:
                pts_rows.append({"연도": yy, "방법": m, "값": float(final_sorted[col].sum())})
    pts_df = pd.DataFrame(pts_rows)

    area = alt.Chart(tot_actual).mark_area(opacity=0.25).encode(
        x=alt.X("연도:O", title="연도"),
        y=alt.Y("합계:Q", title="총합(실적)", axis=alt.Axis(format=",")),
        tooltip=[alt.Tooltip("연도:O"), alt.Tooltip("합계:Q", format=",")]
    )
    line = alt.Chart(tot_actual).mark_line(size=3).encode(
        x="연도:O", y=alt.Y("합계:Q", axis=alt.Axis(format=","))
    )
    pts = alt.Chart(pts_df).mark_point(size=150, filled=True).encode(
        x="연도:O",
        y=alt.Y("값:Q", axis=alt.Axis(format=",")),
        color=alt.Color("방법:N", legend=alt.Legend(title="방법")),
        shape=alt.Shape("방법:N"),
        tooltip=[alt.Tooltip("방법:N"), alt.Tooltip("연도:O"), alt.Tooltip("값:Q", format=",")]
    )
    labels = pts.mark_text(dy=-12, fontWeight="bold").encode(text=alt.Text("값:Q", format=","))

    st.subheader("연도별 총합 그래프 (실적 라인 + 예측 포인트)")
    st.altair_chart((area + line + pts + labels).interactive(), use_container_width=True, theme="streamlit")

    # ───────────────── Top-10: 예측 비교(막대) + 실적 추이(라인) ─────────────────
    st.markdown("### 상위 10개 업종 — 예측 비교 / 실적 추이")
    top10_inds = final_sorted.head(10).index.tolist()

    yy_pick = st.radio("막대그래프 기준 예측연도", FORECAST_YEARS, index=len(FORECAST_YEARS)-1, horizontal=True)

    method_cols_for_plot = [f"{m}({yy_pick})" for m in methods if f"{m}({yy_pick})" in final_sorted.columns]
    pred_long = final_sorted.loc[top10_inds, method_cols_for_plot].reset_index().melt(
        id_vars="index", var_name="방법연도", value_name="예측"
    ).rename(columns={"index": "업종"})
    pred_long["방법"] = pred_long["방법연도"].str.replace(r"\(\d{4}\)$", "", regex=True)

    method_sel = alt.selection_point(fields=["방법"], bind="legend")
    bars = alt.Chart(pred_long).mark_bar().encode(
        x=alt.X("업종:N", sort=top10_inds, title=None),
        xOffset=alt.XOffset("방법:N"),
        y=alt.Y("예측:Q", axis=alt.Axis(format=","), title=f"{yy_pick} 예측"),
        color=alt.Color("방법:N", legend=alt.Legend(title="방법")),
        opacity=alt.condition(method_sel, alt.value(1.0), alt.value(0.25)),
        tooltip=[alt.Tooltip("업종:N"), alt.Tooltip("방법:N"), alt.Tooltip("예측:Q", format=",")]
    ).add_params(method_sel).properties(height=420)
    bar_txt = bars.mark_text(dy=-5, fontSize=10).encode(text=alt.Text("예측:Q", format=","))

    actual_long = pv.loc[top10_inds, TRAIN_YEARS].reset_index().melt(
        id_vars="업종", var_name="연도", value_name="사용량"
    )
    ind_sel = alt.selection_point(fields=["업종"], bind="legend")
    lines = alt.Chart(actual_long).mark_line(point=True, strokeWidth=3).encode(
        x=alt.X("연도:O", title=None),
        y=alt.Y("사용량:Q", axis=alt.Axis(format=",")),
        color=alt.Color("업종:N", sort=top10_inds, legend=alt.Legend(title="업종(클릭으로 강조)")),
        opacity=alt.condition(ind_sel, alt.value(1.0), alt.value(0.25)),
        tooltip=[alt.Tooltip("업종:N"), alt.Tooltip("연도:O"), alt.Tooltip("사용량:Q", format=",")]
    ).add_params(ind_sel).properties(height=420)

    c1, c2 = st.columns(2)
    with c1: st.altair_chart((bars + bar_txt).interactive(), use_container_width=True, theme="streamlit")
    with c2: st.altair_chart(lines.interactive(), use_container_width=True, theme="streamlit")

    # ───────────────── 다운로드 (다중 시트 + 차트 포함) ─────────────────
    st.subheader("다운로드")

    out_all = final_sorted_with_total.copy()
    out_all.insert(0, "업종", out_all.index)

    fname_suffix = f"{FORECAST_YEARS[0]}-{FORECAST_YEARS[-1]}"

    wb = Workbook(); wb.remove(wb.active)

    # 시트 1: 전체
    ws_all = wb.create_sheet("전체")
    for r in dataframe_to_rows(out_all, index=False, header=True):
        ws_all.append(r)

    # 각 방법별 시트 (표 + Top20 Bar + Totals Line)
    def _add_method_sheet(method_label: str):
        ws = wb.create_sheet(method_label)
        # 표: 실적 + 해당 방법의 모든 예측연도 열
        dfm = pv.copy(); dfm.columns = [f"{c} 실적" for c in dfm.columns]
        pred_cols = [f"{method_label}({yy})" for yy in FORECAST_YEARS if f"{method_label}({yy})" in final_sorted.columns]
        for col in pred_cols:
            dfm[col] = final_sorted[col]
        # 업종 열 이름 제대로 복원(중요: rename 대상은 'index')
        order_col = (pred_cols[-1] if pred_cols else dfm.columns[-1])
        dfm = dfm.sort_values(by=order_col, ascending=False).reset_index().rename(columns={"index": "업종"})

        for r in dataframe_to_rows(dfm, index=False, header=True):
            ws.append(r)

        # Top20 Bar (기본: 종료연도)
        if pred_cols:
            topN = min(20, len(dfm))
            chart_year = FORECAST_YEARS[-1]
            use_col = f"{method_label}({chart_year})"
            start_col = dfm.shape[1] + 2
            ws.cell(row=1, column=start_col, value="업종")
            ws.cell(row=1, column=start_col + 1, value=f"{chart_year} 예측")
            for i in range(topN):
                ws.cell(row=i + 2, column=start_col, value=dfm.loc[i, "업종"])
                ws.cell(row=i + 2, column=start_col + 1, value=float(dfm.loc[i, use_col] or 0))
            bar = BarChart(); bar.title = f"Top-20 {chart_year} ({method_label})"
            data = Reference(ws, min_col=start_col + 1, min_row=1, max_row=topN + 1)
            cats = Reference(ws, min_col=start_col,     min_row=2, max_row=topN + 1)
            bar.add_data(data, titles_from_data=True); bar.set_categories(cats)
            bar.y_axis.number_format = '#,##0'
            ws.add_chart(bar, ws.cell(row=2, column=start_col + 3).coordinate)

        # 연도별 총합 Line (실적 + 이 방법의 모든 예측연도 합계)
        line_anchor_col = (dfm.shape[1] + 8)
        ws.cell(row=1, column=line_anchor_col,     value="연도")
        ws.cell(row=1, column=line_anchor_col + 1, value="총합")
        # 실적 총합
        for i, y in enumerate(TRAIN_YEARS, start=2):
            ws.cell(row=i, column=line_anchor_col,     value=y)
            ws.cell(row=i, column=line_anchor_col + 1, value=float(pv[y.replace(" 실적","")] if isinstance(y,str) else pv[y]).sum() if (y if not isinstance(y,str) else int(y.replace(" 실적",""))) in pv.columns else float(pv[y if not isinstance(y,str) else int(y.replace(" 실적",""))].sum()))
        # 보다 간단/안전: 직접 합산
        ws.delete_cols(line_anchor_col, 2)  # 위 줄 보정
        ws.cell(row=1, column=line_anchor_col,     value="연도")
        ws.cell(row=1, column=line_anchor_col + 1, value="총합")
        for i, y in enumerate(TRAIN_YEARS, start=2):
            ws.cell(row=i, column=line_anchor_col,     value=y)
            ws.cell(row=i, column=line_anchor_col + 1, value=float(pv[y].sum()))
        # 예측 총합(해당 방법)
        base = len(TRAIN_YEARS) + 2
        for j, yy in enumerate(FORECAST_YEARS):
            ws.cell(row=base + j, column=line_anchor_col,     value=yy)
            colname = f"{method_label}({yy})"
            tot_val = float(final_sorted[colname].sum()) if colname in final_sorted.columns else 0.0
            ws.cell(row=base + j, column=line_anchor_col + 1, value=tot_val)

        lchart = LineChart(); lchart.title = f"연도별 총합(실적 + 예측, {method_label})"
        max_row = base + len(FORECAST_YEARS) - 1
        d = Reference(ws, min_col=line_anchor_col + 1, min_row=1, max_row=max_row)
        c = Reference(ws, min_col=line_anchor_col,     min_row=2, max_row=max_row)
        lchart.add_data(d, titles_from_data=True); lchart.set_categories(c)
        lchart.y_axis.number_format = '#,##0'
        ws.add_chart(lchart, ws.cell(row=2, column=line_anchor_col + 3).coordinate)

    for m in methods:
        _add_method_sheet(m)

    bio = BytesIO(); wb.save(bio)
    st.download_button("엑셀(xlsx) 다운로드 (모든 예측연도 포함)",
                       bio.getvalue(),
                       file_name=f"industry_forecast_{fname_suffix}.xlsx",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    st.download_button(
        "업종별 예측표 CSV 다운로드 (모든 예측연도 포함)",
        out_all.to_csv(index=False).encode("utf-8-sig"),
        file_name=f"industry_forecast_{fname_suffix}.csv",
        mime="text/csv"
    )
