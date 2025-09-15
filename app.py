# app.py — 도시가스 산업용 업종별 예측
# • 좌측: 학습 연도 선택(멀티), 예측 구간(연·월) 입력
# • 학습: 선택한 연도만 사용해 3가지 방법(OLS/CAGR/Holt)으로 예측
# • 표시/다운로드: 예측 구간의 ‘종료 연도’를 예측 연도로 사용
# • 그래프: 총합(실적+종료연도 포인트), Top10 동적 막대/라인
# • 엑셀: 시트1 전체표, 시트2~4 각 방법(표+Top20 막대+총합 라인)

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

st.set_page_config(page_title="도시가스 공급량·판매량 예측 (업종별)", layout="wide")
st.title("도시가스 공급량·판매량 예측 (업종별)")
st.caption("RAW 엑셀 업로드 → 학습연도 선택 → 추세 예측(3종) → 정렬/총합/그래프/다운로드")

# ───────────────── 사이드바: 파일/방법/버튼 ─────────────────
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
        if pd.notna(pd.to_numeric(pd.Series([s]).str.extract(r"((?:19|20)\d{2})")[0][0], errors="coerce")):
            year_cols.append(c)

    if not year_cols:
        # 헤더에서 숫자 4자리가 없으면 모든 숫자형 열을 후보로
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

def _linear_forecast(x, y, target):
    coef = np.polyfit(x, y, 1)
    return np.polyval(coef, target), np.polyval(coef, x)

def _cagr_forecast(x, y, target):
    y_start, y_end = float(y[0]), float(y[-1]); n = int(x[-1] - x[0])
    if y_start <= 0 or y_end <= 0 or n <= 0:
        yh, fit = _linear_forecast(x, y, target); return yh, np.array(y)
    g = (y_end / y_start) ** (1.0 / n) - 1.0
    # Δ연도 = target - 마지막 학습연도
    return y_end * (1.0 + g) ** (target - x[-1]), np.array(y)

def _holt_forecast(y, steps=1):
    if Holt is None: return None, None
    try:
        model = Holt(np.asarray(y), exponential=False, damped_trend=False, initialization_method="estimated")
        fit = model.fit(optimized=True)
        return float(fit.forecast(steps)[-1]), np.array(fit.fittedvalues)
    except Exception:
        return None, None

def fmt_int_with_comma(x):
    if pd.isna(x): return ""
    try: return f"{int(round(float(x))):,}"
    except Exception: return x

# ───────────────── 본문 실행 ─────────────────
# 데이터 미리 읽어 ‘학습연도/예측구간’ 컨트롤을 그릴 수 있게 함
df_long_for_ui = None
if up is not None:
    df_long_for_ui = read_excel_to_long(up)
elif use_sample and sample_path.exists():
    df_long_for_ui = read_excel_to_long(sample_path)

TRAIN_YEARS = None
TARGET_YEAR = None
meta_months = {}

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
        st.header("④ 예측 설정")
        # 예측 구간(연·월). 월은 표기/메타 정보로만 사용(연단위 데이터).
        # 선택 범위(연)는 학습 가능 범위를 벗어나도 되도록 여유롭게 제공
        future_years = list(range(all_years_sorted[-1], all_years_sorted[-1] + 6))
        yr_options = sorted(set(all_years_sorted + future_years))

        start_year = st.selectbox("예측 시작(연)", yr_options, index=yr_options.index(all_years_sorted[-1]))
        start_month = st.selectbox("예측 시작(월)", list(range(1, 13)), index=0)
        end_year = st.selectbox("예측 종료(연)", yr_options, index=yr_options.index(all_years_sorted[-1]))
        end_month = st.selectbox("예측 종료(월)", list(range(1, 13)), index=11)

        if end_year < start_year:
            st.warning("예측 종료(연)는 예측 시작(연)보다 크거나 같아야 합니다.")
        TARGET_YEAR = max(start_year, end_year)  # 표시/다운로드에 사용할 예측 연도
        meta_months = {"start_year": start_year, "start_month": start_month,
                       "end_year": end_year, "end_month": end_month}

        st.divider()
        run = st.button("예측 시작")

else:
    run = False
    st.info("좌측에서 엑셀 업로드 또는 ‘Repo 내 파일 사용’을 선택하세요.")

if run:
    if df_long_for_ui is None:
        st.error("데이터를 먼저 불러오세요.")
        st.stop()
    if not TRAIN_YEARS:
        st.error("학습 연도를 선택하세요.")
        st.stop()

    # 실제 사용 데이터
    df_long = df_long_for_ui.copy()
    st.success(f"로드 완료: 업종 {df_long['업종'].nunique()}개, 전체 연도 {df_long['연도'].min()}–{df_long['연도'].max()}")

    with st.expander("예측 방법 설명", expanded=False):
        for k, v in METHOD_DESC.items():
            st.markdown(f"**{k}** — {v}")

    if not methods:
        st.warning("최소 1개 이상의 예측 방법을 선택해 주세요.")
        st.stop()

    # 피벗(학습 연도만 사용)
    pv_all = df_long.pivot_table(index="업종", columns="연도", values="사용량", aggfunc="sum")
    # 학습 연도만 포함
    missing = [y for y in TRAIN_YEARS if y not in pv_all.columns]
    if missing:
        st.error(f"선택한 학습 연도에 데이터가 없습니다: {missing}")
        st.stop()
    pv = pv_all.reindex(columns=TRAIN_YEARS).fillna(0)

    # ── 업종별 예측 (TARGET_YEAR 기준) ──
    out_rows, fit_store = [], {}
    for industry, row in pv.iterrows():
        y = row.values.astype(float).tolist(); x = TRAIN_YEARS
        for m in methods:
            if m == "선형추세(OLS)":
                yh, fitted = _linear_forecast(x, y, TARGET_YEAR)
                out_rows.append([industry, m, yh]); fit_store[(industry, m)] = (x, fitted)
            elif m == "CAGR":
                yh, fitted = _cagr_forecast(x, y, TARGET_YEAR)
                out_rows.append([industry, m, yh]); fit_store[(industry, m)] = (x, fitted if fitted is not None else np.array(y))
            elif m == "Holt(지수평활)":
                yh, fitted = _holt_forecast(y, steps=1)
                name = m if yh is not None else "Holt(지수평활,대체:선형)"
                if yh is None: yh, fitted = _linear_forecast(x, y, TARGET_YEAR)
                out_rows.append([industry, name, yh]); fit_store[(industry, name)] = (x, fitted if fitted is not None else np.array(y))

    forecast_df = pd.DataFrame(out_rows, columns=["업종", "방법", f"{TARGET_YEAR} 예측"])
    wide = forecast_df.pivot_table(index="업종", columns="방법", values=f"{TARGET_YEAR} 예측", aggfunc="first")

    # ── 최종표(숫자형) + 정렬 + 총합 ──
    final_numeric = pv.copy(); final_numeric.columns = [f"{c} 실적" for c in final_numeric.columns]
    final_numeric = final_numeric.join(wide)

    # 정렬 기준
    holt_col = next((c for c in wide.columns if c.startswith("Holt")), None)
    sort_pref = []
    if "선형추세(OLS)" in wide.columns: sort_pref.append("선형추세(OLS)")
    if "CAGR" in wide.columns:          sort_pref.append("CAGR")
    if holt_col:                         sort_pref.append(holt_col)
    cand = [m for m in methods if m in sort_pref or (m == "Holt(지수평활)" and holt_col)]
    sort_col = cand[0] if cand else (holt_col or (sort_pref[0] if sort_pref else None))
    if sort_col == "Holt(지수평활)": sort_col = holt_col

    if sort_col is None:
        st.error("정렬 기준으로 사용할 예측 열이 없습니다.")
        st.stop()

    final_sorted = final_numeric.sort_values(by=sort_col, ascending=False)

    totals = {f"{y} 실적": pv[y].sum() for y in TRAIN_YEARS}
    for col in wide.columns: totals[col] = wide[col].sum()
    total_row = pd.DataFrame([totals], index=["총합"])
    final_sorted_with_total = pd.concat([final_sorted, total_row], axis=0)

    # 표시용(콤마)
    display_df = final_sorted_with_total.copy()
    display_df.insert(0, "업종", display_df.index)
    for c in display_df.columns[1:]:
        display_df[c] = display_df[c].apply(fmt_int_with_comma)

    st.subheader(f"업종별 예측 표 — 학습연도 {TRAIN_YEARS[0]}–{TRAIN_YEARS[-1]} / 예측연도 {TARGET_YEAR} "
                 f"({meta_months['start_year']}.{meta_months['start_month']:02d} ~ {meta_months['end_year']}.{meta_months['end_month']:02d})")
    st.dataframe(display_df.reset_index(drop=True), use_container_width=True)

    # ───────────────── 그래프 ─────────────────
    import altair as alt

    st.subheader("연도별 총합 그래프 (실적 + 예측 포인트)")
    tot_actual = pv.sum(axis=0).reset_index()
    tot_actual.columns = ["연도", "합계"]

    area = alt.Chart(tot_actual).mark_area(opacity=0.25).encode(
        x=alt.X("연도:O", title="연도"),
        y=alt.Y("합계:Q", title="총합(실적)", axis=alt.Axis(format=",")),
        tooltip=[alt.Tooltip("연도:O"), alt.Tooltip("합계:Q", format=",")]
    )
    line = alt.Chart(tot_actual).mark_line(size=3).encode(
        x="연도:O", y=alt.Y("합계:Q", axis=alt.Axis(format=","))
    )
    methods_cols = list(wide.columns)
    pts_df = pd.DataFrame({
        "연도": [TARGET_YEAR] * len(methods_cols),
        "방법": methods_cols,
        "값":   [wide[c].sum() for c in methods_cols],
    })
    pts = alt.Chart(pts_df).mark_point(size=150, filled=True).encode(
        x="연도:O", y=alt.Y("값:Q", axis=alt.Axis(format=",")),
        color=alt.Color("방법:N", legend=alt.Legend(title="방법")),
        tooltip=[alt.Tooltip("방법:N"), alt.Tooltip("값:Q", format=",")]
    )
    labels = pts.mark_text(dy=-12, fontWeight="bold").encode(text=alt.Text("값:Q", format=","))

    st.altair_chart((area + line + pts + labels).interactive(), use_container_width=True, theme="streamlit")

    # Top-10 동적 그래프
    st.markdown("### 상위 10개 업종 — 예측 비교 / 실적 추이")
    top10_inds = final_sorted.head(10).index.tolist()

    method_cols_for_plot = []
    if "선형추세(OLS)" in wide.columns: method_cols_for_plot.append("선형추세(OLS)")
    if "CAGR"          in wide.columns: method_cols_for_plot.append("CAGR")
    holt_real = next((c for c in wide.columns if c.startswith("Holt")), None)
    if holt_real: method_cols_for_plot.append(holt_real)

    # (A) 예측 비교 (그룹형 막대)
    pred_long = wide.loc[top10_inds, method_cols_for_plot].reset_index().melt(
        id_vars="업종", var_name="방법", value_name="예측"
    )
    method_sel = alt.selection_point(fields=["방법"], bind="legend")
    bars = alt.Chart(pred_long).mark_bar().encode(
        x=alt.X("업종:N", sort=top10_inds, title=None),
        xOffset=alt.XOffset("방법:N"),
        y=alt.Y("예측:Q", axis=alt.Axis(format=","), title=f"{TARGET_YEAR} 예측"),
        color=alt.Color("방법:N", legend=alt.Legend(title="방법")),
        opacity=alt.condition(method_sel, alt.value(1.0), alt.value(0.25)),
        tooltip=[alt.Tooltip("업종:N"), alt.Tooltip("방법:N"), alt.Tooltip("예측:Q", format=",")]
    ).add_params(method_sel).properties(height=420)
    bar_txt = bars.mark_text(dy=-5, fontSize=10).encode(text=alt.Text("예측:Q", format=","))

    # (B) 실적 추이 (멀티라인)
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

    # 파일명에 예측 구간(연·월) 반영
    fname_suffix = f"{meta_months['start_year']}.{meta_months['start_month']:02d}-{meta_months['end_year']}.{meta_months['end_month']:02d}"

    wb = Workbook(); wb.remove(wb.active)
    # 시트 1: 전체
    ws_all = wb.create_sheet("전체")
    for r in dataframe_to_rows(out_all, index=False, header=True):
        ws_all.append(r)

    # 각 방법별 시트 (표 + Top20 Bar + Totals Line)
    def _add_method_sheet(method_label: str, pred_col_name: str):
        ws = wb.create_sheet(method_label)
        dfm = pv.copy(); dfm.columns = [f"{c} 실적" for c in dfm.columns]
        if pred_col_name not in wide.columns:
            holt_real2 = next((c for c in wide.columns if c.startswith("Holt")), None)
            use_col = holt_real2 if holt_real2 else pred_col_name
        else:
            use_col = pred_col_name
        dfm[use_col] = wide[use_col]
        dfm = dfm.sort_values(by=use_col, ascending=False).reset_index().rename(columns={"업종": "업종"})
        for r in dataframe_to_rows(dfm, index=False, header=True):
            ws.append(r)

        # Top20 Bar
        topN = min(20, len(dfm))
        start_col = dfm.shape[1] + 2
        ws.cell(row=1, column=start_col, value="업종")
        ws.cell(row=1, column=start_col + 1, value=f"{TARGET_YEAR} 예측")
        for i in range(topN):
            ws.cell(row=i + 2, column=start_col, value=dfm.loc[i, "업종"])
            ws.cell(row=i + 2, column=start_col + 1, value=float(dfm.loc[i, use_col] or 0))
        bar = BarChart(); bar.title = f"Top-20 {TARGET_YEAR} 예측"
        data = Reference(ws, min_col=start_col + 1, min_row=1, max_row=topN + 1)
        cats = Reference(ws, min_col=start_col,     min_row=2, max_row=topN + 1)
        bar.add_data(data, titles_from_data=True); bar.set_categories(cats)
        bar.y_axis.number_format = '#,##0'
        ws.add_chart(bar, ws.cell(row=2, column=start_col + 3).coordinate)

        # 연도별 총합 Line
        line_anchor_col = start_col + 6
        ws.cell(row=1, column=line_anchor_col,     value="연도")
        ws.cell(row=1, column=line_anchor_col + 1, value="총합")
        for i, y in enumerate(TRAIN_YEARS, start=2):
            ws.cell(row=i, column=line_anchor_col,     value=y)
            ws.cell(row=i, column=line_anchor_col + 1, value=float(pv[y].sum()))
        ws.cell(row=len(TRAIN_YEARS) + 2, column=line_anchor_col,     value=TARGET_YEAR)
        ws.cell(row=len(TRAIN_YEARS) + 2, column=line_anchor_col + 1, value=float(wide[use_col].sum()))
        lchart = LineChart(); lchart.title = "연도별 총합(실적 + 예측)"
        d = Reference(ws, min_col=line_anchor_col + 1, min_row=1, max_row=len(TRAIN_YEARS) + 2)
        c = Reference(ws, min_col=line_anchor_col,     min_row=2, max_row=len(TRAIN_YEARS) + 2)
        lchart.add_data(d, titles_from_data=True); lchart.set_categories(c)
        lchart.y_axis.number_format = '#,##0'
        ws.add_chart(lchart, ws.cell(row=2, column=line_anchor_col + 3).coordinate)

    _add_method_sheet("선형추세(OLS)", "선형추세(OLS)")
    _add_method_sheet("CAGR", "CAGR")
    holt_real = next((c for c in wide.columns if c.startswith("Holt")), None)
    _add_method_sheet("Holt(지수평활)", holt_real if holt_real else "Holt(지수평활)")

    bio = BytesIO(); wb.save(bio)
    st.download_button("엑셀(xlsx) 다운로드 (다중 시트+차트 포함)",
                       bio.getvalue(),
                       file_name=f"industry_forecast_{TARGET_YEAR}_{fname_suffix}.xlsx",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    st.download_button(
        "업종별 예측표 CSV 다운로드",
        out_all.to_csv(index=False).encode("utf-8-sig"),
        file_name=f"industry_forecast_{TARGET_YEAR}_{fname_suffix}.csv",
        mime="text/csv"
    )
