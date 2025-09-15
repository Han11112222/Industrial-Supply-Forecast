# app.py — 산업용 공급량 예측(추세분석)
# • 좌측: 학습 연도(멀티), 예측 구간(시작연~종료연, 월 제외)
# • 예측: OLS / CAGR / Holt(지수평활) — 다년 예측
# • 결과 유지: 예측 시작 후 session_state에 고정 → 라디오 변경 시 화면 유지
# • 그래프: 총합(실적+예측포인트), Top-10 막대(연도 선택), Top-10 실적추이(예측연도까지 연장)
# • 다운로드: 전체표 + 방법별 시트(Top-20 막대, 연도별 총합 라인)

from pathlib import Path
from io import BytesIO
import re
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

# Holt
try:
    from statsmodels.tsa.holtwinters import Holt
except Exception:
    Holt = None

# openpyxl
from openpyxl import Workbook
from openpyxl.utils.dataframe import dataframe_to_rows
from openpyxl.chart import BarChart, Reference, LineChart

# ───────────────────────── 기본 UI ─────────────────────────
st.set_page_config(page_title="산업용 공급량 예측(추세분석)", layout="wide")
st.title("🏭📈 산업용 공급량 예측(추세분석)")
st.caption("RAW 엑셀 업로드 → 학습연도 선택 → 추세 예측(3종) → 정렬/총합/그래프/다운로드")

# 쉬운 요약(툴팁식)
with st.expander("📚 예측 방법 설명(쉽게 보기)", expanded=False):
    st.markdown(
        "- **선형추세(OLS)**: 해마다 얼마나 늘었는지 ‘직선’으로 맞춰서 앞으로를 그려본다.\n"
        "- **CAGR**: 시작~끝 사이의 연평균 복리성장률로 앞으로를 늘린다(중장기 성장 가정).\n"
        "- **Holt(지수평활)**: 최근 흐름(수준+추세)을 더 반영해 부드럽게 연장한다(계절성 제외).\n"
        "→ 서로 다른 가정이니 **비교해서** 쓰면 안전해."
    )

# ───────────────────────── 사이드바 ─────────────────────────
with st.sidebar:
    st.header("📥 ① 데이터 불러오기")
    up = st.file_uploader("원본 엑셀(.xlsx)", type=["xlsx"])
    sample_path = Path("산업용_업종별.xlsx")
    use_sample = st.checkbox(f"Repo 파일 사용: {sample_path.name}", value=sample_path.exists())

    st.divider()
    st.header("🧪 ② 예측 방법")
    METHOD_CHOICES = ["선형추세(OLS)", "CAGR", "Holt(지수평활)"]
    methods = st.multiselect("방법 선택(정렬 기준은 첫 번째)", METHOD_CHOICES, default=METHOD_CHOICES)

# ───────────────────────── 유틸 ─────────────────────────
@st.cache_data(show_spinner=False)
def read_excel_to_long(file) -> pd.DataFrame:
    """엑셀 → Long(업종, 연도, 사용량). 4자리 연도 헤더만 인식."""
    df = pd.read_excel(file, engine="openpyxl")

    year_cols = [c for c in df.columns if re.search(r"(?:19|20)\d{2}", str(c))]
    non_year_cols = [c for c in df.columns if c not in year_cols]
    obj_non_year = [c for c in non_year_cols if df[c].dtype == "object"]
    cat_col = obj_non_year[0] if obj_non_year else (non_year_cols[0] if non_year_cols else df.columns[0])
    cat_col = str(cat_col)

    if not year_cols:  # 연도머리글 없으면 숫자형 열 사용
        year_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and c != cat_col]

    m = df[[cat_col] + year_cols].copy().melt(id_vars=[cat_col], var_name="연도헤더", value_name="사용량")
    years = m["연도헤더"].astype(str).str.extract(r"((?:19|20)\d{2})")[0]
    m["연도"] = pd.to_numeric(years, errors="coerce").astype("Int64")
    m = m.dropna(subset=["연도"]).rename(columns={cat_col: "업종"})
    m["연도"] = m["연도"].astype(int)
    m["사용량"] = pd.to_numeric(m["사용량"], errors="coerce")
    m = m.dropna(subset=["업종", "사용량"])
    return m[["업종", "연도", "사용량"]]

def _ols(x_years, y_vals, targets):
    coef = np.polyfit(x_years, y_vals, 1)
    fitted = np.polyval(coef, x_years)
    preds = [float(np.polyval(coef, t)) for t in targets]
    return preds, fitted

def _cagr(x_years, y_vals, targets):
    y0, yT = float(y_vals[0]), float(y_vals[-1]); n = int(x_years[-1] - x_years[0])
    if y0 <= 0 or yT <= 0 or n <= 0:
        return _ols(x_years, y_vals, targets)
    g = (yT / y0) ** (1.0 / n) - 1.0
    last = x_years[-1]
    preds = [float(yT * (1.0 + g) ** (t - last)) for t in targets]
    return preds, np.array(y_vals, dtype=float)

def _holt(y_vals, last_train_year, targets):
    x_years = list(range(last_train_year - len(y_vals) + 1, last_train_year + 1))
    if Holt is None or len(y_vals) < 2 or any(t <= last_train_year for t in targets):
        return _ols(x_years, y_vals, targets)
    fit = Holt(np.asarray(y_vals), exponential=False, damped_trend=False,
               initialization_method="estimated").fit(optimized=True)
    max_h = max(t - last_train_year for t in targets)
    fc = fit.forecast(max_h)
    preds = [float(fc[h - 1]) for h in [t - last_train_year for t in targets]]
    return preds, np.array(fit.fittedvalues, dtype=float)

def fmt_int(x):
    if pd.isna(x): return ""
    try: return f"{int(round(float(x))):,}"
    except Exception: return x

# ───────────────────────── 데이터 준비 ─────────────────────────
df_long_ui = None
if up is not None:
    df_long_ui = read_excel_to_long(up)
elif use_sample and sample_path.exists():
    df_long_ui = read_excel_to_long(sample_path)

# UI: 학습/예측 기간
TRAIN_YEARS = []
FORECAST_YEARS = []
run_clicked = False
if df_long_ui is not None:
    years = sorted(df_long_ui["연도"].unique().tolist())
    default_train = years[-5:] if len(years) >= 5 else years

    with st.sidebar:
        st.divider()
        st.header("🗓️ ③ 학습/예측 기간")
        TRAIN_YEARS = st.multiselect("학습 연도", years, default=default_train)
        TRAIN_YEARS = sorted(TRAIN_YEARS) if TRAIN_YEARS else []
        future = list(range(years[-1], years[-1] + 10))
        yr_opts = sorted(set(years + future))
        start_y = st.selectbox("예측 시작(연)", yr_opts, index=yr_opts.index(years[-1]))
        end_y   = st.selectbox("예측 종료(연)", yr_opts, index=min(len(yr_opts)-1, yr_opts.index(years[-1])+1))
        FORECAST_YEARS = list(range(min(start_y, end_y), max(start_y, end_y) + 1))

        st.divider()
        c1, c2 = st.columns(2)
        with c1:
            run_clicked = st.button("🚀 예측 시작", use_container_width=True)
        with c2:
            reset_clicked = st.button("🔄 다시 계산", use_container_width=True)

else:
    st.info("좌측에서 엑셀을 업로드하거나 ‘Repo 파일 사용’을 체크해줘.")

# ───────────────────────── 상태 유지(세션) ─────────────────────────
# 세션 초기화
if "started" not in st.session_state:
    st.session_state.started = False
if "store" not in st.session_state:
    st.session_state.store = {}

def compute_and_store():
    df_long = df_long_ui.copy()
    pv_all = df_long.pivot_table(index="업종", columns="연도", values="사용량", aggfunc="sum")
    missing = [y for y in TRAIN_YEARS if y not in pv_all.columns]
    if missing:
        st.error(f"학습 연도 데이터 없음: {missing}")
        st.stop()
    pv = pv_all.reindex(columns=TRAIN_YEARS).fillna(0)

    # 예측
    result = pv.copy()
    result.columns = [f"{c} 실적" for c in result.columns]

    for ind, row in pv.iterrows():
        y = row.values.astype(float).tolist()
        x = TRAIN_YEARS
        last = x[-1]
        for m in methods:
            if m == "선형추세(OLS)":
                preds, _ = _ols(x, y, FORECAST_YEARS)
            elif m == "CAGR":
                preds, _ = _cagr(x, y, FORECAST_YEARS)
            else:
                preds, _ = _holt(y, last, FORECAST_YEARS)
            for yy, p in zip(FORECAST_YEARS, preds):
                col = f"{m}({yy})"
                if col not in result.columns: result[col] = np.nan
                result.loc[ind, col] = p

    sort_method = methods[0]
    sort_col = f"{sort_method}({FORECAST_YEARS[-1]})"
    if sort_col not in result.columns:
        alt_cols = [c for c in result.columns if c.endswith(f"({FORECAST_YEARS[-1]})")]
        sort_col = alt_cols[0] if alt_cols else result.columns[-1]
    final_sorted = result.sort_values(by=sort_col, ascending=False)
    final_sorted.index.name = "업종"

    totals = final_sorted.sum(axis=0, numeric_only=True)
    total_row = pd.DataFrame([totals.to_dict()], index=["총합"])
    final_with_total = pd.concat([final_sorted, total_row], axis=0)

    # 저장
    st.session_state.store = dict(
        pv=pv, final=final_sorted, final_total=final_with_total,
        train_years=TRAIN_YEARS, fc_years=FORECAST_YEARS, methods=methods
    )
    st.session_state.started = True

# 트리거
if run_clicked and df_long_ui is not None and methods and TRAIN_YEARS and FORECAST_YEARS:
    compute_and_store()
elif reset_clicked and st.session_state.started:
    compute_and_store()

# ───────────────────────── 결과 표시 ─────────────────────────
if st.session_state.started:
    pv = st.session_state.store["pv"]
    final_sorted = st.session_state.store["final"]
    final_total  = st.session_state.store["final_total"]
    TRAIN_YEARS  = st.session_state.store["train_years"]
    FORECAST_YEARS = st.session_state.store["fc_years"]
    methods = st.session_state.store["methods"]

    st.success(f"로드 완료: 업종 {pv.shape[0]}개, 학습 {TRAIN_YEARS[0]}–{TRAIN_YEARS[-1]}, 예측 {FORECAST_YEARS[0]}–{FORECAST_YEARS[-1]}")

    # 표
    st.subheader("🧾 업종별 예측 표")
    disp = final_total.copy()
    disp.insert(0, "업종", disp.index)
    for c in disp.columns[1:]: disp[c] = disp[c].apply(fmt_int)
    st.dataframe(disp.reset_index(drop=True), use_container_width=True)

    # 총합 그래프 (실적+예측 포인트)
    st.subheader("📈 연도별 총합(실적 라인 + 예측 포인트)")
    tot_actual = pv.sum(axis=0).reset_index()
    tot_actual.columns = ["연도", "합계"]
    pts = []
    for m in methods:
        for yy in FORECAST_YEARS:
            col = f"{m}({yy})"
            if col in final_sorted.columns:
                pts.append({"연도": yy, "방법": m, "값": float(final_sorted[col].sum())})
    pts_df = pd.DataFrame(pts)

    area = alt.Chart(tot_actual).mark_area(opacity=0.25).encode(
        x=alt.X("연도:O", title="연도"),
        y=alt.Y("합계:Q", title="총합(실적)", axis=alt.Axis(format=",")),
        tooltip=[alt.Tooltip("연도:O"), alt.Tooltip("합계:Q", format=",")]
    )
    line = alt.Chart(tot_actual).mark_line(size=3).encode(x="연도:O", y=alt.Y("합계:Q", axis=alt.Axis(format=",")))
    if not pts_df.empty:
        ptsch = alt.Chart(pts_df).mark_point(size=150, filled=True).encode(
            x="연도:O", y=alt.Y("값:Q", axis=alt.Axis(format=",")),
            color=alt.Color("방법:N", legend=alt.Legend(title="방법")),
            shape=alt.Shape("방법:N"),
            tooltip=[alt.Tooltip("방법:N"), alt.Tooltip("연도:O"), alt.Tooltip("값:Q", format=",")]
        )
        labels = ptsch.mark_text(dy=-12, fontWeight="bold").encode(text=alt.Text("값:Q", format=","))
        st.altair_chart((area + line + ptsch + labels).interactive(), use_container_width=True, theme="streamlit")
    else:
        st.altair_chart((area + line).interactive(), use_container_width=True, theme="streamlit")

    # Top-10 — 막대/라인
    st.subheader("🏆 상위 10개 업종 — 예측 비교 / 실적 추이")
    yy_pick = st.radio("막대그래프 기준 예측연도", FORECAST_YEARS, index=len(FORECAST_YEARS)-1, horizontal=True, key="yy_pick")

    # 막대(연도별 방법 비교)
    method_cols = [f"{m}({yy_pick})" for m in methods if f"{m}({yy_pick})" in final_sorted.columns]
    if method_cols:
        top10 = final_sorted.head(10).index.tolist()
        bar_base = final_sorted.loc[top10, method_cols].copy()
        bar_base.index.name = "업종"
        pred_long = bar_base.reset_index().melt(id_vars="업종", var_name="방법연도", value_name="예측")
        pred_long["방법"] = pred_long["방법연도"].str.replace(r"\(\d{4}\)$", "", regex=True)

        sel = alt.selection_point(fields=["방법"], bind="legend")
        bars = alt.Chart(pred_long).mark_bar().encode(
            x=alt.X("업종:N", sort=top10, title=None),
            xOffset=alt.XOffset("방법:N"),
            y=alt.Y("예측:Q", axis=alt.Axis(format=","), title=f"{yy_pick} 예측"),
            color=alt.Color("방법:N", legend=alt.Legend(title="방법")),
            opacity=alt.condition(sel, alt.value(1.0), alt.value(0.25)),
            tooltip=[alt.Tooltip("업종:N"), alt.Tooltip("방법:N"), alt.Tooltip("예측:Q", format=",")]
        ).add_params(sel).properties(height=420)
        bar_txt = bars.mark_text(dy=-5, fontSize=10).encode(text=alt.Text("예측:Q", format=","))

        # 라인(실적 + 선택방법 예측연장)
        st.markdown("※ 라인 그래프는 **첫 번째로 선택한 방법**으로 2026·2027을 이어서 보여줘.")
        method_for_line = methods[0]
        pred_cols_for_line = [f"{method_for_line}({yy})" for yy in FORECAST_YEARS if f"{method_for_line}({yy})" in final_sorted.columns]

        actual_long = pv.loc[top10, TRAIN_YEARS].reset_index().melt(id_vars="업종", var_name="연도", value_name="값")
        actual_long["출처"] = "실적"

        pred_line = pd.DataFrame()
        if pred_cols_for_line:
            pred_line = final_sorted.loc[top10, pred_cols_for_line].copy()
            # 열명에서 연도 뽑기
            pred_line.columns = [int(re.search(r"\((\d{4})\)", c).group(1)) for c in pred_line.columns]
            pred_line = pred_line.reset_index().melt(id_vars="업종", var_name="연도", value_name="값")
            pred_line["출처"] = f"예측({method_for_line})"

        line_df = pd.concat([actual_long, pred_line], ignore_index=True)
        # 연도 순서 보장
        year_order = TRAIN_YEARS + [y for y in FORECAST_YEARS if y not in TRAIN_YEARS]
        line_df["연도"] = line_df["연도"].astype(str)  # Altair O-ordinal
        line_df["연도"] = pd.Categorical(line_df["연도"], categories=[str(y) for y in year_order], ordered=True)

        sel2 = alt.selection_point(fields=["업종"], bind="legend")
        lines = alt.Chart(line_df).mark_line(point=True, strokeWidth=3).encode(
            x=alt.X("연도:O", title=None),
            y=alt.Y("값:Q", axis=alt.Axis(format=",")),
            color=alt.Color("업종:N", sort=top10, legend=alt.Legend(title="업종(클릭으로 강조)")),
            opacity=alt.condition(sel2, alt.value(1.0), alt.value(0.25)),
            tooltip=[alt.Tooltip("업종:N"), alt.Tooltip("연도:O"), alt.Tooltip("값:Q", format=","), alt.Tooltip("출처:N")]
        ).add_params(sel2).properties(height=420)

        c1, c2 = st.columns(2)
        with c1:
            st.altair_chart((bars + bar_txt).interactive(), use_container_width=True, theme="streamlit")
        with c2:
            st.altair_chart(lines.interactive(), use_container_width=True, theme="streamlit")
    else:
        st.info(f"{yy_pick}년 예측 열이 없어서 막대그래프는 건너뛰었어.")

    # 다운로드
    st.subheader("💾 다운로드")
    out_all = final_total.copy()
    out_all.insert(0, "업종", out_all.index)
    fname = f"industry_forecast_{FORECAST_YEARS[0]}-{FORECAST_YEARS[-1]}.xlsx"

    wb = Workbook(); wb.remove(wb.active)
    # 시트1: 전체
    ws_all = wb.create_sheet("전체")
    for r in dataframe_to_rows(out_all, index=False, header=True): ws_all.append(r)

    # 방법별 시트
    def add_method_sheet(mth):
        ws = wb.create_sheet(mth)
        dfm = pv.copy(); dfm.columns = [f"{c} 실적" for c in dfm.columns]
        pred_cols = [f"{mth}({yy})" for yy in FORECAST_YEARS if f"{mth}({yy})" in final_sorted.columns]
        for c in pred_cols: dfm[c] = final_sorted[c]
        order_col = pred_cols[-1] if pred_cols else dfm.columns[-1]
        dfm = dfm.sort_values(by=order_col, ascending=False).reset_index().rename(columns={"index":"업종"})
        for r in dataframe_to_rows(dfm, index=False, header=True): ws.append(r)

        # Top-20 막대(종료연도)
        if pred_cols:
            topN = min(20, len(dfm))
            y = FORECAST_YEARS[-1]; use_col = f"{mth}({y})"
            sc = dfm.shape[1] + 2
            ws.cell(row=1, column=sc, value="업종")
            ws.cell(row=1, column=sc+1, value=f"{y} 예측")
            for i in range(topN):
                ws.cell(row=i+2, column=sc, value=dfm.loc[i, "업종"])
                ws.cell(row=i+2, column=sc+1, value=float(dfm.loc[i, use_col] or 0))
            bar = BarChart(); bar.title = f"Top-20 {y} ({mth})"
            data = Reference(ws, min_col=sc+1, min_row=1, max_row=topN+1)
            cats = Reference(ws, min_col=sc,   min_row=2, max_row=topN+1)
            bar.add_data(data, titles_from_data=True); bar.set_categories(cats)
            bar.y_axis.number_format = '#,##0'
            ws.add_chart(bar, ws.cell(row=2, column=sc+3).coordinate)

        # 연도별 총합 라인
        la = dfm.shape[1] + 8
        ws.cell(row=1, column=la,   value="연도")
        ws.cell(row=1, column=la+1, value="총합")
        for i, y in enumerate(TRAIN_YEARS, start=2):
            ws.cell(row=i, column=la,   value=y)
            ws.cell(row=i, column=la+1, value=float(pv[y].sum()))
        base = len(TRAIN_YEARS) + 2
        for j, y in enumerate(FORECAST_YEARS):
            ws.cell(row=base+j, column=la,   value=y)
            col = f"{mth}({y})"
            tot = float(final_sorted[col].sum()) if col in final_sorted.columns else 0.0
            ws.cell(row=base+j, column=la+1, value=tot)
        lch = LineChart(); lch.title = f"연도별 총합(실적+예측, {mth})"
        mr = base + len(FORECAST_YEARS) - 1
        d = Reference(ws, min_col=la+1, min_row=1, max_row=mr)
        c = Reference(ws, min_col=la,   min_row=2, max_row=mr)
        lch.add_data(d, titles_from_data=True); lch.set_categories(c)
        lch.y_axis.number_format = '#,##0'
        ws.add_chart(lch, ws.cell(row=2, column=la+3).coordinate)

    for m in methods: add_method_sheet(m)

    bio = BytesIO(); wb.save(bio)
    st.download_button("엑셀(xlsx) 다운로드", bio.getvalue(), file_name=fname,
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    st.download_button(
        "업종별 예측표 CSV 다운로드",
        out_all.to_csv(index=False).encode("utf-8-sig"),
        file_name=fname.replace(".xlsx",".csv"),
        mime="text/csv"
    )
