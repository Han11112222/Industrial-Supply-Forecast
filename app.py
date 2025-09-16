# app.py — 산업용 공급량 예측(추세분석)
# • 데이터: 연도별 엑셀 여러 개 업로드(또는 Repo의 산업용_*.xlsx 자동 로딩)
# • 전처리: '상품명' == '산업용'(정확일치)만 사용, 집계는 '업종' 기준으로 '판매량' 합계 → (업종, 연도, 사용량)
# • 월→연: 2025년이 8월까지만 있으면 2025-09~12를 월별 시계열(Holt/SES)로 추정해 연간 2025 보정
# • 좌측: 학습 연도(멀티, 2020 포함), 예측 구간(시작연~종료연, 월 제외)
# • 예측: OLS / CAGR / Holt / SES — 다년 예측
# • 결과 유지: session_state 저장(라디오/선택 변경에도 유지)
# • 그래프: 총합(실적+예측포인트), Top-10 막대(연도 선택), Top-10 실적추이(예측연도 연장)
# • 다운로드: 전체표 + 방법별 시트(Top-20 막대, 연도별 총합 라인)

from pathlib import Path
from io import BytesIO
import re
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

# statsmodels (Holt, SES)
try:
    from statsmodels.tsa.holtwinters import Holt, SimpleExpSmoothing
except Exception:
    Holt = None
    SimpleExpSmoothing = None

# openpyxl (엑셀 차트)
from openpyxl import Workbook
from openpyxl.utils.dataframe import dataframe_to_rows
from openpyxl.chart import BarChart, Reference, LineChart

# ───────────────────────── 기본 UI ─────────────────────────
st.set_page_config(page_title="산업용 공급량 예측(추세분석)", layout="wide")
st.title("🏭📈 산업용 공급량 예측(추세분석)")
st.caption("여러 연도 파일 → ‘산업용’만 필터 → 업종·연도 집계 → 4가지 추세 예측(월 보정 포함)")

# 방법 설명(간단 산식)
with st.expander("예측 방법 설명", expanded=False):
    st.markdown(
        """
- **선형추세(OLS)**: `y_t = a + b t`, 예측 `ŷ_{T+h} = a + b (T+h)`
- **CAGR(복리)**: `g = (y_T/y_0)^{1/n} - 1`, 예측 `ŷ_{T+h} = y_T (1+g)^h`
- **Holt(지수평활·추세형)**: `ŷ_{T+h} = l_T + h b_T` (계절성 제외)
- **SES(지수평활)**: `ŷ_{T+h} = l_T` (추세·계절성 제외)

*2025년 월데이터가 8월까지만 있을 때는 2020-01~2025-08 월시계열로 9~12월을 보정한 뒤 2025 연간을 계산합니다.*
"""
    )

# ───────────────────────── 사이드바 ─────────────────────────
with st.sidebar:
    st.header("📥 ① 데이터 불러오기")
    ups = st.file_uploader("연도별 엑셀(.xlsx) 여러 개 업로드", type=["xlsx"], accept_multiple_files=True)

    st.caption("또는 Repo의 **산업용_*.xlsx** 를 자동 읽기")
    repo_files = sorted([p for p in Path(".").glob("산업용_*.xlsx")])
    use_repo = st.checkbox(f"Repo 자동 읽기 ({len(repo_files)}개 감지)", value=bool(repo_files))
    if repo_files:
        st.write("읽을 대상:", "\n\n".join([f"- {p.name}" for p in repo_files]))

    st.divider()
    st.header("🧪 ② 예측 방법")
    METHOD_CHOICES = ["선형추세(OLS)", "CAGR(복리성장)", "Holt(지수평활)", "지수평활(SES)"]
    methods = st.multiselect("방법 선택(정렬 기준은 첫 번째)", METHOD_CHOICES, default=METHOD_CHOICES)

# ───────────────────────── 유틸 ─────────────────────────
def _clean_col(s: str) -> str:
    return re.sub(r"\s+", "", str(s)).strip()

def _extract_year_from_filename(name: str) -> int | None:
    m = re.search(r"(19|20)(\d{2})(\d{2})?$", name.replace(".xlsx",""))
    if m:
        return int(m.group(1)+m.group(2))
    return None

def _parse_ym(s: pd.Series) -> pd.Series:
    """판매년월(예: Jan-25 / 2025-01 등) → pandas Period('M')"""
    x = s.astype(str).str.strip()
    d = pd.to_datetime(x, errors="coerce", infer_datetime_format=True)
    if d.isna().all():
        d = pd.to_datetime(x, format="%b-%y", errors="coerce")  # Jan-25
    # 2자리 연도만 있을 가능성은 위에서 처리됨
    return d.dt.to_period("M")

def _coerce_num(s):
    return pd.to_numeric(s, errors="coerce")

def _safe_sum(x):
    try:
        return float(np.nansum(x))
    except Exception:
        return 0.0

@st.cache_data(show_spinner=False)
def load_monthly(files, repo_use: bool) -> pd.DataFrame:
    """여러 엑셀에서 '산업용' 정확일치만, 월단위 집계 (업종/업종분류/연월/연도/월/사용량)."""
    targets: list[tuple[str, BytesIO | Path]] = []
    if files:
        for f in files:
            targets.append((f.name, f))
    elif repo_use:
        for p in sorted([p for p in Path(".").glob("산업용_*.xlsx")]):
            targets.append((p.name, p))
    if not targets:
        return pd.DataFrame(columns=["업종","업종분류","연월","연도","월","사용량"])

    out = []
    for name, src in targets:
        df = pd.read_excel(src, engine="openpyxl")
        df.columns = [_clean_col(c) for c in df.columns]

        # 필수 열 체크
        need = {"상품명","업종","판매량"}
        if not need.issubset(set(df.columns)):
            continue
        # 선택 열
        col_div = "업종분류" if "업종분류" in df.columns else None
        col_ym  = None
        for c in ("판매년월","년월","월","연도","년도"):
            if c in df.columns:
                col_ym = c; break

        # '산업용' 정확일치
        item_norm = df["상품명"].astype(str).str.replace(r"\s+","", regex=True)
        d = df.loc[item_norm == "산업용", ["업종","판매량"] + ([col_div] if col_div else []) + ([col_ym] if col_ym else [])].copy()
        if d.empty:
            continue

        d["판매량"] = _coerce_num(d["판매량"])

        # 연월 파싱
        if col_ym:
            ym = _parse_ym(d[col_ym])
        else:
            # 파일명에서 연도를 추정하여 1~12월 전부 동일 분배는 위험하므로 스킵
            # (월 정보 없는 파일은 사용하지 않음)
            continue

        d["연월"] = ym
        d = d.dropna(subset=["연월","판매량","업종"])
        d["연월"] = d["연월"].astype("period[M]")
        d["연도"] = d["연월"].dt.year.astype(int)
        d["월"]   = d["연월"].dt.month.astype(int)
        if col_div:
            d.rename(columns={col_div:"업종분류"}, inplace=True)
        else:
            d["업종분류"] = "분류없음"

        out.append(d[["업종","업종분류","연월","연도","월","판매량"]].rename(columns={"판매량":"사용량"}))

    if not out:
        return pd.DataFrame(columns=["업종","업종분류","연월","연도","월","사용량"])

    mdf = pd.concat(out, ignore_index=True)
    # 동월 중복 합치기
    mdf = (mdf.groupby(["업종","업종분류","연월","연도","월"], as_index=False)["사용량"]
              .sum().sort_values(["연월","업종"]))
    return mdf

def _holt_monthly(y: np.ndarray, steps: int) -> np.ndarray:
    """월시계열 보정: Holt(damped) → SES → 마지막 12개월 평균."""
    steps = int(steps)
    if steps <= 0:
        return np.array([])
    y = np.asarray(y, dtype=float)
    y = np.nan_to_num(y, nan=0.0)
    if len(y) < 3:
        return np.full(steps, y[-1] if len(y) else 0.0)
    if Holt is not None:
        try:
            fit = Holt(y, damped_trend=True, initialization_method="estimated").fit(optimized=True)
            return np.maximum(fit.forecast(steps), 0.0)
        except Exception:
            pass
    if SimpleExpSmoothing is not None:
        try:
            fit = SimpleExpSmoothing(y).fit(optimized=True)
            return np.maximum(fit.forecast(steps), 0.0)
        except Exception:
            pass
    # fallback: 최근 12개월 평균
    base = y[-min(12, len(y)):]
    return np.full(steps, np.maximum(base.mean() if len(base) else 0.0, 0.0))

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
    """연간 Holt wrapper — steps가 0이면 안전하게 OLS로 대체(이전 오류 방지)."""
    steps = [t - last_train_year for t in targets if t > last_train_year]
    max_h = max(steps) if steps else 0
    x_years = list(range(last_train_year - len(y_vals) + 1, last_train_year + 1))
    if Holt is None or len(y_vals) < 2 or max_h <= 0:
        return _ols(x_years, y_vals, targets)
    fit = Holt(np.asarray(y_vals), exponential=False, damped_trend=False,
               initialization_method="estimated").fit(optimized=True)
    fc = fit.forecast(max_h)
    preds_map = {last_train_year + h: float(fc[h - 1]) for h in range(1, max_h + 1)}
    preds = [preds_map.get(t, float(np.polyval(np.polyfit(x_years, y_vals, 1), t))) for t in targets]
    return preds, np.array(fit.fittedvalues, dtype=float)

def _ses(y_vals, last_train_year, targets):
    steps = [t - last_train_year for t in targets if t > last_train_year]
    max_h = max(steps) if steps else 0
    x_years = list(range(last_train_year - len(y_vals) + 1, last_train_year + 1))
    if SimpleExpSmoothing is None or len(y_vals) < 2 or max_h <= 0:
        return _ols(x_years, y_vals, targets)
    fit = SimpleExpSmoothing(np.asarray(y_vals)).fit(optimized=True)
    fc = fit.forecast(max_h)
    preds_map = {last_train_year + h: float(fc[h - 1]) for h in range(1, max_h + 1)}
    preds = [preds_map.get(t, float(np.polyval(np.polyfit(x_years, y_vals, 1), t))) for t in targets]
    return preds, np.array(fit.fittedvalues, dtype=float)

def fmt_int(x):
    if pd.isna(x): return ""
    try: return f"{int(round(float(x))):,}"
    except Exception: return x

# ───────────────────────── 데이터 준비 ─────────────────────────
mdf_all = load_monthly(ups, use_repo)

if mdf_all.empty:
    st.info("좌측에서 연도별 엑셀을 올리거나 ‘Repo 자동 읽기’를 켜줘.")
    st.stop()

# 업종분류 필터 (요청 순서 고정)
CATEGORY_ORDER = ["전체","제조업","기타영업용","단독주택","숙박업","음식점업","일반빌딩"]
avail = ["전체"] + [c for c in CATEGORY_ORDER[1:] if (mdf_all["업종분류"] == c).any()]
sel_cat = st.radio("업종분류 선택", avail, index=0, horizontal=True)

mdf = mdf_all.copy()
if sel_cat != "전체":
    mdf = mdf[mdf["업종분류"] == sel_cat]

min_y, max_y = int(mdf["연도"].min()), int(mdf["연도"].max())
latest_y = 2025 if 2025 in mdf["연도"].unique() else max_y
max_month_2025 = int(mdf.loc[mdf["연도"]==2025, "월"].max()) if 2025 in mdf["연도"].unique() else None

# 2025 연간 보정(9~12월 추정)
def make_annual_with_2025_nowcast(mdf: pd.DataFrame) -> pd.DataFrame:
    # 업종×연도 실적(2020~2024) + 2025(보정)
    ann = (mdf.groupby(["업종","연도"], as_index=False)["사용량"].sum())
    if 2025 in ann["연도"].unique():
        # 2025 부분실적
        last_m = int(mdf.loc[mdf["연도"]==2025,"월"].max())
        if last_m < 12:
            add_rows = []
            # 업종별 월시계열 생성 후 9~12월 보정
            for ind, grp in mdf.groupby("업종"):
                # 2020-01 ~ 2025-last_m 까지 월시계열
                idx = pd.period_range("2020-01", f"2025-{last_m:02d}", freq="M")
                s = (grp.set_index("연월")["사용량"]
                       .reindex(idx, fill_value=0.0)
                       .astype(float).values)
                steps = 12 - last_m
                preds = _holt_monthly(s, steps)  # 안전한 폴백 포함
                add_val = float(np.maximum(preds, 0.0).sum())
                base_2025 = _safe_sum(grp.loc[grp["연도"]==2025,"사용량"])
                total_2025 = base_2025 + add_val
                add_rows.append({"업종":ind,"연도":2025,"사용량":total_2025})
            ann_wo_2025 = ann[ann["연도"] != 2025]
            ann_2025 = pd.DataFrame(add_rows)
            ann = pd.concat([ann_wo_2025, ann_2025], ignore_index=True)
    # 피벗
    pv_all = ann.pivot_table(index="업종", columns="연도", values="사용량", aggfunc="sum").fillna(0)
    return pv_all

pv_all = make_annual_with_2025_nowcast(mdf)

st.success(f"로드 완료: 업종 {pv_all.shape[0]:,}개, 연도 범위 {pv_all.columns.min()}–{pv_all.columns.max()} · "
           f"2025 월데이터 최대월: {max_month_2025 if max_month_2025 else '-'}")

# ───────────────────────── 학습/예측 기간 ─────────────────────────
TRAIN_YEARS = []
FORECAST_YEARS = []
run_clicked = False

years_list = sorted([int(c) for c in pv_all.columns.tolist()])
# 기본값: 2020을 포함하도록 고정
default_train = [y for y in years_list if y >= 2020]
with st.sidebar:
    st.divider()
    st.header("🗓️ ③ 학습/예측 기간")
    TRAIN_YEARS = st.multiselect("학습 연도", years_list, default=default_train)
    TRAIN_YEARS = sorted(TRAIN_YEARS) if TRAIN_YEARS else []
    future = list(range(years_list[-1], years_list[-1] + 10))
    yr_opts = sorted(set(years_list + future))
    start_y = st.selectbox("예측 시작(연)", yr_opts, index=yr_opts.index(years_list[-1]))
    end_y   = st.selectbox("예측 종료(연)", yr_opts, index=min(len(yr_opts)-1, yr_opts.index(years_list[-1])+1))
    FORECAST_YEARS = list(range(min(start_y, end_y), max(start_y, end_y) + 1))

    st.divider()
    run_clicked = st.button("🚀 예측 시작", use_container_width=True)

# ───────────────────────── 상태 유지(세션) ─────────────────────────
if "started" not in st.session_state:
    st.session_state.started = False
if "store" not in st.session_state:
    st.session_state.store = {}

def compute_and_store():
    missing = [y for y in TRAIN_YEARS if y not in pv_all.columns]
    if missing:
        st.error(f"학습 연도 데이터 없음: {missing}")
        st.stop()
    pv = pv_all.reindex(columns=TRAIN_YEARS).fillna(0)

    # 예측 결과 베이스
    result = pv.copy()
    result.columns = [f"{c} 실적" for c in result.columns]

    for ind, row in pv.iterrows():
        y = row.values.astype(float).tolist()
        x = TRAIN_YEARS
        last = x[-1]
        for m in methods:
            label = str(m)
            if "OLS" in label:
                preds, _ = _ols(x, y, FORECAST_YEARS)
            elif "CAGR" in label:
                preds, _ = _cagr(x, y, FORECAST_YEARS)
            elif "Holt" in label:
                preds, _ = _holt(y, last, FORECAST_YEARS)
            elif "SES" in label:
                preds, _ = _ses(y, last, FORECAST_YEARS)
            else:
                preds, _ = _ols(x, y, FORECAST_YEARS)

            for yy, p in zip(FORECAST_YEARS, preds):
                col = f"{label}({yy})"
                if col not in result.columns:
                    result[col] = np.nan
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

    st.session_state.store = dict(
        pv=pv, final=final_sorted, final_total=final_with_total,
        train_years=TRAIN_YEARS, fc_years=FORECAST_YEARS, methods=methods
    )
    st.session_state.started = True

if run_clicked and methods and TRAIN_YEARS and FORECAST_YEARS:
    compute_and_store()

# ───────────────────────── 결과 표시 ─────────────────────────
if st.session_state.started:
    pv = st.session_state.store["pv"]
    final_sorted = st.session_state.store["final"]
    final_total  = st.session_state.store["final_total"]
    TRAIN_YEARS  = st.session_state.store["train_years"]
    FORECAST_YEARS = st.session_state.store["fc_years"]
    methods = st.session_state.store["methods"]

    msg_2025 = ""
    if 2025 in TRAIN_YEARS and max_month_2025 and max_month_2025 < 12:
        msg_2025 = f" (※ 2025은 1–{max_month_2025}월 실적 + {max_month_2025+1}–12월 추정)"
    st.success(f"업종 {pv.shape[0]}개, 학습 {TRAIN_YEARS[0]}–{TRAIN_YEARS[-1]}{msg_2025}, 예측 {FORECAST_YEARS[0]}–{FORECAST_YEARS[-1]}")

    # 표
    st.subheader("🧾 업종별 예측 표")
    disp = final_total.copy()
    disp.insert(0, "업종", disp.index)
    for c in disp.columns[1:]:
        disp[c] = disp[c].apply(fmt_int)
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

        st.markdown("※ 라인은 **첫 번째 방법**으로 예측연도를 연장해 실적과 함께 표시")
        method_for_line = methods[0]
        pred_cols_for_line = [f"{method_for_line}({yy})" for yy in FORECAST_YEARS if f"{method_for_line}({yy})" in final_sorted.columns]

        actual_long = pv.loc[top10, TRAIN_YEARS].reset_index().melt(id_vars="업종", var_name="연도", value_name="값")
        actual_long["출처"] = "실적"

        pred_line = pd.DataFrame()
        if pred_cols_for_line:
            pred_line = final_sorted.loc[top10, pred_cols_for_line].copy()
            pred_line.columns = [int(re.search(r"\((\d{4})\)", c).group(1)) for c in pred_line.columns]
            pred_line = pred_line.reset_index().melt(id_vars="업종", var_name="연도", value_name="값")
            pred_line["출처"] = f"예측({method_for_line})"

        line_df = pd.concat([actual_long, pred_line], ignore_index=True)
        year_order = TRAIN_YEARS + [y for y in FORECAST_YEARS if y not in TRAIN_YEARS]
        line_df["연도"] = line_df["연도"].astype(str)
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
                v = dfm.loc[i, use_col]
                ws.cell(row=i+2, column=sc+1, value=float(v if pd.notna(v) else 0))
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
