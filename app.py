# app.py — 도시가스 산업용 업종별 2026 예측 (Streamlit)
# - 엑셀 업로드 또는 리포의 샘플 파일(쓰기 없음)
# - 예측 방법 4종: 선형추세(OLS), 다항추세(2/3차), CAGR, Holt(지수평활)
# - 업종별 예측표 + 연도별 총합 + 그래프 + CSV/XLSX 다운로드

from pathlib import Path
from io import BytesIO
import numpy as np
import pandas as pd
import streamlit as st

# Holt 지수평활
try:
    from statsmodels.tsa.holtwinters import Holt
except Exception:
    Holt = None

st.set_page_config(page_title="도시가스 산업용 업종별 2026 예측", layout="wide")
st.title("도시가스 공급량·판매량 예측 (업종별, 2026)")
st.caption("RAW 엑셀 업로드 → 업종별 추세 예측(4종) → 표/그래프/다운로드")

# ---------------- 사이드바 ----------------
with st.sidebar:
    st.header("① 데이터 불러오기")
    up = st.file_uploader("원본 엑셀 업로드 (.xlsx)", type=["xlsx"])
    # 리포에 올려둔 파일명을 기본 샘플로 사용 (필요시 이름만 바꿔줘)
    sample_path = Path("산업용_업종별.xlsx")
    use_sample = st.checkbox(f"샘플 파일 사용 ({sample_path.name})", value=sample_path.exists())

    st.divider()
    st.header("② 예측 설정")
    methods = st.multiselect(
        "예측 방법(복수 선택)",
        ["선형추세(OLS)", "다항추세(Poly)", "CAGR", "Holt(지수평활)"],
        default=["선형추세(OLS)", "다항추세(Poly)", "CAGR", "Holt(지수평활)"]
    )
    poly_deg = st.selectbox("다항차수(Poly)", [2, 3], index=1)
    run = st.button("예측 시작")

# ---------------- 상수 ----------------
YEARS = [2021, 2022, 2023, 2024, 2025]
TARGET_YEAR = 2026

# ---------------- 방법 설명 ----------------
METHOD_DESC = {
    "선형추세(OLS)": "연도(t)와 사용량(y)의 직선관계 y_t = a + b t 을 최소제곱으로 적합.",
    "다항추세(Poly)": "비선형을 반영하기 위해 t의 2~3차항을 포함해 적합(과적합 방지 위해 차수 제한).",
    "CAGR": "2021→2025 복리성장률 g로 2026 = y25 × (1+g) 로 1년 연장(시작/끝에 민감).",
    "Holt(지수평활)": "수준 l_t, 추세 b_t 를 지수 가중으로 갱신; 2026 = l_T + 1·b_T (계절성 미포함).",
}

# ---------------- 데이터 로딩 ----------------
@st.cache_data
def read_excel_to_long(file) -> pd.DataFrame:
    """엑셀을 (업종, 연도, 사용량) Long 형태로 변환(연도 파싱 튼튼)"""
    df = pd.read_excel(file, engine="openpyxl")
    # 업종 칼럼(첫 번째 object 타입) 탐지
    obj_cols = [c for c in df.columns if df[c].dtype == "object"]
    cat_col = obj_cols[0] if obj_cols else df.columns[0]
    # 연도 칼럼 자동 탐지
    year_cols, year_map = [], {}
    for c in df.columns:
        s = str(c)
        for y in YEARS:
            if str(y) in s:
                year_cols.append(c)
                year_map[c] = y
                break
    year_cols_sorted = [c for c in df.columns if c in year_map]
    m = df[[cat_col] + year_cols_sorted].copy().melt(
        id_vars=[cat_col], var_name="연도열", value_name="사용량"
    )
    # 더 관대한 연도 파서: 19xx/20xx 모두 허용, 매치 실패는 NaN
    years_extracted = m["연도열"].astype(str).str.extract(r"((?:19|20)\d{2})")[0]
    m["연도"] = pd.to_numeric(years_extracted, errors="coerce").astype("Int64")
    m = m[m["연도"].isin(YEARS)].copy()
    m["연도"] = m["연도"].astype(int)

    m = m.rename(columns={cat_col: "업종"})
    m["사용량"] = pd.to_numeric(m["사용량"], errors="coerce")
    m = m.dropna(subset=["업종", "연도", "사용량"])
    return m[["업종", "연도", "사용량"]]

# ---------------- 예측 함수 ----------------
def _linear_forecast(x, y, target):
    coef = np.polyfit(x, y, 1)
    return np.polyval(coef, target), np.polyval(coef, x)

def _poly_forecast(x, y, target, deg=3):
    deg = max(2, min(int(deg), len(x)-1))  # 표본수-1 이하로 제한
    coef = np.polyfit(x, y, deg)
    return np.polyval(coef, target), np.polyval(coef, x)

def _cagr_forecast(x, y, target):
    y2021, y2025 = float(y[0]), float(y[-1]); n = YEARS[-1]-YEARS[0]
    if y2021 <= 0 or y2025 <= 0 or n <= 0:
        # 시작/끝이 0·음수면 선형으로 대체
        yh, fit = _linear_forecast(x, y, target)
        return yh, np.array(y)
    g = (y2025/y2021)**(1.0/n) - 1.0
    return y2025 * (1.0 + g)**(target - YEARS[-1]), np.array(y)

def _holt_forecast(y, steps=1):
    if Holt is None:
        return None, None
    try:
        model = Holt(np.asarray(y), exponential=False, damped_trend=False, initialization_method="estimated")
        fit = model.fit(optimized=True)
        return float(fit.forecast(steps)[-1]), np.array(fit.fittedvalues)
    except Exception:
        return None, None

# ---------------- 실행 ----------------
df_long = None
if run:
    if up is not None:
        df_long = read_excel_to_long(up)
    elif use_sample and sample_path.exists():
        df_long = read_excel_to_long(sample_path)
    else:
        st.error("엑셀 업로드 또는 샘플 파일 사용을 선택해 주세요.")

if df_long is not None:
    st.success(f"로드 완료: 업종 {df_long['업종'].nunique()}개, {df_long.shape[0]}행")
    with st.expander("예측 방법 설명", expanded=False):
        for k, v in METHOD_DESC.items():
            st.markdown(f"**{k}** — {v}")

    # 피벗(업종 × 연도)
    pv = df_long.pivot_table(index="업종", columns="연도", values="사용량", aggfunc="sum").reindex(columns=YEARS)
    st.dataframe(pv.reset_index().head(10), use_container_width=True)

    # 업종별 예측
    out_rows = []
    fit_store = {}
    for industry, row in pv.fillna(0).iterrows():
        y = row.values.astype(float).tolist()
        x = YEARS
        for m in methods:
            if m == "선형추세(OLS)":
                yh, fitted = _linear_forecast(x, y, TARGET_YEAR)
                out_rows.append([industry, m, yh]); fit_store[(industry, m)] = (x, fitted)
            elif m == "다항추세(Poly)":
                yh, fitted = _poly_forecast(x, y, TARGET_YEAR, poly_deg)
                name = f"{m}({poly_deg}차)"
                out_rows.append([industry, name, yh]); fit_store[(industry, name)] = (x, fitted)
            elif m == "CAGR":
                yh, fitted = _cagr_forecast(x, y, TARGET_YEAR)
                out_rows.append([industry, m, yh]); fit_store[(industry, m)] = (x, fitted if fitted is not None else np.array(y))
            elif m == "Holt(지수평활)":
                yh, fitted = _holt_forecast(y, steps=1)
                name = m if yh is not None else "Holt(지수평활,대체:선형)"
                if yh is None:
                    yh, fitted = _linear_forecast(x, y, TARGET_YEAR)
                out_rows.append([industry, name, yh]); fit_store[(industry, name)] = (x, fitted if fitted is not None else np.array(y))

    forecast_df = pd.DataFrame(out_rows, columns=["업종", "방법", f"{TARGET_YEAR} 예측"])
    wide = forecast_df.pivot_table(index="업종", columns="방법", values=f"{TARGET_YEAR} 예측", aggfunc="first")

    # 최종표: 실적 + 방법별 2026 예측
    final_table = pv.copy()
    final_table.columns = [f"{c} 실적" for c in final_table.columns]
    final_table = final_table.join(wide)

    st.subheader("업종별 예측 표 (2026)")
    st.dataframe(final_table.reset_index(), use_container_width=True)

    # 연도별 총합
    total_actual = pv.sum(axis=0).rename(lambda y: f"{y} 실적").to_frame("값").reset_index().rename(columns={"index": "구분"})
    total_2026 = pd.DataFrame({"방법": list(wide.columns),
                               "구분": "총합",
                               f"{TARGET_YEAR} 합계": [wide[c].sum() for c in wide.columns]})
    st.subheader("연도별 총합")
    colA, colB = st.columns([2, 1])
    with colA: st.dataframe(total_actual, use_container_width=True)
    with colB: st.dataframe(total_2026, use_container_width=True)

    # 업종/합계별 2026 예측값 요약 표
    st.subheader("방법별 2026 예측값")
    industries = ["(전체 합계)"] + list(pv.index)
    sel_ind = st.selectbox("업종 선택", industries, index=0)

    if sel_ind == "(전체 합계)":
        _sum = pd.DataFrame({"방법": list(wide.columns),
                             "2026 합계": [wide[c].sum() for c in wide.columns]}).round(2)
        st.table(_sum)
    else:
        _sel = forecast_df[forecast_df["업종"] == sel_ind][["방법", f"{TARGET_YEAR} 예측"]].round(2)
        st.table(_sel)

    # 그래프
    import altair as alt

    def build_series(industry):
        frames = []
        if industry == "(전체 합계)":
            base = pd.DataFrame({"연도": YEARS, "값": pv.sum(axis=0).values, "시리즈": "실적(합계)"})
            frames.append(base)
            for mcol in wide.columns:
                frames.append(pd.DataFrame({"연도": [TARGET_YEAR], "값": [wide[mcol].sum()], "시리즈": f"예측:{mcol}"}))
        else:
            series_actual = pv.loc[industry, YEARS]
            frames.append(pd.DataFrame({"연도": YEARS, "값": series_actual.values, "시리즈": f"실적:{industry}"}))
            for (ind, mname), (xs, fitted) in fit_store.items():
                if ind == industry:
                    frames.append(pd.DataFrame({"연도": xs, "값": fitted, "시리즈": f"적합:{mname}"}))
            rowf = forecast_df[forecast_df["업종"] == industry]
            for _, r in rowf.iterrows():
                frames.append(pd.DataFrame({"연도": [TARGET_YEAR], "값": [r[f'{TARGET_YEAR} 예측']], "시리즈": f"예측:{r['방법']}"}))
        return pd.concat(frames, ignore_index=True)

    st.subheader("업종별 시계열 그래프 (실적 + 2026 예측)")
    plot_df = build_series(sel_ind)
    st.altair_chart(
        alt.Chart(plot_df).mark_line(point=True).encode(
            x="연도:O", y="값:Q", color="시리즈:N"
        ),
        use_container_width=True, theme="streamlit"
    )

    st.subheader("연도별 총합 그래프")
    tot_plot = pd.DataFrame({"연도": YEARS, "값": pv.sum(axis=0).values, "시리즈": "실적(합계)"})
    for mcol in wide.columns:
        tot_plot = pd.concat([tot_plot, pd.DataFrame({"연도": [TARGET_YEAR], "값": [wide[mcol].sum()], "시리즈": f"예측:{mcol}"})], ignore_index=True)
    st.altair_chart(
        alt.Chart(tot_plot).mark_line(point=True).encode(
            x="연도:O", y="값:Q", color="시리즈:N"
        ),
        use_container_width=True, theme="streamlit"
    )

    # 다운로드
    st.subheader("다운로드")
    st.download_button(
        "업종별 예측표 CSV 다운로드",
        final_table.reset_index().to_csv(index=False).encode("utf-8-sig"),
        file_name="industry_forecast_2026.csv", mime="text/csv"
    )

    bio = BytesIO()
    with pd.ExcelWriter(bio, engine="openpyxl") as writer:
        final_table.reset_index().to_excel(writer, index=False, sheet_name="업종별예측")
        total_actual.to_excel(writer, index=False, sheet_name="연도별총합_실적")
        total_2026.to_excel(writer, index=False, sheet_name="연도별총합_2026")
    st.download_button("엑셀(xlsx) 다운로드", bio.getvalue(),
                       file_name="industry_forecast_2026.xlsx",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
else:
    st.info("왼쪽에서 엑셀 업로드하거나 ‘샘플 파일 사용’을 체크한 뒤 [예측 시작]을 눌러주세요.")
