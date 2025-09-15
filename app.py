# Create a ready-to-deploy Streamlit app for industry usage forecasting with multiple methods,
# plus requirements.txt, README.md, and a sample Excel file.

from textwrap import dedent
import pandas as pd
import numpy as np
from io import BytesIO

# -------------------- app.py --------------------
app_py = dedent("""
    # app.py — 도시가스 산업용 업종별 2026 예측 (Streamlit)
    # 기능:
    # - 엑셀 업로드(원본 RAW) 또는 리포의 샘플 파일 사용
    # - 예측 방법 4종: 선형추세(OLS), 다항추세(2/3차), CAGR, Holt(지수평활 추세형)
    # - 업종별 예측표 + 연도별 총합 + 다운로드(CSV/XLSX)
    # - 업종 선택 그래프(실적 2021~2025 + 2026 예측), 연도별 총합 그래프
    
    import numpy as np
    import pandas as pd
    import streamlit as st
    from pathlib import Path
    
    # statsmodels는 Holt(지수평활)를 위해 사용
    try:
        from statsmodels.tsa.holtwinters import Holt
    except Exception as e:
        Holt = None  # 설치 이전에도 앱이 뜨도록
    
    st.set_page_config(page_title="도시가스 산업용 업종별 2026 예측", layout="wide")
    st.title("도시가스 공급량·판매량 예측 (업종별, 2026)")
    st.caption("RAW 엑셀 업로드 → 업종별 추세 예측(4종) → 표/그래프/다운로드")
    
    # -------------------- 사이드바: 파일 로드 --------------------
    with st.sidebar:
        st.header("① 데이터 불러오기")
        st.write("엑셀(예: 업종, 2021년 사용량 ~ 2025년 사용량)")
        up = st.file_uploader("원본 엑셀 업로드 (.xlsx)", type=["xlsx"])
        st.write("또는, 리포지토리의 샘플 파일을 사용")
        sample_path = Path("sample_industry_usage.xlsx")
        use_sample = st.checkbox("샘플 파일 사용 (sample_industry_usage.xlsx)", value=sample_path.exists())
    
        st.divider()
        st.header("② 예측 설정")
        methods = st.multiselect(
            "예측 방법(복수 선택)",
            ["선형추세(OLS)", "다항추세(Poly)", "CAGR", "Holt(지수평활)"],
            default=["선형추세(OLS)", "다항추세(Poly)", "CAGR", "Holt(지수평활)"]
        )
        poly_deg = st.selectbox("다항차수(Poly)", [2, 3], index=1)
        st.caption("※ 데이터 포맷이 다를 수 있으므로 첫 번째 문자형 칼럼을 업종으로 인식하고, 연도 열은 '2021', '2022' 같은 숫자로 탐지합니다.")
        run = st.button("예측 시작")
    
    # -------------------- 유틸: 데이터 읽기/정리 --------------------
    YEARS = [2021, 2022, 2023, 2024, 2025]
    TARGET_YEAR = 2026
    
    @st.cache_data
    def read_excel_to_long(file) -> pd.DataFrame:
        df = pd.read_excel(file, engine="openpyxl")
        # 업종 칼럼 추정: 첫 번째 object 타입 칼럼
        obj_cols = [c for c in df.columns if df[c].dtype == "object"]
        if not obj_cols:
            # 혹시 모를 경우 첫 칼럼 사용
            cat_col = df.columns[0]
        else:
            cat_col = obj_cols[0]
        # 연도 칼럼 자동 탐지
        year_cols = []
        for c in df.columns:
            s = str(c)
            for y in YEARS:
                if str(y) in s:
                    year_cols.append(c)
                    break
        # 안전장치: 중복/정렬
        year_map = {}
        for c in year_cols:
            # '2021년 사용량' 같은 텍스트에서 정수 연도 추출
            for y in YEARS:
                if str(y) in str(c):
                    year_map[c] = y
                    break
        year_cols_sorted = [c for c in df.columns if c in year_map]
        # melt
        m = df[[cat_col] + year_cols_sorted].copy()
        m = m.melt(id_vars=[cat_col], var_name="연도열", value_name="사용량")
        m["연도"] = m["연도열"].astype(str).str.extract(r"(20\\d{2})").astype(int)
        m = m[m["연도"].isin(YEARS)]
        m = m.rename(columns={cat_col: "업종"})
        m = m.dropna(subset=["사용량"])
        m["사용량"] = pd.to_numeric(m["사용량"], errors="coerce")
        m = m.dropna(subset=["사용량"])
        return m[["업종", "연도", "사용량"]]
    
    def _linear_forecast(x, y, target):
        # 1차 다항 적합 (OLS)
        coef = np.polyfit(x, y, deg=1)
        yhat = np.polyval(coef, target)
        fit = np.polyval(coef, x)
        return yhat, fit
    
    def _poly_forecast(x, y, target, deg=3):
        deg = max(2, int(deg))
        deg = min(deg, len(x)-1)  # 과적합 방지(데이터 수-1 이하)
        coef = np.polyfit(x, y, deg=deg)
        yhat = np.polyval(coef, target)
        fit = np.polyval(coef, x)
        return yhat, fit
    
    def _cagr_forecast(x, y, target):
        # 2021~2025 사이 CAGR 계산 후 2026 1년치 연장
        # y가 양수일 때만 유효
        y2021 = float(y[0]); y2025 = float(y[-1])
        n = (YEARS[-1] - YEARS[0])
        if y2021 <= 0 or y2025 <= 0 or n <= 0:
            # 비정상치면 선형으로 대체
            return _linear_forecast(x, y, target)[0], np.array(y)
        cagr = (y2025 / y2021) ** (1.0 / n) - 1.0
        yhat = y2025 * (1.0 + cagr) ** (target - YEARS[-1])
        # 과거 적합치는 단순 스무딩 없이 기존 y 사용
        return yhat, np.array(y)
    
    def _holt_forecast(y, steps=1):
        if Holt is None:
            return None, None
        try:
            # trend='add'로 단조 추세 반영, damping 없음
            model = Holt(np.asarray(y), exponential=False, damped_trend=False, initialization_method="estimated")
            fit = model.fit(optimized=True)
            fc = fit.forecast(steps)
            fitted = fit.fittedvalues
            return float(fc[-1]), np.array(fitted)
        except Exception:
            return None, None
    
    # -------------------- 처리 --------------------
    df_long = None
    if run:
        if up is not None:
            df_long = read_excel_to_long(up)
        elif use_sample and sample_path.exists():
            df_long = read_excel_to_long(sample_path)
        else:
            st.error("엑셀을 업로드하거나 샘플 파일 사용을 체크해 주세요.")
    
    if df_long is not None:
        st.success(f"데이터 로드 완료: {df_long['업종'].nunique()}개 업종, {df_long.shape[0]}행")
        # 피벗 테이블(업종 × 연도)
        pv = df_long.pivot_table(index="업종", columns="연도", values="사용량", aggfunc="sum").reindex(columns=YEARS)
        st.dataframe(pv.reset_index().head(10), use_container_width=True)
    
        # 예측 루프
        out_rows = []
        fit_store = {}  # (업종, 방법) -> (연도 리스트, 적합치)
        for industry, row in pv.fillna(0).iterrows():
            y = row.values.astype(float).tolist()
            x = YEARS
            for m in methods:
                if m == "선형추세(OLS)":
                    yh, fitted = _linear_forecast(x, y, TARGET_YEAR)
                    out_rows.append([industry, m, yh])
                    fit_store[(industry, m)] = (x, fitted)
                elif m == "다항추세(Poly)":
                    yh, fitted = _poly_forecast(x, y, TARGET_YEAR, deg=poly_deg)
                    out_rows.append([industry, f"{m}({poly_deg}차)", yh])
                    fit_store[(industry, f"{m}({poly_deg}차)")] = (x, fitted)
                elif m == "CAGR":
                    yh, fitted = _cagr_forecast(x, y, TARGET_YEAR)
                    out_rows.append([industry, m, yh])
                    fit_store[(industry, m)] = (x, fitted if fitted is not None else np.array(y))
                elif m == "Holt(지수평활)":
                    yh, fitted = _holt_forecast(y, steps=1)
                    if yh is None:
                        # 실패 시 선형으로 대체
                        yh, fitted = _linear_forecast(x, y, TARGET_YEAR)
                        m_name = "Holt(지수평활,대체:선형)"
                    else:
                        m_name = m
                    out_rows.append([industry, m_name, yh])
                    fit_store[(industry, m_name)] = (x, fitted if fitted is not None else np.array(y))
    
        forecast_df = pd.DataFrame(out_rows, columns=["업종", "방법", f"{TARGET_YEAR} 예측"])
    
        # 업종별 결과표(가로 확장)
        wide = forecast_df.pivot_table(index="업종", columns="방법", values=f"{TARGET_YEAR} 예측", aggfunc="first")
        # 기존 실적과 결합
        final_table = pv.copy()
        final_table.columns = [f"{c} 실적" for c in final_table.columns]
        final_table = final_table.join(wide)
    
        st.subheader("업종별 예측 표 (2026)")
        st.dataframe(final_table.reset_index(), use_container_width=True)
    
        # 연도별 총합(실적 + 예측)
        total_actual = pv.sum(axis=0).rename(lambda y: f"{y} 실적")
        total_rows = []
        for col in wide.columns:
            total_rows.append([col, "총합", wide[col].sum()])
        total_2026 = pd.DataFrame(total_rows, columns=["방법", "구분", f"{TARGET_YEAR} 합계"])
        total_all = total_actual.to_frame(name="값").reset_index().rename(columns={"index": "구분"})
        st.subheader("연도별 총합")  # 실적(2021~2025)과 방법별 2026 합계
        colA, colB = st.columns([2,1])
        with colA:
            st.dataframe(total_all, use_container_width=True)
        with colB:
            st.dataframe(total_2026, use_container_width=True)
    
        # -------------------- 그래프 --------------------
        st.subheader("업종별 시계열 그래프 (실적 + 2026 예측)")
        industries = ["(전체 합계)"] + list(pv.index)
        sel_ind = st.selectbox("업종 선택", industries, index=0)
    
        import altair as alt
    
        def build_series_for_plot(industry_name):
            frames = []
            if industry_name == "(전체 합계)":
                series_actual = pv.sum(axis=0)
                base = pd.DataFrame({"연도": YEARS, "값": series_actual.values, "시리즈": "실적(합계)"})
                frames.append(base)
                for mcol in wide.columns:
                    frames.append(pd.DataFrame({"연도": [TARGET_YEAR], "값": [wide[mcol].sum()], "시리즈": f"예측:{mcol}"}))
            else:
                series_actual = pv.loc[industry_name, YEARS]
                base = pd.DataFrame({"연도": YEARS, "값": series_actual.values, "시리즈": f"실적:{industry_name}"})
                frames.append(base)
                # 적합치(회귀선) 표시
                for (ind, mname), (xs, fitted) in fit_store.items():
                    if ind == industry_name:
                        frames.append(pd.DataFrame({"연도": xs, "값": fitted, "시리즈": f"적합:{mname}"}))
                # 2026 포인트
                rowf = forecast_df[forecast_df["업종"]==industry_name]
                for _, r in rowf.iterrows():
                    frames.append(pd.DataFrame({"연도": [TARGET_YEAR], "값": [r[f"{TARGET_YEAR} 예측"]], "시리즈": f"예측:{r['방법']}"}))
            return pd.concat(frames, ignore_index=True)
    
        plot_df = build_series_for_plot(sel_ind)
        line = alt.Chart(plot_df).mark_line(point=True).encode(x="연도:O", y="값:Q", color="시리즈:N")
        st.altair_chart(line, use_container_width=True, theme="streamlit")
    
        st.subheader("연도별 총합 그래프")
        tot_plot = pd.DataFrame({"연도": YEARS, "값": pv.sum(axis=0).values, "시리즈": "실적(합계)"})
        for mcol in wide.columns:
            tot_plot = pd.concat([tot_plot, pd.DataFrame({"연도": [TARGET_YEAR], "값": [wide[mcol].sum()], "시리즈": f"예측:{mcol}"})], ignore_index=True)
        bar = alt.Chart(tot_plot).mark_line(point=True).encode(x="연도:O", y="값:Q", color="시리즈:N")
        st.altair_chart(bar, use_container_width=True, theme="streamlit")
    
        # -------------------- 다운로드 --------------------
        st.subheader("다운로드")
        # CSV
        csv_bytes = final_table.reset_index().to_csv(index=False).encode("utf-8-sig")
        st.download_button("업종별 예측표 CSV 다운로드", csv_bytes, file_name="industry_forecast_2026.csv", mime="text/csv")
    
        # XLSX (시트 2개: 업종별 예측표, 연도별 총합)
        from io import BytesIO
        bio = BytesIO()
        with pd.ExcelWriter(bio, engine="openpyxl") as writer:
            final_table.reset_index().to_excel(writer, index=False, sheet_name="업종별예측")
            total_all.to_excel(writer, index=False, sheet_name="연도별총합_실적")
            total_2026.to_excel(writer, index=False, sheet_name="연도별총합_2026")
        st.download_button("엑셀(xlsx) 다운로드", bio.getvalue(), file_name="industry_forecast_2026.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    else:
        st.info("좌측에서 파일을 업로드하고 [예측 시작]을 눌러주세요. 샘플 파일도 사용할 수 있어요.")
""")

# -------------------- requirements.txt --------------------
requirements = dedent("""
    streamlit==1.49.1
    pandas==2.2.2
    numpy==1.26.4
    openpyxl==3.1.5
    statsmodels==0.14.2
    altair==5.3.0
""")

# -------------------- README.md --------------------
readme = dedent("""
    # 도시가스 산업용 업종별 2026 예측 (Streamlit)

    업종별 2021~2025 사용량(엑셀)을 업로드하면 2026년을 **여러 추세 방법(최대 4종)**으로 예측하고
    표/그래프/다운로드까지 한 번에 제공하는 앱입니다.

    ## 기능
    - 엑셀 RAW 업로드 또는 리포의 샘플 파일 사용
    - 예측 방법: **선형추세(OLS), 다항추세(Poly 2/3차), CAGR, Holt(지수평활)**  
    - 업종별 예측표 + 연도별 총합(실적/예측)
    - 업종별 시계열 그래프, 연도별 총합 그래프
    - CSV/XLSX 다운로드

    ## 데이터 포맷
    - 첫 번째 **문자형 칼럼 = 업종명**
    - 연도 칼럼: `2021`, `2022`, `2023`, `2024`, `2025` 가 포함된 열(예: '2021년 사용량')
    - 예시 컬럼: `업종, 2021년 사용량, 2022년 사용량, 2023년 사용량, 2024년 사용량, 2025년 사용량`

    ## 로컬 실행
    ```bash
    pip install -r requirements.txt
    streamlit run app.py
    ```

    ## Streamlit Cloud 배포
    1. 이 리포를 GitHub에 올림
    2. https://share.streamlit.io → **New app**
    3. Repository/Branch/Main file = `app.py` 선택 → **Deploy**

    ## 참고
    - Holt(지수평활)는 `statsmodels`를 사용합니다.
    - 다항차수는 데이터 수에 맞춰 자동으로 `<= (표본수-1)`로 제한합니다.
""")

# -------------------- sample Excel --------------------
sample = pd.DataFrame({
    "업종": ["섬유제품 제조업", "1차금속 제조업", "화학물질 및 화학제품", "기타 제조업"],
    "2021년 사용량": [5_619_083, 3_310_296, 5_870_787, 3_035_526],
    "2022년 사용량": [5_734_493, 3_377_853, 5_990_691, 3_428_032],
    "2023년 사용량": [5_851_523, 3_456_789, 6_115_025, 3_564_975],
    "2024년 사용량": [5_970_942, 3_517_318, 6_237_705, 3_689_123],
    "2025년 사용량": [5_886_021, 2_158_700, 4_000_805, 3_188_953],
})
sample_path = "/mnt/data/sample_industry_usage.xlsx"
with pd.ExcelWriter(sample_path, engine="openpyxl") as w:
    sample.to_excel(w, index=False, sheet_name="업종별사용량")

# Write files
paths = []
for path, content in [
    ("/mnt/data/app.py", app_py),
    ("/mnt/data/requirements.txt", requirements),
    ("/mnt/data/README.md", readme),
]:
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    paths.append(path)

paths.append(sample_path)

paths
