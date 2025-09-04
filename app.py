# app.py — Forecast of City Gas Supply by Product & Usage Type (Poly-3)
# 분석 유형: ① 공급량 분석(기온↔공급량, Poly-3) ② 판매량 분석(냉난방, 검침기간 평균기온 기반)
# 업데이트:
#  - 좌측 바에 분석 유형 분리(공급량/판매량)
#  - 공급량: '데이터' 시트만 사용, 기온 컬럼 자동탐지, 예측 시작~종료월, 정수 표시, 연(Year) 콤마 제거,
#            '총공급량'은 합산이 아닌 별도 모델로 예측
#  - 판매량(냉난방): 판매량 엑셀 + 일별 기온 RAW 업로드, 전월16~당월15 평균기온으로 회귀/예측

import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import streamlit as st

# ─────────────────────────────────────────────────────────────
# 0) 기본 설정
# ─────────────────────────────────────────────────────────────
st.set_page_config(page_title="도시가스 분석(공급/판매) — Poly-3", layout="wide")
st.title("도시가스 공급·판매 분석 (Poly-3)")
st.caption("공급량: 기온↔공급량 3차 다항식 · 판매량(냉난방): 검침기간 평균기온 기반")

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

# ─────────────────────────────────────────────────────────────
# 1) 한글 폰트(가능하면 적용)
# ─────────────────────────────────────────────────────────────
def set_korean_font():
    candidates = [
        "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/System/Library/Fonts/AppleSDGothicNeo.ttc",
    ]
    for p in candidates:
        try:
            if os.path.exists(p):
                mpl.font_manager.fontManager.addfont(p)
                plt.rcParams["font.family"] = mpl.font_manager.FontProperties(fname=p).get_name()
                break
        except Exception:
            pass
    plt.rcParams["axes.unicode_minus"] = False

set_korean_font()

# ─────────────────────────────────────────────────────────────
# 2) 공통 유틸
# ─────────────────────────────────────────────────────────────
META_COLS = {"날짜", "일자", "date", "연", "년", "월"}
TEMP_HINTS = ["평균기온", "기온", "temperature", "temp"]

KNOWN_PRODUCT_ORDER = [
    "개별난방용","중앙난방용","자가열전용","일반용(2)","업무난방용","냉난방용","주한미군","총공급량"
]

def normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    # 날짜 생성/보정
    if "날짜" in df.columns:
        df["날짜"] = pd.to_datetime(df["날짜"], errors="coerce")
    elif "일자" in df.columns:
        df["날짜"] = pd.to_datetime(df["일자"], errors="coerce")
    elif "date" in df.columns:
        df["날짜"] = pd.to_datetime(df["date"], errors="coerce")
    else:
        if ("연" in df.columns or "년" in df.columns) and "월" in df.columns:
            y = df["연"] if "연" in df.columns else df["년"]
            df["날짜"] = pd.to_datetime(y.astype(str) + "-" + df["월"].astype(str) + "-01", errors="coerce")
    if "연" not in df.columns:
        if "년" in df.columns:
            df["연"] = df["년"]
        elif "날짜" in df.columns:
            df["연"] = df["날짜"].dt.year
    if "월" not in df.columns and "날짜" in df.columns:
        df["월"] = df["날짜"].dt.month
    # 숫자형 변환
    for c in df.columns:
        if df[c].dtype == "object":
            df[c] = pd.to_numeric(df[c].astype(str).str.replace(",", "").str.replace(" ", ""), errors="ignore")
    return df

def detect_temp_col(df: pd.DataFrame) -> str | None:
    for c in df.columns:
        name = str(c).lower()
        if any(h in name for h in [h.lower() for h in TEMP_HINTS]) and pd.api.types.is_numeric_dtype(df[c]):
            return c
    for c in df.columns:
        if "온" in str(c) and pd.api.types.is_numeric_dtype(df[c]):
            return c
    return None

def guess_product_cols(df: pd.DataFrame) -> list[str]:
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    candidates = [c for c in numeric_cols if c not in META_COLS]
    ordered = [c for c in KNOWN_PRODUCT_ORDER if c in candidates]
    others  = [c for c in candidates if c not in ordered]
    return ordered + others

@st.cache_data(ttl=600)
def read_excel_sheet(path_or_buffer, prefer_sheet: str = "데이터"):
    """prefer_sheet이 있으면 그 시트, 없으면 첫 시트로 읽어서 normalize."""
    try:
        xls = pd.ExcelFile(path_or_buffer, engine="openpyxl")
        sheet = prefer_sheet if prefer_sheet in xls.sheet_names else xls.sheet_names[0]
        df = pd.read_excel(xls, sheet_name=sheet)
    except Exception:
        # CSV 등도 들어올 수 있음
        df = pd.read_excel(path_or_buffer, engine="openpyxl")
    return normalize_cols(df)

def month_start(x) -> pd.Timestamp:
    x = pd.to_datetime(x)
    return pd.Timestamp(x.year, x.month, 1)

def fit_poly3_and_predict(x_train: np.ndarray, y_train: np.ndarray, x_future: np.ndarray):
    x_train = x_train.reshape(-1, 1)
    x_future = x_future.reshape(-1, 1)
    poly = PolynomialFeatures(degree=3, include_bias=False)
    Xtr = poly.fit_transform(x_train)
    model = LinearRegression()
    model.fit(Xtr, y_train)
    r2 = model.score(Xtr, y_train)
    y_future = model.predict(poly.transform(x_future))
    return y_future, r2

# ─────────────────────────────────────────────────────────────
# 3) 좌측바 — 분석 유형 선택
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("분석 유형")
    mode = st.radio("선택", ["공급량 분석", "판매량 분석(냉난방)"], index=0)

# ===================================================================
#  A) 공급량 분석
# ===================================================================
if mode == "공급량 분석":
    with st.sidebar:
        st.header("데이터 불러오기")
        data_src = st.radio("방식", ["Repo 내 파일 사용", "파일 업로드"], index=0)
        df = None
        if data_src == "Repo 내 파일 사용":
            data_dir = Path("data"); data_dir.mkdir(exist_ok=True)
            repo_files = sorted([str(p) for p in data_dir.glob("*.xlsx")])
            if len(repo_files) == 0:
                st.info("data 폴더에 엑셀 파일이 없습니다. 업로드 탭을 사용하세요.")
            default_idx = next((i for i, p in enumerate(repo_files) if "상품별공급량_MJ" in p), 0) if repo_files else 0
            file_choice = st.selectbox("실적 파일(Excel)", repo_files if repo_files else ["<None>"], index=default_idx if repo_files else 0)
            if repo_files:
                df = read_excel_sheet(file_choice, prefer_sheet="데이터")
        else:
            up = st.file_uploader("엑셀 업로드(xlsx) — '데이터' 시트 사용", type=["xlsx"])
            if up is not None:
                df = read_excel_sheet(up, prefer_sheet="데이터")

        if df is None or len(df) == 0:
            st.stop()

        # 학습 연도
        st.subheader("학습 데이터 연도 선택")
        years = sorted([int(y) for y in pd.Series(df["연"]).dropna().unique()])
        sel_years = st.multiselect("연도 선택", years, default=years, help="필요 없는 연도는 ×로 제거")

        # 기온 컬럼 자동탐지
        temp_col = detect_temp_col(df)
        if temp_col is None:
            st.error("기온 컬럼을 자동으로 찾지 못했습니다. 엑셀의 기온 열 이름에 '평균기온' 또는 '기온'을 포함시켜 주세요.")
            st.stop()

        # 예측할 상품
        st.subheader("예측할 상품 선택")
        product_cols = guess_product_cols(df)
        default_products = [c for c in KNOWN_PRODUCT_ORDER if c in product_cols] or product_cols[:6]
        sel_products = st.multiselect("상품(용도) 선택", product_cols, default=default_products)

        # 예측 범위 & 시나리오
        st.subheader("예측 설정")
        last_hist_m = month_start(pd.to_datetime(df["날짜"].max()))
        default_start = (last_hist_m + pd.offsets.MonthBegin(1)).date()
        default_end   = (last_hist_m + pd.DateOffset(months=12)).date()
        c1, c2 = st.columns(2)
        with c1:
            forecast_start = st.date_input("예측 시작월", value=default_start)
        with c2:
            forecast_end = st.date_input("예측 종료월", value=default_end)
        if pd.to_datetime(forecast_end) < pd.to_datetime(forecast_start):
            st.error("예측 종료월이 시작월보다 빠릅니다.")
            st.stop()

        scen = st.radio("기온 시나리오", ["학습기간 월별 평균", "학습 마지막해 월별 복사", "사용자 업로드(월·기온)"], index=0)
        delta = st.slider("기온 보정(Δ°C)", -5.0, 5.0, 0.0, step=0.1)

        scen_df = None
        if scen == "사용자 업로드(월·기온)":
            up_scen = st.file_uploader("CSV/XLSX 업로드 (열: 월, 기온 또는 month, temp)", type=["csv","xlsx"], key="temp_scen_upload")
            if up_scen is not None:
                if up_scen.name.lower().endswith(".csv"):
                    scen_df = pd.read_csv(up_scen)
                else:
                    scen_df = pd.read_excel(up_scen)
                scen_df.columns = [str(c).strip().lower() for c in scen_df.columns]
                if "월" in scen_df.columns: scen_df["month"] = scen_df["월"]
                if "기온" in scen_df.columns: scen_df["temp"] = scen_df["기온"]
                if "month" not in scen_df.columns or "temp" not in scen_df.columns:
                    st.error("업로드 형식: '월/month'와 '기온/temp' 열이 필요합니다.")
                    st.stop()

    # ─ 데이터 전처리
    df = df.dropna(subset=["날짜"]).sort_values("날짜").reset_index(drop=True)
    train_df = df[df["연"].isin(sel_years)].copy()
    if len(train_df) < 6:
        st.warning("학습 샘플이 적습니다. 연도를 더 선택하세요.")

    monthly_avg_temp = train_df.groupby("월")[temp_col].mean().rename("temp").to_frame()

    # 예측 타임라인
    start_m = month_start(forecast_start)
    end_m   = month_start(forecast_end)
    future_index = pd.date_range(start=start_m, end=end_m, freq="MS")
    future_df = pd.DataFrame({"날짜": future_index})
    future_df["연"] = future_df["날짜"].dt.year.astype(int)
    future_df["월"] = future_df["날짜"].dt.month.astype(int)

    # 기온 시나리오
    if scen == "학습기간 월별 평균":
        future_df = future_df.merge(monthly_avg_temp.reset_index(), on="월", how="left")
    elif scen == "학습 마지막해 월별 복사":
        last_train_year = int(train_df["연"].max()) if len(train_df) else int(df["연"].max())
        base = df[df["연"] == last_train_year][["월", temp_col]].groupby("월")[temp_col].mean().rename("temp").reset_index()
        future_df = future_df.merge(base, on="월", how="left")
    else:
        scen_df["month"] = pd.to_numeric(scen_df["month"], errors="coerce").astype(int)
        scen_df = scen_df[["month","temp"]].dropna()
        future_df = future_df.merge(scen_df, left_on="월", right_on="month", how="left").drop(columns=["month"])

    # Δ°C 보정 & 결측 보강
    fallback_monthly = df.groupby("월")[temp_col].mean()
    future_df["temp"] = (future_df["temp"] + delta).fillna(future_df["월"].map(monthly_avg_temp["temp"])).fillna(future_df["월"].map(fallback_monthly))

    # ─ 학습 + 예측
    results = []
    x_train = train_df[temp_col].astype(float).values
    x_future = future_df["temp"].astype(float).values

    for col in sel_products:
        if col not in df.columns or not pd.api.types.is_numeric_dtype(df[col]): 
            continue
        y_train = train_df[col].astype(float).values
        y_pred_future, r2 = fit_poly3_and_predict(x_train, y_train, x_future)
        y_pred_future = np.rint(y_pred_future).astype(np.int64)

        tmp = future_df[["날짜","연","월"]].copy()
        tmp["상품"] = col
        tmp["예측공급량"] = np.clip(y_pred_future, a_min=0, a_max=None)
        tmp["R2(Train)"] = r2
        results.append(tmp)

        # 차트
        hist = df[["날짜", col]].copy().rename(columns={col:"실제"})
        pred = tmp[["날짜","예측공급량"]].rename(columns={"예측공급량":"예측"})
        merged = pd.merge(hist, pred, on="날짜", how="outer").sort_values("날짜")

        fig = plt.figure(figsize=(9, 3.6))
        plt.plot(merged["날짜"], merged["실제"], label=f"{col} 실제")
        plt.plot(merged["날짜"], merged["예측"], linestyle="--", label=f"{col} 예측")
        plt.title(f"{col} — Poly-3 Forecast (Train R²={r2:.3f})")
        plt.xlabel("날짜"); plt.ylabel("공급량"); plt.legend(loc="best"); plt.tight_layout()
        st.pyplot(fig)

    if not results:
        st.error("예측할 상품이 선택되지 않았거나 데이터 형식이 맞지 않습니다.")
        st.stop()

    pred_df = pd.concat(results, ignore_index=True)
    pred_pivot = pred_df.pivot_table(index=["날짜","연","월"], columns="상품", values="예측공급량").reset_index()

    # 총공급량: 별도 학습/예측
    if "총공급량" in df.columns and pd.api.types.is_numeric_dtype(df["총공급량"]):
        y_total_train = train_df["총공급량"].astype(float).values
        y_total_pred, r2_tot = fit_poly3_and_predict(x_train, y_total_train, x_future)
        y_total_pred = np.rint(y_total_pred).astype(np.int64)
        pred_pivot["총공급량"] = np.clip(y_total_pred, a_min=0, a_max=None)
    else:
        st.warning("원본 데이터에 '총공급량' 열이 없어 총량 예측을 표시하지 않습니다.")

    # 컬럼 정렬: 날짜/연/월 → 선택상품 → 총공급량
    ordered_cols = ["날짜","연","월"] + [c for c in sel_products if c in pred_pivot.columns and c != "총공급량"]
    if "총공급량" in pred_pivot.columns:
        ordered_cols += ["총공급량"]
    pred_pivot = pred_pivot[ordered_cols]

    # 숫자형 정수화(연은 콤마 방지를 위해 문자열 처리)
    pred_pivot["연"] = pred_pivot["연"].astype(int).astype(str)   # <-- 연도 콤마 제거
    pred_pivot["월"] = pred_pivot["월"].astype("Int64")
    for c in pred_pivot.columns:
        if c not in ["날짜","연","월"] and (pd.api.types.is_float_dtype(pred_pivot[c]) or pd.api.types.is_integer_dtype(pred_pivot[c])):
            pred_pivot[c] = pd.to_numeric(pred_pivot[c], errors="coerce").round().astype("Int64")

    st.subheader("예측 결과 미리보기")
    st.dataframe(
        pred_pivot.head(24),
        use_container_width=True,
        column_config={
            "연": st.column_config.TextColumn("연"),  # 콤마 없는 년도
        },
    )

    csv = pred_pivot.to_csv(index=False).encode("utf-8-sig")
    st.download_button("예측 결과 CSV 다운로드", data=csv, file_name="citygas_forecast.csv", mime="text/csv")

    with st.expander("사용된 기온 시나리오 보기"):
        scen_view = future_df[["날짜","월","temp"]].copy()
        scen_view["월"] = scen_view["월"].astype("Int64")
        st.dataframe(scen_view.rename(columns={"temp":"기온(°C)"}), use_container_width=True)

# ===================================================================
#  B) 판매량 분석(냉난방)
# ===================================================================
else:
    st.header("판매량 분석 — 냉난방(전월 16일 ~ 당월 15일 기준)")
    st.write("판매량 엑셀과 **일별 기온 RAW**를 업로드해 검침기간 평균기온으로 판매량을 분석/예측합니다.")

    col1, col2 = st.columns(2)
    with col1:
        sales_file = st.file_uploader("냉난방상품 **판매량 엑셀(xlsx)** 업로드", type=["xlsx"])
    with col2:
        temp_raw_file = st.file_uploader("**일별 기온 RAW** (xlsx/csv)", type=["xlsx", "csv"])

    if sales_file is None or temp_raw_file is None:
        st.info("두 파일을 모두 업로드하세요.")
        st.stop()

    # 판매량 파일 읽기(시트: '데이터' 우선)
    sales_df = read_excel_sheet(sales_file, prefer_sheet="데이터")
    # 후보 컬럼 탐지
    # 날짜(판매월) 후보
    cand_date_cols = [c for c in sales_df.columns if any(x in str(c) for x in ["판매월","판매_월","월","날짜","date"])]
    cand_val_cols  = [c for c in sales_df.columns if any(x in str(c) for x in ["판매","수량","사용","냉난방"]) and pd.api.types.is_numeric_dtype(sales_df[c])]

    st.subheader("판매량 컬럼 매핑")
    cA, cB = st.columns(2)
    with cA:
        sales_month_col = st.selectbox("판매월 컬럼", options=cand_date_cols if cand_date_cols else list(sales_df.columns))
    with cB:
        sales_value_col = st.selectbox("판매량 컬럼", options=cand_val_cols if cand_val_cols else list(sales_df.columns))

    # 판매월 정규화
    sales_df["판매월"] = pd.to_datetime(sales_df[sales_month_col], errors="coerce")
    sales_df = sales_df.dropna(subset=["판매월"]).copy()
    # 월 라벨은 해당 월의 1일로 통일
    sales_df["판매월"] = sales_df["판매월"].dt.to_period("M").dt.to_timestamp()
    sales_df["판매량"] = pd.to_numeric(sales_df[sales_value_col], errors="coerce")

    # 기온 RAW 읽기
    if temp_raw_file.name.lower().endswith(".csv"):
        temp_raw = pd.read_csv(temp_raw_file)
    else:
        temp_raw = read_excel_sheet(temp_raw_file, prefer_sheet="데이터")
    # 기온 컬럼/날짜 컬럼 자동탐지
    temp_raw.columns = [str(c).strip() for c in temp_raw.columns]
    temp_date_col = None
    for c in temp_raw.columns:
        if str(c).lower() in ["날짜","일자","date"]:
            temp_date_col = c; break
    if temp_date_col is None:
        # 첫 날짜형 컬럼
        for c in temp_raw.columns:
            try:
                pd.to_datetime(temp_raw[c])
                temp_date_col = c; break
            except Exception:
                pass
    temp_value_col = detect_temp_col(temp_raw) or "기온"
    temp_raw["일자"] = pd.to_datetime(temp_raw[temp_date_col], errors="coerce")
    temp_raw["기온"] = pd.to_numeric(temp_raw[temp_value_col], errors="coerce")
    temp_raw = temp_raw.dropna(subset=["일자","기온"]).sort_values("일자")

    # 검침기간 평균기온 계산 함수: 판매월 m에 대해 (m-1월 16일 ~ m월 15일)
    def billing_period_avg_temp(daily_df: pd.DataFrame, label_month: pd.Timestamp) -> tuple[float, pd.Timestamp, pd.Timestamp]:
        m = month_start(label_month)  # 해당 월 1일
        start = (m - pd.offsets.MonthBegin(1)) + pd.DateOffset(days=15)  # 전월 16일
        end   = m + pd.DateOffset(days=14)                                # 당월 15일
        mask = (daily_df["일자"] >= start) & (daily_df["일자"] <= end)
        avg = daily_df.loc[mask, "기온"].mean()
        return avg, start, end

    # 학습용: 판매월이 존재하는 구간에 대해 평균기온 매칭
    rows = []
    for m in sales_df["판매월"].unique():
        avg, s, e = billing_period_avg_temp(temp_raw, m)
        rows.append({"판매월": m, "기간시작": s, "기간끝": e, "기간평균기온": avg})
    period_df = pd.DataFrame(rows)
    sales_join = pd.merge(sales_df[["판매월","판매량"]], period_df, on="판매월", how="left").dropna(subset=["기간평균기온","판매량"])

    if len(sales_join) < 6:
        st.warning("학습에 사용할 샘플이 적습니다. 판매월 데이터를 더 제공하세요.")
        st.stop()

    # 회귀 학습(Poly-3)
    x_train = sales_join["기간평균기온"].astype(float).values
    y_train = sales_join["판매량"].astype(float).values
    y_fit, r2_fit = fit_poly3_and_predict(x_train, y_train, x_train)
    # 예측 범위(판매월 라벨)
    st.subheader("예측 범위(판매월)")
    default_sale_start = (sales_join["판매월"].max() + pd.offsets.MonthBegin(1)).date()
    default_sale_end   = (pd.Timestamp(default_sale_start) + pd.DateOffset(months=5)).date()
    c1, c2 = st.columns(2)
    with c1:
        sale_fore_start = st.date_input("예측 판매월 시작", value=default_sale_start, key="sale_start")
    with c2:
        sale_fore_end   = st.date_input("예측 판매월 종료", value=default_sale_end, key="sale_end")
    if pd.to_datetime(sale_fore_end) < pd.to_datetime(sale_fore_start):
        st.error("예측 종료월이 시작월보다 빠릅니다.")
        st.stop()

    # 예측용 판매월 timeline
    sale_months = pd.date_range(start=month_start(sale_fore_start), end=month_start(sale_fore_end), freq="MS")
    pred_rows = []
    for m in sale_months:
        avg, s, e = billing_period_avg_temp(temp_raw, m)
        pred_rows.append({"판매월": m, "기간시작": s, "기간끝": e, "기간평균기온": avg})
    pred_input = pd.DataFrame(pred_rows)
    x_future = pred_input["기간평균기온"].astype(float).values
    y_future, r2_model = fit_poly3_and_predict(x_train, y_train, x_future)
    y_future = np.rint(y_future).astype(np.int64)
    pred_input["예측판매량"] = np.clip(y_future, a_min=0, a_max=None)

    # 학습구간과 비교 테이블(있는 월)
    compare = pd.merge(sales_join, pred_input[["판매월","기간평균기온","예측판매량"]], on=["판매월","기간평균기온"], how="left")
    compare["오차"] = (compare["예측판매량"] - compare["판매량"]).astype("Int64")

    st.subheader("판매량 결과(냉난방)")
    # 표시 형식: 정수, 연 콤마 없음
    disp_cols = ["판매월","기간시작","기간끝","기간평균기온","판매량","예측판매량","오차"]
    # 연도 콤마 제거를 위해 판매월을 문자열(YYYY-MM)로도 제공
    table_show = compare.copy()
    table_show["판매월"] = table_show["판매월"].dt.strftime("%Y-%m")
    st.dataframe(table_show[disp_cols], use_container_width=True)

    st.download_button(
        "판매량 분석 결과 CSV 다운로드",
        data=table_show[disp_cols].to_csv(index=False).encode("utf-8-sig"),
        file_name="cooling_heating_sales_analysis.csv",
        mime="text/csv",
    )

    # 차트: 실제 vs 예측
    chart_df = compare.sort_values("판매월")
    fig = plt.figure(figsize=(9, 3.6))
    plt.plot(chart_df["판매월"], chart_df["판매량"], label="실제 판매량")
    plt.plot(chart_df["판매월"], chart_df["예측판매량"], linestyle="--", label="예측 판매량")
    plt.title(f"냉난방 판매량 — Poly-3 (Train R²={r2_fit:.3f})")
    plt.xlabel("판매월"); plt.ylabel("판매량"); plt.legend(loc="best"); plt.tight_layout()
    st.pyplot(fig)

    # 앞으로 예측만 모은 표
    future_only = pred_input.copy()
    future_only["판매월"] = future_only["판매월"].dt.strftime("%Y-%m")
    st.subheader("향후 예측(판매월 기준)")
    st.dataframe(future_only[["판매월","기간시작","기간끝","기간평균기온","예측판매량"]], use_container_width=True)
    st.download_button(
        "향후 예측 CSV 다운로드",
        data=future_only[["판매월","기간시작","기간끝","기간평균기온","예측판매량"]].to_csv(index=False).encode("utf-8-sig"),
        file_name="cooling_heating_sales_forecast.csv",
        mime="text/csv",
    )
