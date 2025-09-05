# app.py — 도시가스 공급·판매 분석 (Poly-3) — RAW 기온(일단위) 자동 집계 지원
import os
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# =============== 기본 환경 ===============
st.set_page_config(page_title="도시가스 공급·판매 분석 (Poly-3)", layout="wide")

# (선택) 한글 폰트 적용 (있을 때만)
def set_korean_font():
    try:
        candidates = [
            Path("data/fonts/NanumGothic-Regular.ttf"),
            Path("data/NanumGothic-Regular.ttf"),
        ]
        for p in candidates:
            if p.exists():
                from matplotlib import font_manager, rcParams
                font_manager.fontManager.addfont(str(p))
                rcParams["font.family"] = "NanumGothic"
                break
    except Exception:
        pass

set_korean_font()

# ===== 공통 유틸 =====
TEMP_COLS = ["당월평균기온", "기간평균기온"]

def yyyymm_col(df):
    out = df.copy()
    out["연월"] = out["연"].astype(int).astype(str).str.zfill(4) + "." + out["월"].astype(int).astype(str).str.zfill(2)
    return out

def pick_value(val):
    if pd.isna(val):
        return ""
    # 온도: 소수1자리(절대값이 200 미만 & 실수)
    if isinstance(val, (float, np.floating)) and abs(val) < 200 and abs(val - round(val)) > 0:
        return f"{val:.1f}"
    # 수치: 콤마
    try:
        return f"{int(round(val)):,}"
    except Exception:
        return f"{val}"

def style_table(df: pd.DataFrame):
    sty = (df.style
           .format(pick_value, na_rep="")
           .set_properties(**{"text-align": "center"})
           .set_table_styles([dict(selector="th", props=[("text-align", "center")])])
           )
    return sty

def add_total_row(df: pd.DataFrame, label="종계") -> pd.DataFrame:
    out = df.copy()
    num_cols = out.select_dtypes(include=[np.number]).columns.tolist()
    total = {c: out[c].sum(skipna=True) for c in num_cols}
    for c in out.columns:
        if c not in total:
            total[c] = ""
    total["연월"] = label
    if "연" in total: total["연"] = ""
    if "월" in total: total["월"] = ""
    out = pd.concat([out, pd.DataFrame([total])], ignore_index=True)
    return out

def poly3_fit(x: np.ndarray, y: np.ndarray):
    poly = PolynomialFeatures(degree=3, include_bias=False)
    X = poly.fit_transform(x.reshape(-1,1))
    model = LinearRegression()
    model.fit(X, y)
    return model, poly

def poly3_predict(model, poly, x: np.ndarray) -> np.ndarray:
    return model.predict(poly.transform(x.reshape(-1,1)))

def list_repo_excels(subdir="data"):
    try:
        files = sorted([f for f in os.listdir(subdir) if f.lower().endswith(".xlsx")])
        return [os.path.join(subdir, f) for f in files]
    except Exception:
        return []

# ============== 엑셀 로더 (기온 RAW 자동 집계) ==============
def _normalize_temp_header(df0: pd.DataFrame) -> pd.DataFrame:
    """
    헤더가 위쪽 설명행에 묻혀 있을 때 '날짜' 또는 '연'이 등장하는 행을 찾아
    그 다음 행부터 데이터로 쓰도록 헤더를 재구성한다.
    """
    df = df0.copy()
    cols = [str(c) for c in df.columns]
    # 이미 정상적으로 읽힌 경우
    if ("날짜" in cols) or (("연" in cols) and ("월" in cols)):
        return df

    # 헤더 탐색: 첫 번째 컬럼에서 '날짜'가 있는 행 찾기
    first_col = df.columns[0]
    idx = None
    for i, v in df[first_col].items():
        if str(v).strip() in ["날짜", "연"]:
            idx = i
            break
    if idx is not None:
        new_header = df.iloc[idx].astype(str).tolist()
        df = df.iloc[idx+1:].copy()
        df.columns = new_header
        return df

    # 그래도 못 찾으면 그대로 반환
    return df

def _temp_from_daily(df_daily: pd.DataFrame) -> pd.DataFrame:
    """
    일 단위 RAW(열: 날짜, 평균기온(℃) 또는 평균기온)를
    월 단위 `연,월,당월평균기온` + (m-1 16 ~ m 15) `기간평균기온`으로 변환
    """
    df = df_daily.copy()
    # 컬럼명 통일
    cols = {c:str(c) for c in df.columns}
    df.rename(columns=cols, inplace=True)

    # 날짜컬럼 찾기
    date_col = None
    for cand in df.columns:
        if "날짜" in str(cand):
            date_col = cand
            break
    if date_col is None:
        raise ValueError("기온 RAW에서 '날짜' 열을 찾지 못했습니다.")

    # 값 컬럼(평균기온) 찾기
    val_col = None
    for cand in df.columns:
        s = str(cand)
        if ("평균기온" in s) or (s == "기온") or ("기온(℃)" in s):
            if cand != date_col:
                val_col = cand
                break
    if val_col is None:
        raise ValueError("기온 RAW에서 '평균기온' 열을 찾지 못했습니다.")

    # 파싱
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df[val_col] = pd.to_numeric(df[val_col], errors="coerce")
    df = df.dropna(subset=[date_col, val_col]).copy()

    # 당월평균기온(달력월 평균)
    df["연"] = df[date_col].dt.year
    df["월"] = df[date_col].dt.month
    m1 = df.groupby(["연","월"], as_index=False)[val_col].mean().rename(columns={val_col:"당월평균기온"})

    # 기간평균기온: (전월 16일 ~ 당월 15일)
    d = df.copy()
    d["pm_year"] = d[date_col].dt.year
    d["pm_month"] = d[date_col].dt.month
    d["day"] = d[date_col].dt.day
    # 16일 이상 → 다음 달로 귀속, 12월 처리
    m = d["pm_month"].values
    y = d["pm_year"].values
    flag = d["day"].values >= 16
    m2 = m.copy()
    y2 = y.copy()
    m2[flag] += 1
    y2[(flag) & (m2==13)] += 1
    m2[m2==13] = 1
    d["연"] = y2
    d["월"] = m2
    m2 = d.groupby(["연","월"], as_index=False)[val_col].mean().rename(columns={val_col:"기간평균기온"})

    out = pd.merge(m1, m2, on=["연","월"], how="outer")
    out = out.sort_values(["연","월"]).reset_index(drop=True)
    return out[["연","월","당월평균기온","기간평균기온"]]

def prepare_temp_any(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    - 일 단위 RAW(날짜/평균기온) → 월단위 집계 + 기간평균 자동생성
    - 월 단위(연/월/당월평균기온) → 기간평균 없으면 당월평균으로 대체
    """
    df = _normalize_temp_header(df_raw)

    cols = [str(c) for c in df.columns]
    if ("날짜" in cols) or any("날짜" in str(c) for c in cols):
        # RAW 일자료
        return _temp_from_daily(df)

    # 월단위 파일 케이스
    rename_map = {}
    for c in df.columns:
        s = str(c)
        if s.strip() in ["평균기온", "평균기온(℃)", "월평균기온(적용)"]:
            rename_map[c] = "당월평균기온"
    if rename_map:
        df = df.rename(columns=rename_map)

    need = {"연","월"}
    if not need.issubset(set(df.columns)):
        raise ValueError("기온 파일에는 최소 '연','월' 칼럼이 있어야 합니다.")

    if "당월평균기온" not in df.columns:
        raise ValueError("기온 파일에서 '당월평균기온'을 찾을 수 없습니다. (RAW 일자료라면 '날짜/평균기온' 형태여야 합니다)")

    if "기간평균기온" not in df.columns:
        df["기간평균기온"] = df["당월평균기온"]

    df = df[["연","월","당월평균기온","기간평균기온"]].copy()
    return df

def load_excel_any(path_or_buf):
    """
    업로드 파일/로컬 경로 모두 지원 + 헤더 보정
    """
    try:
        df0 = pd.read_excel(path_or_buf)
    except Exception as e:
        st.error(f"엑셀 로딩 실패: {e}")
        return None
    return df0

# ============== 사이드바 ==============
with st.sidebar:
    st.header("분석 유형")
    mode = st.radio("선택", ["공급량 분석", "판매량 분석(냉방용)"], index=0)

    st.header("데이터 불러오기")
    src = st.radio("방식", ["Repo 내 파일 사용", "파일 업로드"], index=0)

    use_repo = (src == "Repo 내 파일 사용")

    st.caption("실적파일(공급·판매·기온 중 필요할 것만 사용)")
    if use_repo:
        repo_files = list_repo_excels("data")
        if len(repo_files) == 0:
            st.warning("data/ 폴더에 .xlsx 파일을 올려 두세요.")
        if mode.startswith("공급"):
            supply_path = st.selectbox("공급 파일", repo_files, key="repo_supply")
            temp_path   = st.selectbox("기온 파일", repo_files, key="repo_temp")
            upload = {"supply": None, "sales": None, "temp": None}
        else:
            sales_path  = st.selectbox("판매 실적파일", repo_files, key="repo_sales")
            temp_path   = st.selectbox("기온 파일", repo_files, key="repo_temp2")
            upload = {"supply": None, "sales": None, "temp": None}
    else:
        upload_supply = st.file_uploader("공급 파일(.xlsx)", type=["xlsx"], key="up_supply")
        upload_sales  = st.file_uploader("판매 실적파일(.xlsx)", type=["xlsx"], key="up_sales")
        upload_temp   = st.file_uploader("기온 파일(.xlsx)", type=["xlsx"], key="up_temp")
        upload = {"supply": upload_supply, "sales": upload_sales, "temp": upload_temp}
        supply_path = sales_path = temp_path = None

    st.header("예측 기간")
    c1, c2 = st.columns(2)
    with c1:
        y0 = st.selectbox("예측 시작(연)", list(range(2015, 2031)), index=list(range(2015,2031)).index(2025), key="pred_y0")
        m0 = st.selectbox("예측 시작(월)", list(range(1,13)), index=0, key="pred_m0")
    with c2:
        y1 = st.selectbox("예측 종료(연)", list(range(2015, 2031)), index=list(range(2015,2031)).index(2025), key="pred_y1")
        m1 = st.selectbox("예측 종료(월)", list(range(1,13)), index=11, key="pred_m1")

# ============== 본문: 제목 & ΔT 컨트롤 ==============
st.title("도시가스 공급·판매 분석 (Poly-3)")

for k in ["dt_normal", "dt_best", "dt_cons"]:
    if k not in st.session_state:
        st.session_state[k] = 0.0

def minus(key): st.session_state[key] = round(st.session_state[key] - 0.5, 2)
def plus(key):  st.session_state[key] = round(st.session_state[key] + 0.5, 2)

st.subheader("ΔT 시나리오 (℃)")
c1, c2, c3 = st.columns(3)

with c1:
    st.markdown("**ΔT(Normal)**")
    b1, b2, _ = st.columns([1,1,6])
    with b1: st.button("−", key="btn_norm_minus", on_click=minus, args=("dt_normal",))
    with b2: st.button("+", key="btn_norm_plus",  on_click=plus,  args=("dt_normal",))
    st.metric("기온 보정(℃)", f"{st.session_state.dt_normal:+.2f}")

with c2:
    st.markdown("**ΔT(Best)**")
    b1, b2, _ = st.columns([1,1,6])
    with b1: st.button("−", key="btn_best_minus", on_click=minus, args=("dt_best",))
    with b2: st.button("+", key="btn_best_plus",  on_click=plus,  args=("dt_best",))
    st.metric("기온 보정(℃)", f"{st.session_state.dt_best:+.2f}")

with c3:
    st.markdown("**ΔT(Conservative)**")
    b1, b2, _ = st.columns([1,1,6])
    with b1: st.button("−", key="btn_cons_minus", on_click=minus, args=("dt_cons",))
    with b2: st.button("+", key="btn_cons_plus",  on_click=plus,  args=("dt_cons",))
    st.metric("기온 보정(℃)", f"{st.session_state.dt_cons:+.2f}")

# ============== 데이터 로드 ==============
def read_temp_prepared(source):
    df0 = load_excel_any(source)
    if df0 is None:
        return None
    try:
        temp = prepare_temp_any(df0)
        return temp
    except Exception as e:
        st.error(f"기온 파일 처리 실패: {e}")
        return None

if mode.startswith("공급"):
    if (use_repo and (supply_path is None or temp_path is None)) or ((not use_repo) and (upload["supply"] is None or upload["temp"] is None)):
        st.info("좌측에서 파일을 선택/업로드하세요.")
        st.stop()

    supply_df = load_excel_any(supply_path if use_repo else upload["supply"])
    temp_df   = read_temp_prepared(temp_path if use_repo else upload["temp"])
    if supply_df is None or temp_df is None:
        st.stop()

    for req in ["연","월"]:
        if req not in supply_df.columns:
            st.error("공급 파일에는 '연','월' 칼럼이 필요합니다.")
            st.stop()

    years_all = sorted(supply_df["연"].dropna().unique().tolist())
    choose_years = st.multiselect("학습 연도 선택", years_all, default=years_all[-3:] if len(years_all)>=3 else years_all)

    pred_index = pd.date_range(pd.Timestamp(y0, m0, 1), pd.Timestamp(y1, m1, 1), freq="MS")
    base = pd.DataFrame({"연": pred_index.year, "월": pred_index.month})
    base = base.merge(temp_df, on=["연","월"], how="left")

    base["월평균기온(적용)"] = base["기간평균기온"] + st.session_state.dt_normal
    base_best = base.assign(**{"월평균기온(적용)": base["기간평균기온"] + st.session_state.dt_best})
    base_cons = base.assign(**{"월평균기온(적용)": base["기간평균기온"] + st.session_state.dt_cons})

    train = supply_df.merge(temp_df, on=["연","월"], how="left")
    train = train[train["연"].isin(choose_years)].copy()

    exclude = {"연","월"} | set(TEMP_COLS)
    prod_cols = [c for c in supply_df.columns if c not in exclude]
    if len(prod_cols)==0:
        st.error("공급 파일에 예측할 수치 칼럼(상품)이 없습니다.")
        st.stop()

    def forecast_table(base_df):
        out = yyyymm_col(base_df[["연","월","월평균기온(적용)"]].copy())
        for col in prod_cols:
            try:
                df2 = train[["기간평균기온", col]].dropna()
                if len(df2) < 6:
                    out[col] = np.nan
                    continue
                x = df2["기간평균기온"].values.astype(float)
                y = df2[col].values.astype(float)
                model, poly = poly3_fit(x, y)
                yhat = np.clip(np.rint(poly3_predict(model, poly, base_df["월평균기온(적용)"].values.astype(float))), 0, None)
                out[col] = yhat
            except Exception:
                out[col] = np.nan
        out = add_total_row(out, label="종계")
        return out

    st.markdown("### 예측 결과 — Normal")
    tbl_normal = forecast_table(base)
    st.dataframe(style_table(tbl_normal), use_container_width=True)
    st.download_button("Normal CSV", data=tbl_normal.to_csv(index=False).encode("utf-8-sig"),
                       file_name="supply_forecast_normal.csv")

    st.markdown("### 예측 결과 — Best")
    tbl_best = forecast_table(base_best)
    st.dataframe(style_table(tbl_best), use_container_width=True)
    st.download_button("Best CSV", data=tbl_best.to_csv(index=False).encode("utf-8-sig"),
                       file_name="supply_forecast_best.csv")

    st.markdown("### 예측 결과 — Conservative")
    tbl_cons = forecast_table(base_cons)
    st.dataframe(style_table(tbl_cons), use_container_width=True)
    st.download_button("Conservative CSV", data=tbl_cons.to_csv(index=False).encode("utf-8-sig"),
                       file_name="supply_forecast_conservative.csv")

else:
    if (use_repo and (sales_path is None or temp_path is None)) or ((not use_repo) and (upload["sales"] is None or upload["temp"] is None)):
        st.info("좌측에서 파일을 선택/업로드하세요.")
        st.stop()

    sales_df = load_excel_any(sales_path if use_repo else upload["sales"])
    temp_df  = read_temp_prepared(temp_path if use_repo else upload["temp"])

    if sales_df is None or temp_df is None:
        st.stop()

    for req in ["연","월"]:
        if req not in sales_df.columns:
            st.error("판매 실적파일에는 '연','월' 칼럼이 필요합니다.")
            st.stop()
    if "냉방용" not in sales_df.columns:
        st.error("판매 실적파일에 '냉방용' 칼럼이 필요합니다.")
        st.stop()

    years_all = sorted(sales_df["연"].dropna().unique().tolist())
    choose_years = st.multiselect("학습 연도 선택", years_all, default=years_all[-3:] if len(years_all)>=3 else years_all)

    pred_index = pd.date_range(pd.Timestamp(y0, m0, 1), pd.Timestamp(y1, m1, 1), freq="MS")
    base = pd.DataFrame({"연": pred_index.year, "월": pred_index.month})
    base = base.merge(temp_df, on=["연","월"], how="left")
    base["월평균기온(적용)"] = base["기간평균기온"] + st.session_state.dt_normal

    train = sales_df.merge(temp_df, on=["연","월"], how="left")
    train = train[train["연"].isin(choose_years)].copy()

    # 요약표(예측만)
    out = yyyymm_col(base[["연","월","당월평균기온","기간평균기온","월평균기온(적용)"]].copy())
    df2 = train[["기간평균기온","냉방용"]].dropna()
    if len(df2) >= 6:
        x = df2["기간평균기온"].values.astype(float)
        y = df2["냉방용"].values.astype(float)
        model, poly = poly3_fit(x,y)
        yhat = np.clip(np.rint(poly3_predict(model, poly, base["월평균기온(적용)"].values.astype(float))), 0, None)
        out["예측판매량"] = yhat
    else:
        out["예측판매량"] = np.nan
    out = add_total_row(out, label="종계")

    st.markdown("### 판매량 예측(요약) — Normal")
    st.dataframe(style_table(out), use_container_width=True)
    st.download_button("판매량 예측 CSV 다운로드",
                       data=out.to_csv(index=False).encode("utf-8-sig"),
                       file_name="sales_forecast_normal.csv")

    # 예측 검증표
    st.markdown("### 판매량 예측 검증")
    valid = yyyymm_col(
        base[["연","월","기간평균기온","월평균기온(적용)"]].merge(
            sales_df[["연","월","냉방용"]], on=["연","월"], how="left"
        )
    )
    if len(df2) >= 6:
        valid["예측판매량"] = np.clip(np.rint(poly3_predict(model, poly, base["월평균기온(적용)"].values.astype(float))),0,None)
        valid["실제판매량"] = valid["냉방용"]
        valid["오차"] = valid["실제판매량"] - valid["예측판매량"]
        valid["오차율(%)"] = np.where(valid["실제판매량"]>0, valid["오차"]/valid["실제판매량"]*100, np.nan)
        valid = valid[["연월","실제판매량","예측판매량","오차","오차율(%)"]]
    else:
        valid = valid.assign(**{"실제판매량":np.nan,"예측판매량":np.nan,"오차":np.nan,"오차율(%)":np.nan})
        valid = valid[["연월","실제판매량","예측판매량","오차","오차율(%)"]]

    st.dataframe(style_table(valid), use_container_width=True)

    # 산점도 + Poly-3 + ±1.96 + 식
    if len(df2) >= 6:
        fig, ax = plt.subplots(figsize=(9,5.5), dpi=160)
        ax.scatter(df2["기간평균기온"], df2["냉방용"], alpha=0.6, label="학습 샘플")

        xx = np.linspace(df2["기간평균기온"].min()-2, df2["기간평균기온"].max()+2, 200)
        yy = poly3_predict(model, poly, xx)

        resid = df2["냉방용"].values - poly3_predict(model, poly, df2["기간평균기온"].values)
        s = np.nanstd(resid)
        ax.plot(xx, yy, lw=3, label="Poly-3")
        ax.fill_between(xx, yy-1.96*s, yy+1.96*s, alpha=0.2, label="±1.96")

        coefs = model.coef_
        intercept = model.intercept_
        a = coefs[2] if len(coefs)>=3 else 0.0
        b = coefs[1] if len(coefs)>=2 else 0.0
        c = coefs[0] if len(coefs)>=1 else 0.0
        d = intercept
        eq = f"y = {a:+.5e}x³ {b:+.5e}x² {c:+.5e}x {d:+.5e}"
        r2 = model.score(PolynomialFeatures(3, include_bias=False).fit_transform(df2["기간평균기온"].values.reshape(-1,1)),
                         df2["냉방용"].values)

        ax.text(0.02, 0.02, eq, transform=ax.transAxes, fontsize=10,
                bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"))
        ax.set_title(f"기온-냉방용 실적 상관관계 (Train, R²={r2:.3f})")
        ax.set_xlabel("기간평균기온 (℃)")
        ax.set_ylabel("판매량 (MJ)")
        ax.legend(loc="best"); ax.grid(alpha=0.2)
        st.pyplot(fig)
    else:
        st.info("학습 표본이 충분하지 않아(최소 6개 권장) 검증 그래프를 생략합니다.")
