# app.py — 도시가스 공급·판매 분석 (Poly-3)
# - 좌측 [예측 시작]으로 학습 ▶ 우측 ΔT 보정은 즉시 반영
# - Normal / Best(+0.5℃) / Conservative(-0.5℃) 표를 모두 상단에 배치
# - 표 포맷: 기온 1자리, 나머지 천단위 콤마
# - 판매량 분석: 훈련 산점도 + Poly-3 회귀곡선 + 95% 신뢰대역

import os
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.utils import resample
import streamlit as st

# ─────────────────────────────────────────────────────────────
# 기본
st.set_page_config(page_title="도시가스 공급·판매 분석 (Poly-3)", layout="wide")
st.title("도시가스 공급·판매 분석 (Poly-3)")
st.caption("공급량: 기온↔공급량 3차 다항식 · 판매량(냉방용): (전월16~당월15) 평균기온 기반 (단위: MJ)")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
warnings.filterwarnings("ignore", category=UserWarning, module="openpyxl")

# ─────────────────────────────────────────────────────────────
# 한글 폰트(레포의 data/fonts/NanumGothic-*.ttf 우선)
def set_korean_font():
    here = Path(__file__).parent if "__file__" in globals() else Path.cwd()
    candidates = [
        here / "data" / "fonts" / "NanumGothic-Regular.ttf",
        here / "data" / "fonts" / "NanumGothic.ttf",
        Path("/usr/share/fonts/truetype/nanum/NanumGothic.ttf"),
        Path("C:/Windows/Fonts/malgun.ttf"),
    ]
    for p in candidates:
        try:
            if p.exists():
                mpl.font_manager.fontManager.addfont(str(p))
                fam = mpl.font_manager.FontProperties(fname=str(p)).get_name()
                plt.rcParams["font.family"] = [fam]
                plt.rcParams["axes.unicode_minus"] = False
                return
        except Exception:
            pass
    plt.rcParams["font.family"] = ["DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
set_korean_font()

# ─────────────────────────────────────────────────────────────
# 공통 유틸
META_COLS = {"날짜", "일자", "date", "연", "년", "월"}

def normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    # 날짜 파생
    if "날짜" in df.columns:
        df["날짜"] = pd.to_datetime(df["날짜"], errors="coerce")
    elif "일자" in df.columns:
        df["날짜"] = pd.to_datetime(df["일자"], errors="coerce")
    elif "date" in df.columns:
        df["날짜"] = pd.to_datetime(df["date"], errors="coerce")
    else:
        if ("연" in df.columns or "년" in df.columns) and "월" in df.columns:
            y = df["연"] if "연" in df.columns else df["년"]
            df["날짜"] = pd.to_datetime(y.astype(str)+"-"+df["월"].astype(str)+"-01", errors="coerce")
    if "연" not in df.columns:
        df["연"] = df["날짜"].dt.year
    if "월" not in df.columns:
        df["월"] = df["날짜"].dt.month
    # 숫자화
    for c in df.columns:
        if df[c].dtype == "object":
            df[c] = pd.to_numeric(
                df[c].astype(str).str.replace(",", "", regex=False).str.replace(" ", "", regex=False),
                errors="ignore"
            )
    return df

def detect_temp_col(df: pd.DataFrame) -> str | None:
    for c in df.columns:
        nm = str(c).lower()
        if any(k in nm for k in ["평균기온","기온","temperature","temp"]):
            if pd.api.types.is_numeric_dtype(df[c]): return c
    for c in df.columns:
        if "온" in str(c) and pd.api.types.is_numeric_dtype(df[c]): return c
    return None

def month_start(x): x = pd.to_datetime(x); return pd.Timestamp(x.year, x.month, 1)
def month_range_inclusive(s, e): return pd.date_range(start=month_start(s), end=month_start(e), freq="MS")

# 포맷팅: 기온 1자리, 나머지 천단위 콤마(표시용 복사본)
def format_table_for_display(df, temp_cols=()):
    out = df.copy()
    temp_cols = set(temp_cols)
    for c in out.columns:
        if c in temp_cols:
            out[c] = pd.to_numeric(out[c], errors="coerce").map(lambda v: "" if pd.isna(v) else f"{v:.1f}")
        elif pd.api.types.is_numeric_dtype(out[c]):
            out[c] = pd.to_numeric(out[c], errors="coerce").map(lambda v: "" if pd.isna(v) else f"{int(round(v)):,}")
    return out

def poly3_fit(x, y):
    X = x.reshape(-1,1)
    poly = PolynomialFeatures(degree=3, include_bias=True)  # include_bias=True -> 절편 명확
    X3 = poly.fit_transform(X)
    model = LinearRegression().fit(X3, y)
    r2 = model.score(X3, y)
    # 계수 (절편, x, x^2, x^3)
    b = model.intercept_
    c = model.coef_     # include_bias=True일 땐 c[0]는 항상 0, b가 절편
    # 우리가 보기 좋은 순서: 3차 ~ 1차 ~ 절편
    # 실제 예측은 model.predict(poly.transform(...)) 사용
    return model, poly, r2

def poly3_predict(model, poly, x):
    X = np.asarray(x).reshape(-1,1)
    return model.predict(poly.transform(X))

def poly3_equation_string(model, poly, digits=5):
    # include_bias=True: y = b + c1*x + c2*x^2 + c3*x^3
    # scikit의 coef_는 [0, c1, c2, c3], intercept_=b
    b = model.intercept_
    c = model.coef_
    c1, c2, c3 = c[1], c[2], c[3]
    return f"y = {c3:.{digits}f}x³ + {c2:.{digits}f}x² + {c1:.{digits}f}x + {b:.{digits}f}"

# ─────────────────────────────────────────────────────────────
# 데이터 읽기(Repo 기본 / 업로드)
@st.cache_data(ttl=600)
def read_excel_sheet(path_or_file, prefer_sheet="데이터"):
    try:
        xls = pd.ExcelFile(path_or_file, engine="openpyxl")
        sheet = prefer_sheet if prefer_sheet in xls.sheet_names else xls.sheet_names[0]
        df = pd.read_excel(xls, sheet_name=sheet)
    except Exception:
        df = pd.read_excel(path_or_file, engine="openpyxl")
    return normalize_cols(df)

@st.cache_data(ttl=600)
def read_temperature_raw(file):
    def _finalize(df):
        df.columns = [str(c).strip() for c in df.columns]
        # 날짜/기온 추정
        date_col = None
        for c in df.columns:
            if str(c).lower() in ["날짜","일자","date"]:
                date_col = c; break
        if date_col is None:
            for c in df.columns:
                try:
                    pd.to_datetime(df[c], errors="raise"); date_col=c; break
                except Exception: pass
        temp_col = None
        for c in df.columns:
            if ("평균기온" in str(c)) or ("기온" in str(c)) or (str(c).lower() in ["temp","temperature"]):
                temp_col = c; break
        if (date_col is None) or (temp_col is None):
            return None
        out = pd.DataFrame({
            "일자": pd.to_datetime(df[date_col], errors="coerce"),
            "기온": pd.to_numeric(df[temp_col], errors="coerce")
        }).dropna()
        return out.sort_values("일자").reset_index(drop=True)
    name = getattr(file, "name", str(file))
    if name.lower().endswith(".csv"):
        return _finalize(pd.read_csv(file))
    xls = pd.ExcelFile(file, engine="openpyxl")
    sheet = xls.sheet_names[0]
    head = pd.read_excel(xls, sheet_name=sheet, header=None, nrows=40)
    header_row = None
    for i in range(len(head)):
        row = [str(v) for v in head.iloc[i].tolist()]
        if (any(v in ["날짜","일자","date","Date"] for v in row) and
            any(("평균기온" in v) or ("기온" in v) or (isinstance(v,str) and v.lower() in ["temp","temperature"]) for v in row)):
            header_row = i; break
    df = pd.read_excel(xls, sheet_name=sheet) if header_row is None else pd.read_excel(xls, sheet_name=sheet, header=header_row)
    return _finalize(df)

# ─────────────────────────────────────────────────────────────
# 사이드바
with st.sidebar:
    st.header("분석 유형")
    mode = st.radio("선택", ["공급량 분석", "판매량 분석(냉방용)"], index=0)

# =============== A) 공급량 분석 ==========================================
if mode == "공급량 분석":
    with st.sidebar:
        st.header("데이터 불러오기")
        src = st.radio("방식", ["Repo 내 파일 사용", "파일 업로드"], index=0)

        df = None
        if src == "Repo 내 파일 사용":
            data_dir = Path("data"); data_dir.mkdir(exist_ok=True)
            repo_files = sorted([str(p) for p in data_dir.glob("*.xlsx")])
            if repo_files:
                # 기본: '상품별공급량_MJ.xlsx' 우선
                default_idx = next((i for i,p in enumerate(repo_files) if "공급" in Path(p).name), 0)
                file_choice = st.selectbox("실적 파일(Excel)", repo_files, index=default_idx)
                df = read_excel_sheet(file_choice, prefer_sheet="데이터")
            else:
                st.info("data 폴더에 엑셀 파일이 없습니다. 업로드로 진행하세요.")
        else:
            up = st.file_uploader("엑셀 업로드(xlsx)", type=["xlsx"])
            if up is not None:
                df = read_excel_sheet(up, prefer_sheet="데이터")

        if df is None or len(df)==0:
            st.stop()

        st.subheader("학습 데이터 연도 선택")
        years_all = sorted([int(y) for y in pd.Series(df["연"]).dropna().unique()])
        years_sel = st.multiselect("연도 선택", years_all, default=years_all[-5:] if len(years_all)>=5 else years_all)

        temp_col = detect_temp_col(df)
        if temp_col is None:
            st.error("기온 열을 찾지 못했습니다. 열 이름에 '평균기온' 또는 '기온' 포함 필요.")
            st.stop()

        st.subheader("예측할 상품(용도)")
        numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        prod_candidates = [c for c in numeric_cols if c not in META_COLS and c!=temp_col]
        # 정렬: 사용자가 원하던 순서 우선
        order = ["개별난방용","중앙난방용","자가열전용","일반용(2)","업무난방용","냉난방용","주한미군","총공급량"]
        prods = [c for c in order if c in prod_candidates] + [c for c in prod_candidates if c not in order]
        prods = st.multiselect("상품(용도) 선택", prod_candidates, default=prods)

        st.subheader("예측 기간")
        last_year = int(df["연"].max())
        c1, c2 = st.columns(2)
        with c1:
            start_y = st.selectbox("예측 시작(연)", list(range(2010,2036)), index=list(range(2010,2036)).index(last_year))
            end_y   = st.selectbox("예측 종료(연)", list(range(2010,2036)), index=list(range(2010,2036)).index(last_year))
        with c2:
            start_m = st.selectbox("예측 시작(월)", list(range(1,13)), index=0)
            end_m   = st.selectbox("예측 종료(월)", list(range(1,13)), index=11)

        scen = st.radio("기본 온도 시나리오", ["학습기간 월별 평균", "학습 마지막해 월별 복사"], index=0)

        run_btn = st.button("예측 시작", type="primary")

    # 계산은 버튼 클릭 시 1회
    if run_btn:
        base = df.dropna(subset=["날짜"]).sort_values("날짜").reset_index(drop=True)
        train_df = base[base["연"].isin(years_sel)].copy()

        # 월별 평균기온
        monthly_avg_temp = train_df.groupby("월")[temp_col].mean().rename("temp").to_frame()

        # 예측 그리드
        f_start = pd.Timestamp(year=int(start_y), month=int(start_m), day=1)
        f_end   = pd.Timestamp(year=int(end_y),   month=int(end_m),   day=1)
        fut_idx = month_range_inclusive(f_start, f_end)
        fut = pd.DataFrame({"연": fut_idx.year.astype(int), "월": fut_idx.month.astype(int)})
        if scen == "학습기간 월별 평균":
            fut = fut.merge(monthly_avg_temp.reset_index(), on="월", how="left")
        else:
            last_train_year = int(train_df["연"].max()) if len(train_df) else int(base["연"].max())
            base_temp = base[base["연"]==last_train_year][["월",temp_col]].groupby("월")[temp_col].mean().rename("temp").reset_index()
            fut = fut.merge(base_temp, on="월", how="left")

        # 각 상품 모델 학습(기온→공급량)
        models = {}
        for col in prods:
            y_train = train_df.groupby(["연","월"])[col].sum().reset_index()
            # 월평균 기온과 매칭
            x_train = train_df.groupby(["연","월"])[temp_col].mean().reset_index()
            d = pd.merge(x_train, y_train, on=["연","월"], how="inner").dropna()
            if len(d) < 8:
                continue
            model, poly, r2 = poly3_fit(d[temp_col].values.astype(float), d[col].values.astype(float))
            models[col] = {"model":model, "poly":poly, "r2":r2}

        # 세션 저장
        st.session_state["SUPPLY_CTX"] = {
            "base_df": base,
            "train_years": years_sel,
            "fut_base": fut,  # temp 포함
            "models": models,
            "prods": prods,
            "years_all": sorted([int(y) for y in base["연"].dropna().unique()])
        }
        st.success("공급량 예측 학습이 완료되었습니다. 오른쪽의 ΔT 보정을 조절하면 즉시 반영됩니다.")

    if "SUPPLY_CTX" not in st.session_state:
        st.info("좌측에서 설정 후 **예측 시작**을 눌러 실행하세요.")
        st.stop()

    ctx = st.session_state["SUPPLY_CTX"]

    # ΔT 보정 즉시 반영 + Normal/Best/Conservative
    st.subheader("ΔT(℃) 보정 — 움직이면 표/그래프가 즉시 반영됩니다.")
    colA, colB, colC = st.columns([1,1,1])
    with colA:
        dT = st.number_input("Normal ΔT(℃)", value=0.0, step=0.5, format="%.1f")
    with colB:
        dT_best = dT + 0.5
        st.write(f"**Best ΔT = {dT_best:.1f}℃**")
    with colC:
        dT_cons = dT - 0.5
        st.write(f"**Conservative ΔT = {dT_cons:.1f}℃**")

    def make_forecast_table(delta):
        fut = ctx["fut_base"].copy()
        fut["조정기온"] = fut["temp"] + float(delta)
        rows = []
        for col in ctx["prods"]:
            if col not in ctx["models"]: 
                continue
            m = ctx["models"][col]["model"]; p = ctx["models"][col]["poly"]
            pred = np.clip(np.rint(poly3_predict(m, p, fut["조정기온"].values)), a_min=0, a_max=None).astype(np.int64)
            rows.append(pd.DataFrame({"연":fut["연"], "월":fut["월"], col:pred}))
        tbl = fut[["연","월","조정기온"]].copy().rename(columns={"조정기온":"월평균기온(적용)"})
        if rows:
            tbl = tbl.merge(pd.concat(rows, axis=1)[["연","월"]+[c for c in ctx["prods"] if c in ctx["models"]]], on=["연","월"], how="left")
        # 총공급량이 포함되어 있지 않다면 합산 컬럼 제공
        if "총공급량" not in tbl.columns:
            sum_cols = [c for c in ctx["prods"] if c!="총공급량" and c in tbl.columns]
            if sum_cols:
                tbl["총공급량(합산)"] = tbl[sum_cols].sum(axis=1)
        return tbl

    normal_tbl = make_forecast_table(dT)
    best_tbl   = make_forecast_table(dT_best)
    cons_tbl   = make_forecast_table(dT_cons)

    # 3개 표 — 상단에 배치
    st.markdown("### 예측 결과 — **Normal**")
    st.dataframe(format_table_for_display(normal_tbl, temp_cols=["월평균기온(적용)"]), use_container_width=True)
    st.download_button("Normal CSV", data=normal_tbl.to_csv(index=False).encode("utf-8-sig"),
                       file_name="supply_normal.csv", mime="text/csv")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### 예측 결과 — **Best(+0.5℃)**")
        st.dataframe(format_table_for_display(best_tbl, temp_cols=["월평균기온(적용)"]), use_container_width=True)
        st.download_button("Best CSV", data=best_tbl.to_csv(index=False).encode("utf-8-sig"),
                           file_name="supply_best.csv", mime="text/csv")
    with c2:
        st.markdown("### 예측 결과 — **Conservative(-0.5℃)**")
        st.dataframe(format_table_for_display(cons_tbl, temp_cols=["월평균기온(적용)"]), use_container_width=True)
        st.download_button("Conservative CSV", data=cons_tbl.to_csv(index=False).encode("utf-8-sig"),
                           file_name="supply_conservative.csv", mime="text/csv")

    # 그래프: 연도별 실적(선) + 예측(Normal, 점선)  / y축 단위(MJ)
    st.markdown("### 연도별 실적 vs 예측(Normal)")
    months = list(range(1,13))
    for prod in ctx["prods"]:
        if prod not in ctx["models"]:
            continue
        m = ctx["models"][prod]["model"]; p = ctx["models"][prod]["poly"]; r2 = ctx["models"][prod]["r2"]
        # Poly 방정식
        eq = poly3_equation_string(m, p, digits=5)

        fig = plt.figure(figsize=(10,3.6)); ax = plt.gca()
        # 최근 5개년 실적
        hist = ctx["base_df"][["연","월",prod]].dropna()
        show_years = sorted(ctx["years_all"][-5:])
        for y in show_years:
            s = hist[hist["연"]==y].set_index("월")[prod].reindex(months)
            ax.plot(months, s.values, label=f"{y} 실적")
        # 예측(Normal)
        # 12개월 시퀀스
        y, m0 = int(normal_tbl.iloc[0]["연"]), int(normal_tbl.iloc[0]["월"])
        P = normal_tbl[["연","월", prod if prod in normal_tbl.columns else "총공급량(합산)"]].copy()
        name = prod if prod in normal_tbl.columns else "총공급량(합산)"
        pred_vals = []
        y1, m1 = y, m0
        for _ in range(12):
            row = P[(P["연"]==y1)&(P["월"]==m1)]
            pred_vals.append(row.iloc[0][name] if len(row) else np.nan)
            if m1==12: y1+=1; m1=1
            else: m1+=1
        ax.plot(months, pred_vals, linestyle="--", label="예측(Normal)")

        ax.set_xlim(1,12); ax.set_xticks(months); ax.set_xticklabels([f"{mm}월" for mm in months])
        ax.set_xlabel("월"); ax.set_ylabel("공급량 (MJ)")
        ax.set_title(f"{prod} — Poly-3 (Train R²={r2:.3f})")
        ax.legend(loc="best")
        ax.text(0.01, -0.25, eq, transform=ax.transAxes, fontsize=10)  # 방정식 전체 표시
        plt.tight_layout()
        st.pyplot(fig, clear_figure=True)

# =============== B) 판매량 분석(냉방용) =====================================
else:
    st.header("판매량 분석(냉방용) — 전월 16일 ~ 당월 15일 평균기온 기준")
    with st.sidebar:
        st.header("데이터 불러오기")
        sales_src = st.radio("방식", ["Repo 내 파일 사용", "파일 업로드"], index=0)

    def _find_repo_sales_and_temp():
        here = Path(__file__).parent if "__file__" in globals() else Path.cwd()
        data_dir = here / "data"
        sales_candidates = [data_dir / "상품별판매량.xlsx", *data_dir.glob("*판매*.xlsx")]
        temp_candidates  = [data_dir / "기온.xlsx", *data_dir.glob("*기온*.xlsx"), *data_dir.glob("*temp*.csv")]
        s = next((p for p in sales_candidates if p.exists()), None)
        t = next((p for p in temp_candidates if p.exists()), None)
        return s, t

    c1, c2 = st.columns(2)
    if sales_src == "Repo 내 파일 사용":
        repo_sales_path, repo_temp_path = _find_repo_sales_and_temp()
        if not repo_sales_path or not repo_temp_path:
            with c1: sales_file = st.file_uploader("냉방용 **판매 실적 엑셀(xlsx)**", type=["xlsx"])
            with c2: temp_raw_file = st.file_uploader("**기온 RAW(일별)** (xlsx/csv)", type=["xlsx","csv"])
        else:
            st.success(f"레포 파일 사용: {Path(repo_sales_path).name} · {Path(repo_temp_path).name}")
            sales_file = open(repo_sales_path, "rb")
            temp_raw_file = open(repo_temp_path, "rb")
    else:
        with c1: sales_file = st.file_uploader("냉방용 **판매 실적 엑셀(xlsx)**", type=["xlsx"])
        with c2: temp_raw_file = st.file_uploader("**기온 RAW(일별)** (xlsx/csv)", type=["xlsx","csv"])

    if sales_file is None or temp_raw_file is None:
        st.info("두 파일을 모두 준비하세요."); st.stop()

    # 판매 실적(시트 자동 추정)
    try:
        xls = pd.ExcelFile(sales_file, engine="openpyxl")
        sheet = "냉방용" if "냉방용" in xls.sheet_names else xls.sheet_names[0]
        raw_sales = pd.read_excel(xls, sheet_name=sheet)
    except Exception:
        raw_sales = pd.read_excel(sales_file, engine="openpyxl")
    sales_df = normalize_cols(raw_sales)

    # 날짜/냉방 열 추정
    date_candidates = [c for c in ["판매월","날짜","일자","date"] if c in sales_df.columns]
    if date_candidates: date_col = date_candidates[0]
    else:
        score = {}
        for c in sales_df.columns:
            try: score[c] = pd.to_datetime(sales_df[c], errors="coerce").notna().mean()
            except Exception: pass
        date_col = max(score, key=score.get) if score else None
    cool_cols = [c for c in sales_df.columns if ("냉방" in str(c)) and pd.api.types.is_numeric_dtype(sales_df[c])]
    value_col = None
    for c in cool_cols:
        if "냉방용" in str(c): value_col = c; break
    value_col = value_col or (cool_cols[0] if cool_cols else None)
    if date_col is None or value_col is None:
        st.error("날짜 열 또는 '냉방' 수치 열을 찾지 못했습니다."); st.stop()

    sales_df["판매월"] = pd.to_datetime(sales_df[date_col], errors="coerce").dt.to_period("M").dt.to_timestamp()
    sales_df["판매량"] = pd.to_numeric(sales_df[value_col], errors="coerce")
    sales_df = sales_df.dropna(subset=["판매월","판매량"]).copy()
    sales_df["연"] = sales_df["판매월"].dt.year.astype(int)
    sales_df["월"] = sales_df["판매월"].dt.month.astype(int)

    temp_raw = read_temperature_raw(temp_raw_file)
    if temp_raw is None or temp_raw.empty:
        st.error("기온 RAW에서 날짜/기온 열을 찾지 못했습니다."); st.stop()

    with st.sidebar:
        st.subheader("학습 데이터 연도 선택")
        years_all = sorted(sales_df["연"].unique().tolist())
        years_sel = st.multiselect("연도 선택", options=years_all, default=years_all[-5:] if len(years_all)>=5 else years_all)

        st.subheader("예측 기간")
        last_year = int(sales_df["연"].max())
        c1, c2 = st.columns(2)
        with c1:
            start_y = st.selectbox("예측 시작(연)", list(range(2010,2036)), index=list(range(2010,2036)).index(last_year))
            end_y   = st.selectbox("예측 종료(연)", list(range(2010,2036)), index=list(range(2010,2036)).index(last_year))
        with c2:
            start_m = st.selectbox("예측 시작(월)", list(range(1,13)), index=0)
            end_m   = st.selectbox("예측 종료(월)", list(range(1,13)), index=11)

        run_btn = st.button("예측 시작", type="primary", key="sales_run")

    if run_btn:
        # 일별 -> (m-1,16 ~ m,15) 기간평균기온
        def period_avg(label_m: pd.Timestamp) -> float:
            m = month_start(label_m)
            s = (m - pd.offsets.MonthBegin(1)) + pd.DateOffset(days=15)  # 전월16
            e = m + pd.DateOffset(days=14)                                # 당월15
            mask = (temp_raw["일자"]>=s)&(temp_raw["일자"]<=e)
            return temp_raw.loc[mask,"기온"].mean()

        train = sales_df[sales_df["연"].isin(years_sel)].copy()
        x_list, y_list = [], []
        for m in train["판매월"].unique():
            x_list.append(period_avg(m))
        tmp = pd.DataFrame({"판매월":sorted(train["판매월"].unique()), "기간평균기온":x_list})
        sj = pd.merge(train[["판매월","판매량"]], tmp, on="판매월", how="left").dropna()
        X = sj["기간평균기온"].values.astype(float); Y = sj["판매량"].values.astype(float)
        model, poly, r2 = poly3_fit(X, Y)

        # 예측 입력(월별 표)
        f_start = pd.Timestamp(year=int(start_y), month=int(start_m), day=1)
        f_end   = pd.Timestamp(year=int(end_y),   month=int(end_m),   day=1)
        months = month_range_inclusive(f_start, f_end)
        rows = []
        for m in months:
            s = (m - pd.offsets.MonthBegin(1)) + pd.DateOffset(days=15)
            e = m + pd.DateOffset(days=14)
            mask = (temp_raw["일자"]>=s)&(temp_raw["일자"]<=e)
            avg_period = temp_raw.loc[mask,"기온"].mean()
            rows.append({"연":int(m.year),"월":int(m.month),"기간평균기온":avg_period})
        pred_base = pd.DataFrame(rows)

        # 세션 저장
        st.session_state["SALES_CTX"] = {
            "train_df": sj,           # (기간평균기온, 판매량)
            "model": model, "poly": poly, "r2": r2,
            "pred_base": pred_base,   # ΔT만 바꿔서 즉시 예측
            "hist": sales_df[["연","월","판매량"]],
            "years_all": years_all
        }
        st.success("판매량 예측 학습이 완료되었습니다. 오른쪽의 ΔT 보정을 조절하면 즉시 반영됩니다.")

    if "SALES_CTX" not in st.session_state:
        st.info("좌측에서 설정 후 **예측 시작**을 눌러 실행하세요.")
        st.stop()

    sctx = st.session_state["SALES_CTX"]

    # ΔT 보정 즉시 반영 + Normal/Best/Conservative
    st.subheader("ΔT(℃) 보정 — 움직이면 표/그래프가 즉시 반영됩니다.")
    colA, colB, colC = st.columns([1,1,1])
    with colA:
        dT = st.number_input("Normal ΔT(℃)", value=0.0, step=0.5, format="%.1f", key="sales_dT")
    with colB:
        dT_best = dT + 0.5
        st.write(f"**Best ΔT = {dT_best:.1f}℃**")
    with colC:
        dT_cons = dT - 0.5
        st.write(f"**Conservative ΔT = {dT_cons:.1f}℃**")

    def make_sales_table(delta):
        base = sctx["pred_base"].copy()
        base["월평균기온(적용)"] = base["기간평균기온"] + float(delta)
        y = np.clip(np.rint(poly3_predict(sctx["model"], sctx["poly"], base["월평균기온(적용)"].values)),
                    a_min=0, a_max=None).astype(np.int64)
        out = base[["연","월","월평균기온(적용)"]].copy()
        out["예측판매량"] = y
        return out

    normal_tbl = make_sales_table(dT)
    best_tbl   = make_sales_table(dT_best)
    cons_tbl   = make_sales_table(dT_cons)

    # 3개 표 — 상단에 배치
    st.markdown("### 예측 결과 — **Normal**")
    st.dataframe(format_table_for_display(normal_tbl, temp_cols=["월평균기온(적용)"]), use_container_width=True)
    st.download_button("Normal CSV", data=normal_tbl.to_csv(index=False).encode("utf-8-sig"),
                       file_name="sales_normal.csv", mime="text/csv")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### 예측 결과 — **Best(+0.5℃)**")
        st.dataframe(format_table_for_display(best_tbl, temp_cols=["월평균기온(적용)"]), use_container_width=True)
        st.download_button("Best CSV", data=best_tbl.to_csv(index=False).encode("utf-8-sig"),
                           file_name="sales_best.csv", mime="text/csv")
    with c2:
        st.markdown("### 예측 결과 — **Conservative(-0.5℃)**")
        st.dataframe(format_table_for_display(cons_tbl, temp_cols=["월평균기온(적용)"]), use_container_width=True)
        st.download_button("Conservative CSV", data=cons_tbl.to_csv(index=False).encode("utf-8-sig"),
                           file_name="sales_conservative.csv", mime="text/csv")

    # 예측 검증 표(실제/오차/오차율)
    st.markdown("### 판매량 예측 검증 (Normal)")
    actual = sctx["hist"].copy()
    check = pd.merge(normal_tbl[["연","월","예측판매량"]], actual.rename(columns={"판매량":"실제판매량"}), on=["연","월"], how="left")
    check["오차"] = (check["예측판매량"] - check["실제판매량"]).astype("Int64")
    check["오차율(%)"] = np.where(check["실제판매량"].notna() & (check["실제판매량"]!=0),
                             100.0*check["오차"]/check["실제판매량"], np.nan)
    show = check.copy()
    show["오차율(%)"] = show["오차율(%)"].map(lambda v: "" if pd.isna(v) else f"{v:.1f}")
    st.dataframe(format_table_for_display(show, temp_cols=[]), use_container_width=True)

    # 훈련 산점도 + Poly-3 + 95% 신뢰대역
    st.markdown("### 기온-냉방용 실적 상관관계 (Train, R²={:.3f})".format(sctx["r2"]))
    X = sctx["train_df"]["기간평균기온"].values
    Y = sctx["train_df"]["판매량"].values
    x_grid = np.linspace(min(X)-1, max(X)+1, 200)
    y_hat = poly3_predict(sctx["model"], sctx["poly"], x_grid)

    # 간단한 bootstrap으로 95% 밴드
    boot_preds = []
    rng = np.random.RandomState(42)
    for _ in range(200):
        xi, yi = resample(X, Y, replace=True, random_state=rng)
        m, p, _ = poly3_fit(xi, yi)
        boot_preds.append(poly3_predict(m, p, x_grid))
    boot_preds = np.vstack(boot_preds)
    y_lo = np.percentile(boot_preds, 2.5, axis=0)
    y_hi = np.percentile(boot_preds, 97.5, axis=0)

    fig = plt.figure(figsize=(10,4)); ax = plt.gca()
    ax.scatter(X, Y, alpha=0.6, label="학습 샘플")
    ax.plot(x_grid, y_hat, linewidth=2.5, label="Poly-3")
    ax.fill_between(x_grid, y_lo, y_hi, alpha=0.2, label="±1.96(근사)")
    # 온도별 중앙값(시각 보조)
    bins = np.linspace(min(X), max(X), 14)
    med_x, med_y = [], []
    for i in range(len(bins)-1):
        mask = (X>=bins[i])&(X<bins[i+1])
        if mask.sum()>0:
            med_x.append((bins[i]+bins[i+1])/2)
            med_y.append(np.median(Y[mask]))
    if med_x:
        ax.scatter(med_x, med_y, color="orange", label="온도별 중앙값")
    ax.set_xlabel("기간평균기온 (m-1 16일 ~ m 15일)")
    ax.set_ylabel("판매량 (MJ)")
    ax.legend(loc="best")
    eq = poly3_equation_string(sctx["model"], sctx["poly"], digits=5)
    ax.text(0.01, -0.22, eq, transform=ax.transAxes, fontsize=10)
    plt.tight_layout()
    st.pyplot(fig, clear_figure=True)
