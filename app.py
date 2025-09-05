os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
warnings.filterwarnings("ignore", category=UserWarning, module="openpyxl")


# ─────────────────────────────────────────────────────────────
# 한글 폰트: 레포의 data/fonts 또는 fonts/NanumGothic-Regular.ttf 우선 적용
# 한글 폰트: 레포의 data/fonts 혹은 fonts 폴더에서 우선 적용
def set_korean_font():
here = Path(__file__).parent if "__file__" in globals() else Path.cwd()
candidates = [
here / "data" / "fonts" / "NanumGothic-Regular.ttf",
        here / "fonts" / "NanumGothic-Regular.ttf",
here / "data" / "fonts" / "NanumGothic.ttf",
        here / "fonts" / "NanumGothic-Regular.ttf",
here / "fonts" / "NanumGothic.ttf",
Path("/usr/share/fonts/truetype/nanum/NanumGothic.ttf"),
Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
@@ -49,14 +48,25 @@ def set_korean_font():
plt.rcParams["font.family"] = ["DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
return False
set_korean_font()

set_korean_font()

# ─────────────────────────────────────────────────────────────
# 공통 유틸
META_COLS = {"날짜", "일자", "date", "연", "년", "월"}
TEMP_HINTS = ["평균기온", "기온", "temperature", "temp"]
KNOWN_PRODUCT_ORDER = ["개별난방용","중앙난방용","자가열전용","일반용(2)","업무난방용","냉난방용","주한미군","총공급량"]
# 중앙난방(신규) 포함
KNOWN_PRODUCT_ORDER = [
    "개별난방용",
    "중앙난방용",
    "중앙난방",  # 추가
    "자가열전용",
    "일반용(2)",
    "업무난방용",
    "냉난방용",
    "주한미군",
    "총공급량",
]

def normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
df = df.copy()
@@ -70,7 +80,7 @@ def normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
else:
if ("연" in df.columns or "년" in df.columns) and "월" in df.columns:
y = df["연"] if "연" in df.columns else df["년"]
            df["날짜"] = pd.to_datetime(y.astype(str) + "-" + df["월"].astype(str) + "-01", errors="coerce")
            df["날짜"] = pd.to_datetime(y.astype(str)+"-"+df["월"].astype(str)+"-01", errors="coerce")
if "연" not in df.columns:
if "년" in df.columns: df["연"] = df["년"]
elif "날짜" in df.columns: df["연"] = df["날짜"].dt.year
@@ -121,14 +131,16 @@ def _finalize(df):
if date_col is None:
for c in df.columns:
try:
                    pd.to_datetime(df[c], errors="raise"); date_col = c; break
                    pd.to_datetime(df[c], errors="raise")
                    date_col = c; break
except Exception:
pass
temp_col = None
for c in df.columns:
            if ("평균기온" in str(c)) or ("기온" in str(c)) or (str(c).lower() in ["temp","temperature"]):
            if ("평균기온" in str(c)) or ("기온" in str(c)) or (isinstance(c,str) and c.lower() in ["temp","temperature"]):
temp_col = c; break
        if date_col is None or temp_col is None: return None
        if date_col is None or temp_col is None:
            return None
out = pd.DataFrame({
"일자": pd.to_datetime(df[date_col], errors="coerce"),
"기온": pd.to_numeric(df[temp_col], errors="coerce")
@@ -144,20 +156,27 @@ def _finalize(df):
header_row = None
for i in range(len(head)):
row = [str(v) for v in head.iloc[i].tolist()]
        if any(v in ["날짜","일자","date","Date"] for v in row) and \
           any(("평균기온" in v) or ("기온" in v) or (isinstance(v,str) and v.lower() in ["temp","temperature"]) for v in row):
        if any(v in ["날짜","일자","date","Date"] for v in row) and any(
            ("평균기온" in v) or ("기온" in v) or (isinstance(v,str) and v.lower() in ["temp","temperature"]) for v in row
        ):
header_row = i; break
df = pd.read_excel(xls, sheet_name=sheet) if header_row is None else pd.read_excel(xls, sheet_name=sheet, header=header_row)
return _finalize(df)

def month_start(x): x = pd.to_datetime(x); return pd.Timestamp(x.year, x.month, 1)
def month_range_inclusive(s, e): return pd.date_range(start=month_start(s), end=month_start(e), freq="MS")
def month_start(x): 
    x = pd.to_datetime(x)
    return pd.Timestamp(x.year, x.month, 1)

def month_range_inclusive(s, e): 
    return pd.date_range(start=month_start(s), end=month_start(e), freq="MS")

def fit_poly3_and_predict(x_train, y_train, x_future):
m = (~np.isnan(x_train)) & (~np.isnan(y_train))
x_train, y_train = x_train[m], y_train[m]
    if np.isnan(x_future).any(): raise ValueError("예측 입력에 결측이 포함되어 있습니다.")
    x_train = x_train.reshape(-1,1); x_future = x_future.reshape(-1,1)
    if np.isnan(x_future).any():
        raise ValueError("예측 입력에 결측이 포함되어 있습니다.")
    x_train = x_train.reshape(-1,1)
    x_future = x_future.reshape(-1,1)
poly = PolynomialFeatures(degree=3, include_bias=False)
Xtr = poly.fit_transform(x_train)
model = LinearRegression().fit(Xtr, y_train)
@@ -166,7 +185,8 @@ def fit_poly3_and_predict(x_train, y_train, x_future):
return y_future, r2

def render_centered_table(df: pd.DataFrame, float1_cols=None, int_cols=None, index=False):
    float1_cols = float1_cols or []; int_cols = int_cols or []
    float1_cols = float1_cols or []
    int_cols = int_cols or []
show = df.copy()
for c in float1_cols:
if c in show.columns:
@@ -182,16 +202,27 @@ def render_centered_table(df: pd.DataFrame, float1_cols=None, int_cols=None, ind
   """, unsafe_allow_html=True)
st.markdown(show.to_html(index=index, classes="centered-table"), unsafe_allow_html=True)

def add_total_row(pivot: pd.DataFrame) -> pd.DataFrame:
    """표 끝에 '종계' 한 줄 추가"""
    p = pivot.copy()
    num_cols = [c for c in p.columns if c not in ["연","월"]]
    total_series = p[num_cols].sum(numeric_only=True)
    total_row = {"연": "", "월": "종계"}
    for c in num_cols:
        v = total_series.get(c, np.nan)
        total_row[c] = np.nan if pd.isna(v) else int(round(v))
    p = pd.concat([p, pd.DataFrame([total_row])], ignore_index=True)
    for c in num_cols:
        p[c] = pd.to_numeric(p[c], errors="coerce").round().astype("Int64")
    return p

# ─────────────────────────────────────────────────────────────
# 분석 유형
with st.sidebar:
st.header("분석 유형")
mode = st.radio("선택", ["공급량 분석", "판매량 분석(냉방용)"], index=0)

# ======================================================================
# A) 공급량 분석
# ======================================================================
# =============== A) 공급량 분석 ==========================================
if mode == "공급량 분석":
with st.sidebar:
st.header("데이터 불러오기")
@@ -202,24 +233,28 @@ def render_centered_table(df: pd.DataFrame, float1_cols=None, int_cols=None, ind
data_dir = Path("data"); data_dir.mkdir(exist_ok=True)
repo_files = sorted([str(p) for p in data_dir.glob("*.xlsx")])
if repo_files:
                # 기본: 상품별공급량_MJ.xlsx가 있으면 우선
default_idx = next((i for i,p in enumerate(repo_files) if "상품별공급량_MJ" in p), 0)
file_choice = st.selectbox("실적 파일(Excel)", repo_files, index=default_idx)
df = read_excel_sheet(file_choice, prefer_sheet="데이터")
else:
st.info("data 폴더에 엑셀 파일이 없습니다. 업로드로 진행하세요.")
else:
up = st.file_uploader("엑셀 업로드(xlsx) — '데이터' 시트", type=["xlsx"])
            if up is not None: df = read_excel_sheet(up, prefer_sheet="데이터")
            if up is not None: 
                df = read_excel_sheet(up, prefer_sheet="데이터")

        if df is None or len(df)==0: st.stop()
        if df is None or len(df)==0: 
            st.stop()

st.subheader("학습 데이터 연도 선택")
years_all = sorted([int(y) for y in pd.Series(df["연"]).dropna().unique()])
years_sel = st.multiselect("연도 선택", years_all, default=years_all)

temp_col = detect_temp_col(df)
if temp_col is None:
            st.error("기온 열을 찾지 못했습니다. 열 이름에 '평균기온' 또는 '기온' 포함 필요."); st.stop()
            st.error("기온 열을 찾지 못했습니다. 열 이름에 '평균기온' 또는 '기온' 포함 필요.")
            st.stop()

st.subheader("예측할 상품 선택")
product_cols = guess_product_cols(df)
@@ -228,10 +263,30 @@ def render_centered_table(df: pd.DataFrame, float1_cols=None, int_cols=None, ind

st.subheader("예측 설정")
last_year = int(df["연"].max())
        start_y = st.selectbox("예측 시작(연)", list(range(2010,2036)), index=list(range(2010,2036)).index(last_year))
        start_m = st.selectbox("예측 시작(월)", list(range(1,13)), index=0)   # 1월 기본
        end_y   = st.selectbox("예측 종료(연)", list(range(2010,2036)), index=list(range(2010,2036)).index(last_year))
        end_m   = st.selectbox("예측 종료(월)", list(range(1,13)), index=11) # 12월 기본

        # 시작(좌: 연도 / 우: 월)
        csy, csm = st.columns(2)
        with csy:
            start_y = st.selectbox(
                "예측 시작(연)", list(range(2010,2036)),
                index=list(range(2010,2036)).index(last_year), key="supply_start_year"
            )
        with csm:
            start_m = st.selectbox(
                "예측 시작(월)", list(range(1,13)), index=0, key="supply_start_month"
            )

        # 종료(좌: 연도 / 우: 월)
        cey, cem = st.columns(2)
        with cey:
            end_y = st.selectbox(
                "예측 종료(연)", list(range(2010,2036)),
                index=list(range(2010,2036)).index(last_year), key="supply_end_year"
            )
        with cem:
            end_m = st.selectbox(
                "예측 종료(월)", list(range(1,13)), index=11, key="supply_end_month"
            )

scen = st.radio("기온 시나리오", ["학습기간 월별 평균", "학습 마지막해 월별 복사", "사용자 업로드(월·기온)"], index=0)
delta = st.slider("기온 보정(Δ°C)", -5.0, 5.0, 0.0, step=0.1)
@@ -254,7 +309,9 @@ def render_centered_table(df: pd.DataFrame, float1_cols=None, int_cols=None, ind
monthly_avg_temp = train_df.groupby("월")[temp_col].mean().rename("temp").to_frame()
f_start = pd.Timestamp(year=int(start_y), month=int(start_m), day=1)
f_end   = pd.Timestamp(year=int(end_y),   month=int(end_m),   day=1)
        if f_end < f_start: st.error("예측 종료가 시작보다 빠릅니다."); st.stop()
        if f_end < f_start: 
            st.error("예측 종료가 시작보다 빠릅니다.")
            st.stop()

fut_idx = month_range_inclusive(f_start, f_end)
fut = pd.DataFrame({"연": fut_idx.year.astype(int), "월": fut_idx.month.astype(int)})
@@ -265,7 +322,9 @@ def render_centered_table(df: pd.DataFrame, float1_cols=None, int_cols=None, ind
base_temp = base[base["연"]==last_train_year][["월",temp_col]].groupby("월")[temp_col].mean().rename("temp").reset_index()
fut = fut.merge(base_temp, on="월", how="left")
else:
            if scen_df is None: st.error("월·기온 시나리오 파일 필요"); st.stop()
            if scen_df is None: 
                st.error("월·기온 시나리오 파일 필요")
                st.stop()
scen_df["month"] = pd.to_numeric(scen_df["month"], errors="coerce").astype(int)
fut = fut.merge(scen_df[["month","temp"]], left_on="월", right_on="month", how="left").drop(columns=["month"])

@@ -274,7 +333,9 @@ def render_centered_table(df: pd.DataFrame, float1_cols=None, int_cols=None, ind

x_train = train_df[temp_col].astype(float).values
x_future = fut["temp"].astype(float).values
        if np.isnan(x_future).any(): st.error("예측 기온 시나리오에 결측이 있습니다."); st.stop()
        if np.isnan(x_future).any():
            st.error("예측 기온 시나리오에 결측이 있습니다.")
            st.stop()

result = {
"forecast_start": f_start,
@@ -285,10 +346,12 @@ def render_centered_table(df: pd.DataFrame, float1_cols=None, int_cols=None, ind

pred_rows = []
for col in prods:
            if col not in base.columns or not pd.api.types.is_numeric_dtype(base[col]): continue
            if col not in base.columns or not pd.api.types.is_numeric_dtype(base[col]): 
                continue
y_train = train_df[col].astype(float).values
y_future, r2 = fit_poly3_and_predict(x_train, y_train, x_future)
            pred = fut[["연","월"]].copy(); pred["pred"] = np.clip(np.rint(y_future).astype(np.int64), a_min=0, a_max=None)
            pred = fut[["연","월"]].copy()
            pred["pred"] = np.clip(np.rint(y_future).astype(np.int64), a_min=0, a_max=None)
hist = base[["연","월",col]].rename(columns={col:"val"}).copy()
result["per_product"][col] = {"hist": hist, "pred": pred, "r2": r2}
tmp = pred.copy(); tmp["상품"] = col; tmp = tmp.rename(columns={"pred":"예측공급량"})
@@ -315,9 +378,11 @@ def render_centered_table(df: pd.DataFrame, float1_cols=None, int_cols=None, ind

res = st.session_state["supply_result"]
st.caption("그래프 아래 ‘표시할 실적 연도’는 즉시 반영됩니다. 좌측 설정은 ‘예측 시작’ 버튼을 눌러야 반영됩니다.")
    years_view = st.multiselect("표시할 실적 연도", options=res["years_all"],
                                default=st.session_state.get("supply_years_view", res["years_all"][-5:]),
                                key="supply_years_view")
    years_view = st.multiselect(
        "표시할 실적 연도", options=res["years_all"],
        default=st.session_state.get("supply_years_view", res["years_all"][-5:]),
        key="supply_years_view"
    )

months = list(range(1,13))
for prod, pkg in res["per_product"].items():
@@ -342,13 +407,18 @@ def render_centered_table(df: pd.DataFrame, float1_cols=None, int_cols=None, ind
plt.tight_layout(); st.pyplot(fig, clear_figure=True)

st.subheader("예측 결과 미리보기")
    render_centered_table(res["pred_table"].head(24), int_cols=[c for c in res["pred_table"].columns if c not in ["연","월"]])
    st.download_button("예측 결과 CSV 다운로드", data=res["pred_table"].to_csv(index=False).encode("utf-8-sig"),
                       file_name="citygas_supply_forecast.csv", mime="text/csv")
    preview = add_total_row(res["pred_table"])
    render_centered_table(
        preview,
        int_cols=[c for c in preview.columns if c not in ["연","월"]]
    )
    st.download_button(
        "예측 결과 CSV 다운로드",
        data=preview.to_csv(index=False).encode("utf-8-sig"),
        file_name="citygas_supply_forecast.csv", mime="text/csv"
    )

# ======================================================================
# B) 판매량 분석(냉방용)
# ======================================================================
# =============== B) 판매량 분석(냉방용) =====================================
else:
st.header("판매량 분석(냉방용) — 전월 16일 ~ 당월 15일 평균기온 기준")
st.write("냉방용 **판매 실적 엑셀**과 **기온 RAW(일별)**을 준비하세요.")
@@ -389,7 +459,8 @@ def _find_repo_sales_and_temp():
with c2: temp_raw_file = st.file_uploader("**기온 RAW(일별)** (xlsx/csv)", type=["xlsx","csv"])

if sales_file is None or temp_raw_file is None:
        st.info("두 파일을 모두 준비하세요."); st.stop()
        st.info("두 파일을 모두 준비하세요.")
        st.stop()

# 판매 실적 자동 매핑
try:
@@ -402,29 +473,37 @@ def _find_repo_sales_and_temp():

# 날짜/냉방 열 추정
date_candidates = [c for c in ["판매월","날짜","일자","date"] if c in sales_df.columns]
    if date_candidates: date_col = date_candidates[0]
    if date_candidates: 
        date_col = date_candidates[0]
else:
score = {}
for c in sales_df.columns:
            try: score[c] = pd.to_datetime(sales_df[c], errors="coerce").notna().mean()
            except Exception: pass
            try: 
                score[c] = pd.to_datetime(sales_df[c], errors="coerce").notna().mean()
            except Exception: 
                pass
date_col = max(score, key=score.get) if score else None

cool_cols = [c for c in sales_df.columns if ("냉방" in str(c)) and pd.api.types.is_numeric_dtype(sales_df[c])]
value_col = None
for c in cool_cols:
        if "냉방용" in str(c): value_col = c; break
        if "냉방용" in str(c): 
            value_col = c; break
value_col = value_col or (cool_cols[0] if cool_cols else None)
if date_col is None or value_col is None:
        st.error("날짜 열 또는 '냉방' 수치 열을 찾지 못했습니다."); st.stop()
        st.error("날짜 열 또는 '냉방' 수치 열을 찾지 못했습니다.")
        st.stop()

sales_df["판매월"] = pd.to_datetime(sales_df[date_col], errors="coerce").dt.to_period("M").dt.to_timestamp()
sales_df["판매량"] = pd.to_numeric(sales_df[value_col], errors="coerce")
sales_df = sales_df.dropna(subset=["판매월","판매량"]).copy()
    sales_df["연"] = sales_df["판매월"].dt.year.astype(int); sales_df["월"] = sales_df["판매월"].dt.month.astype(int)
    sales_df["연"] = sales_df["판매월"].dt.year.astype(int)
    sales_df["월"] = sales_df["판매월"].dt.month.astype(int)

temp_raw = read_temperature_raw(temp_raw_file)
if temp_raw is None or temp_raw.empty:
        st.error("기온 RAW에서 날짜/기온 열을 찾지 못했습니다."); st.stop()
        st.error("기온 RAW에서 날짜/기온 열을 찾지 못했습니다.")
        st.stop()

with st.sidebar:
st.subheader("학습 데이터 연도 선택")
@@ -433,15 +512,36 @@ def _find_repo_sales_and_temp():

st.subheader("예측 설정")
last_year = int(sales_df["연"].max())
        start_y = st.selectbox("예측 시작(연)", list(range(2010,2036)), index=list(range(2010,2036)).index(last_year))
        start_m = st.selectbox("예측 시작(월)", list(range(1,13)), index=0)
        end_y   = st.selectbox("예측 종료(연)", list(range(2010,2036)), index=list(range(2010,2036)).index(last_year))
        end_m   = st.selectbox("예측 종료(월)", list(range(1,13)), index=11)

        # 시작 (좌: 연도 / 우: 월)
        csy, csm = st.columns(2)
        with csy:
            start_y = st.selectbox(
                "예측 시작(연)", list(range(2010,2036)),
                index=list(range(2010,2036)).index(last_year), key="sales_start_year"
            )
        with csm:
            start_m = st.selectbox(
                "예측 시작(월)", list(range(1,13)), index=0, key="sales_start_month"
            )

        # 종료 (좌: 연도 / 우: 월)
        cey, cem = st.columns(2)
        with cey:
            end_y   = st.selectbox(
                "예측 종료(연)", list(range(2010,2036)),
                index=list(range(2010,2036)).index(last_year), key="sales_end_year"
            )
        with cem:
            end_m   = st.selectbox(
                "예측 종료(월)", list(range(1,13)), index=11, key="sales_end_month"
            )
run_btn = st.button("예측 시작", type="primary")

if run_btn:
# 보조 집계
        temp_raw["연"] = temp_raw["일자"].dt.year; temp_raw["월"] = temp_raw["일자"].dt.month
        temp_raw["연"] = temp_raw["일자"].dt.year
        temp_raw["월"] = temp_raw["일자"].dt.month
monthly_cal = temp_raw.groupby(["연","월"])["기온"].mean().reset_index()
fallback_by_M = temp_raw.groupby("월")["기온"].mean()

@@ -452,67 +552,59 @@ def period_avg(label_m: pd.Timestamp) -> float:
mask = (temp_raw["일자"]>=s)&(temp_raw["일자"]<=e)
return temp_raw.loc[mask,"기온"].mean()

        # 학습 데이터(선택 연도) — 기간평균기온 계산
        # 학습 데이터
train_sales = sales_df[sales_df["연"].isin(years_sel)].copy()
rows = [{"판매월":m, "기간평균기온": period_avg(m)} for m in train_sales["판매월"].unique()]
        sj = pd.merge(train_sales[["판매월","판매량","연","월"]], pd.DataFrame(rows), on="판매월", how="left")
        sj = pd.merge(train_sales[["판매월","판매량"]], pd.DataFrame(rows), on="판매월", how="left")
miss = sj["기간평균기온"].isna()
        if miss.any(): sj.loc[miss,"기간평균기온"] = sj.loc[miss,"월"].map(fallback_by_M)
        if miss.any(): 
            sj.loc[miss,"기간평균기온"] = sj.loc[miss,"판매월"].dt.month.map(fallback_by_M)
sj = sj.dropna(subset=["기간평균기온","판매량"])

        # 회귀 학습(그래프용 회귀선도 여기서 계산)
        x_train = sj["기간평균기온"].astype(float).values.reshape(-1,1)
        x_train = sj["기간평균기온"].astype(float).values
y_train = sj["판매량"].astype(float).values
        poly = PolynomialFeatures(degree=3, include_bias=False)
        Xtr = poly.fit_transform(x_train)
        reg = LinearRegression().fit(Xtr, y_train)
        r2_fit = reg.score(Xtr, y_train)
        _fit_pred, r2_fit = fit_poly3_and_predict(x_train, y_train, x_train)

# 예측 입력
f_start = pd.Timestamp(year=int(start_y), month=int(start_m), day=1)
f_end   = pd.Timestamp(year=int(end_y),   month=int(end_m),   day=1)
        if f_end < f_start: st.error("예측 종료가 시작보다 빠릅니다."); st.stop()
        if f_end < f_start: 
            st.error("예측 종료가 시작보다 빠릅니다.")
            st.stop()

        months = month_range_inclusive(f_start, f_end)
        months_idx = month_range_inclusive(f_start, f_end)
rows = []
        for m in months:
        for m in months_idx:
s = (m - pd.offsets.MonthBegin(1)) + pd.DateOffset(days=15)
e = m + pd.DateOffset(days=14)
mask = (temp_raw["일자"]>=s)&(temp_raw["일자"]<=e)
avg_period = temp_raw.loc[mask,"기온"].mean()
avg_month  = monthly_cal.loc[(monthly_cal["연"]==m.year)&(monthly_cal["월"]==m.month),"기온"].mean()
            rows.append({"연":int(m.year), "월":int(m.month), "기간평균기온":avg_period, "당월평균기온":avg_month})
            rows.append({"연":int(m.year),"월":int(m.month),"기간평균기온":avg_period,"당월평균기온":avg_month})
pred = pd.DataFrame(rows)
        for c in ["기간평균기온", "당월평균기온"]:
        for c in ["기간평균기온","당월평균기온"]:
miss = pred[c].isna()
            if miss.any():
                pred.loc[miss, c] = pred.loc[miss, "월"].map(fallback_by_M)
            if miss.any(): 
                pred.loc[miss,c] = pred.loc[miss,"월"].map(fallback_by_M)

        # 예측(회귀 모델 재사용)
        x_future = pred["기간평균기온"].astype(float).values.reshape(-1,1)
        y_future = reg.predict(poly.transform(x_future))
        x_future = pred["기간평균기온"].astype(float).values
        y_future, _ = fit_poly3_and_predict(x_train, y_train, x_future)
pred["예측판매량"] = np.clip(np.rint(y_future).astype(np.int64), a_min=0, a_max=None)

        # 검증용(실제/오차/오차율)
        # 실제/오차 (검증용)
actual = sales_df[["연","월","판매량"]].rename(columns={"판매량":"실제판매량"})
        pred_all = pd.merge(pred, actual, on=["연","월"], how="left")
        pred_all["오차"] = (pred_all["예측판매량"] - pred_all["실제판매량"]).astype("Int64")
        pred_all["오차율(%)"] = np.where(
            pred_all["실제판매량"].notna() & (pred_all["실제판매량"] != 0),
            (pred_all["오차"].astype(float) / pred_all["실제판매량"].astype(float)) * 100.0,
            np.nan
        )
        out = pd.merge(pred, actual, on=["연","월"], how="left")
        out["오차"] = (out["예측판매량"] - out["실제판매량"]).astype("Int64")
        out["오차율(%)"] = ((out["예측판매량"] - out["실제판매량"]) / out["실제판매량"] * 100.0).round(1)
        out.loc[out["실제판매량"].isna(), "오차율(%)"] = pd.NA

st.session_state["sales_result"] = {
"forecast_start": f_start,
"years_all": years_all,
"hist": sales_df.rename(columns={"판매량":"val"})[["연","월","val"]],
            "pred_only": pred,            # 상단(요약)용
            "verify": pred_all,           # 검증표용
            "pred": out,
"r2": r2_fit,
            "train_xy": (sj["기간평균기온"].values, sj["판매량"].values),
            "reg_poly": poly,
            "reg_model": reg
            "train_scatter": pd.DataFrame({"기간평균기온":x_train, "판매량":y_train})
}
if "sales_years_view" not in st.session_state:
default_years = years_all[-5:] if len(years_all)>=5 else years_all
@@ -525,18 +617,20 @@ def period_avg(label_m: pd.Timestamp) -> float:

res = st.session_state["sales_result"]
st.caption("그래프 아래 ‘표시할 실적 연도’는 즉시 반영됩니다. 좌측 설정은 ‘예측 시작’ 버튼을 눌러야 반영됩니다.")
    years_view = st.multiselect("표시할 실적 연도", options=res["years_all"],
                                default=st.session_state.get("sales_years_view", res["years_all"][-5:]),
                                key="sales_years_view")
    years_view = st.multiselect(
        "표시할 실적 연도", options=res["years_all"],
        default=st.session_state.get("sales_years_view", res["years_all"][-5:]),
        key="sales_years_view"
    )

    # 실적/예측 라인 그래프(최근 5개년 + 예측)
    # 실적/예측 그래프(12개월 라인)
months = list(range(1,13))
fig = plt.figure(figsize=(9,3.6)); ax = plt.gca()
for y in sorted([int(v) for v in years_view]):
s = (res["hist"][res["hist"]["연"]==y].set_index("월")["val"]).reindex(months)
ax.plot(months, s.values, label=f"{y} 실적")
pred_vals, y, m = [], int(res["forecast_start"].year), int(res["forecast_start"].month)
    P = res["pred_only"][["연","월","예측판매량"]].copy(); P["연"]=P["연"].astype(int); P["월"]=P["월"].astype(int)
    P = res["pred"][["연","월","예측판매량"]].copy(); P["연"]=P["연"].astype(int); P["월"]=P["월"].astype(int)
for _ in range(12):
row = P[(P["연"]==y)&(P["월"]==m)]
pred_vals.append(row.iloc[0]["예측판매량"] if len(row) else np.nan)
@@ -548,42 +642,44 @@ def period_avg(label_m: pd.Timestamp) -> float:
ax.set_title(f"냉방용 — Poly-3 (Train R²={res['r2']:.3f})"); ax.legend(loc="best")
plt.tight_layout(); st.pyplot(fig, clear_figure=True)

    # 1) 상단 요약표 — 예측판매량까지만
    show_pred = res["pred_only"].rename(columns={"기간평균기온":"기간평균기온 (m-1, 16일 ~ m15일)"})
    show_pred["연"] = show_pred["연"].astype(int).astype(str); show_pred["월"] = show_pred["월"].astype("Int64")
    # 1) 요약 표 — 예측판매량까지만 (실제/오차 제거)
    show = res["pred"].rename(columns={"기간평균기온":"기간평균기온 (m-1, 16일 ~ m15일)"})
    show["연"] = show["연"].astype(int).astype(str); show["월"] = show["월"].astype("Int64")

st.subheader("판매량 예측(요약)")
render_centered_table(
        show_pred[["연","월","당월평균기온","기간평균기온 (m-1, 16일 ~ m15일)","예측판매량"]],
        show[["연","월","당월평균기온","기간평균기온 (m-1, 16일 ~ m15일)","예측판매량"]],
float1_cols=["당월평균기온","기간평균기온 (m-1, 16일 ~ m15일)"],
int_cols=["예측판매량"], index=False
)
st.download_button(
"판매량 예측 CSV 다운로드",
        data=show_pred.to_csv(index=False).encode("utf-8-sig"),
        data=show.to_csv(index=False).encode("utf-8-sig"),
file_name="cooling_sales_forecast.csv", mime="text/csv"
)

    # 2) 검증표 — 실제/오차/오차율
    # 2) 검증 표 — 실제/오차/오차율
st.subheader("판매량 예측 검증")
    ver = res["verify"].copy()
    ver["연"] = ver["연"].astype(int).astype(str); ver["월"] = ver["월"].astype("Int64")
    # 오차율 표현(%.1f)
    ver_disp = ver.copy()
    ver_disp["오차율(%)"] = ver_disp["오차율(%)"].round(1).map(lambda x: "" if pd.isna(x) else f"{x:.1f}")
    check = res["pred"].copy()
    check["연"] = check["연"].astype(int).astype(str); check["월"] = check["월"].astype("Int64")
render_centered_table(
        ver_disp[["연","월","실제판매량","예측판매량","오차","오차율(%)"]],
        check[["연","월","실제판매량","예측판매량","오차","오차율(%)"]],
int_cols=["실제판매량","예측판매량","오차"], index=False
)

    # 3) R² 시각화 — 기간평균기온 vs 냉방용 판매량(산점도 + 회귀곡선)
    st.subheader("기온–냉방용 판매량 상관관계(학습데이터)")
    x_tr, y_tr = res["train_xy"]
    poly, model = res["reg_poly"], res["reg_model"]
    xx = np.linspace(np.nanmin(x_tr), np.nanmax(x_tr), 200).reshape(-1,1)
    yy = model.predict(poly.transform(xx))
    # 3) 상관관계 그래프 — 기간평균기온 vs 냉방용 실적, 회귀곡선 + R²
    st.subheader("기온-냉방용 실적 상관관계 (Train)")
    sc = res["train_scatter"]
    x = sc["기간평균기온"].values
    y = sc["판매량"].values
    # 회귀곡선(3차)
    xs = np.linspace(np.nanmin(x), np.nanmax(x), 200)
    ys, r2_scatter = fit_poly3_and_predict(x, y, xs)

fig2 = plt.figure(figsize=(8,4)); ax2 = plt.gca()
    ax2.scatter(x_tr, y_tr, alpha=0.5, label="학습 샘플")
    ax2.plot(xx.ravel(), yy, linestyle="--", label=f"3차 회귀선 (R²={res['r2']:.3f})")
    ax2.set_xlabel("기간평균기온(°C)"); ax2.set_ylabel("판매량")
    ax2.legend(loc="best"); ax2.grid(True, alpha=0.2)
    ax2.scatter(x, y, s=20, alpha=0.6, label="학습 샘플")
    ax2.plot(xs, ys, lw=2, label=f"Poly-3 Fit (R²={r2_scatter:.3f})")
    ax2.set_xlabel("기간평균기온 (m-1, 16일 ~ m15일)")
    ax2.set_ylabel("냉방용 판매량")
    ax2.legend(loc="best")
plt.tight_layout(); st.pyplot(fig2, clear_figure=True)
