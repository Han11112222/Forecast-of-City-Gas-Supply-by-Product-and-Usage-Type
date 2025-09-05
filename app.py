os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
warnings.filterwarnings("ignore", category=UserWarning, module="openpyxl")


# ─────────────────────────────────────────────────────────────
# 한글 폰트: 레포의 data/fonts/NanumGothic-Regular.ttf 우선 적용
# 한글 폰트: 레포의 data/fonts 또는 fonts/NanumGothic-Regular.ttf 우선 적용
def set_korean_font():
here = Path(__file__).parent if "__file__" in globals() else Path.cwd()
candidates = [
here / "data" / "fonts" / "NanumGothic-Regular.ttf",
        here / "fonts" / "NanumGothic-Regular.ttf",
here / "data" / "fonts" / "NanumGothic.ttf",
        here / "fonts" / "NanumGothic.ttf",
Path("/usr/share/fonts/truetype/nanum/NanumGothic.ttf"),
Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
Path("/Library/Fonts/AppleSDGothicNeo.ttc"),
@@ -48,6 +51,7 @@ def set_korean_font():
return False
set_korean_font()


# ─────────────────────────────────────────────────────────────
# 공통 유틸
META_COLS = {"날짜", "일자", "date", "연", "년", "월"}
@@ -66,7 +70,7 @@ def normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
else:
if ("연" in df.columns or "년" in df.columns) and "월" in df.columns:
y = df["연"] if "연" in df.columns else df["년"]
            df["날짜"] = pd.to_datetime(y.astype(str)+"-"+df["월"].astype(str)+"-01", errors="coerce")
            df["날짜"] = pd.to_datetime(y.astype(str) + "-" + df["월"].astype(str) + "-01", errors="coerce")
if "연" not in df.columns:
if "년" in df.columns: df["연"] = df["년"]
elif "날짜" in df.columns: df["연"] = df["날짜"].dt.year
@@ -116,15 +120,19 @@ def _finalize(df):
date_col = c; break
if date_col is None:
for c in df.columns:
                try: pd.to_datetime(df[c], errors="raise"); date_col = c; break
                except Exception: pass
                try:
                    pd.to_datetime(df[c], errors="raise"); date_col = c; break
                except Exception:
                    pass
temp_col = None
for c in df.columns:
if ("평균기온" in str(c)) or ("기온" in str(c)) or (str(c).lower() in ["temp","temperature"]):
temp_col = c; break
if date_col is None or temp_col is None: return None
        out = pd.DataFrame({"일자": pd.to_datetime(df[date_col], errors="coerce"),
                            "기온": pd.to_numeric(df[temp_col], errors="coerce")}).dropna()
        out = pd.DataFrame({
            "일자": pd.to_datetime(df[date_col], errors="coerce"),
            "기온": pd.to_numeric(df[temp_col], errors="coerce")
        }).dropna()
return out.sort_values("일자").reset_index(drop=True)

name = getattr(file, "name", str(file))
@@ -136,7 +144,8 @@ def _finalize(df):
header_row = None
for i in range(len(head)):
row = [str(v) for v in head.iloc[i].tolist()]
        if any(v in ["날짜","일자","date","Date"] for v in row) and any(("평균기온" in v) or ("기온" in v) or (isinstance(v,str) and v.lower() in ["temp","temperature"]) for v in row):
        if any(v in ["날짜","일자","date","Date"] for v in row) and \
           any(("평균기온" in v) or ("기온" in v) or (isinstance(v,str) and v.lower() in ["temp","temperature"]) for v in row):
header_row = i; break
df = pd.read_excel(xls, sheet_name=sheet) if header_row is None else pd.read_excel(xls, sheet_name=sheet, header=header_row)
return _finalize(df)
@@ -173,13 +182,16 @@ def render_centered_table(df: pd.DataFrame, float1_cols=None, int_cols=None, ind
   """, unsafe_allow_html=True)
st.markdown(show.to_html(index=index, classes="centered-table"), unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# 분석 유형
with st.sidebar:
st.header("분석 유형")
mode = st.radio("선택", ["공급량 분석", "판매량 분석(냉방용)"], index=0)

# =============== A) 공급량 분석 ==========================================
# ======================================================================
# A) 공급량 분석
# ======================================================================
if mode == "공급량 분석":
with st.sidebar:
st.header("데이터 불러오기")
@@ -334,7 +346,9 @@ def render_centered_table(df: pd.DataFrame, float1_cols=None, int_cols=None, ind
st.download_button("예측 결과 CSV 다운로드", data=res["pred_table"].to_csv(index=False).encode("utf-8-sig"),
file_name="citygas_supply_forecast.csv", mime="text/csv")

# =============== B) 판매량 분석(냉방용) =====================================
# ======================================================================
# B) 판매량 분석(냉방용)
# ======================================================================
else:
st.header("판매량 분석(냉방용) — 전월 16일 ~ 당월 15일 평균기온 기준")
st.write("냉방용 **판매 실적 엑셀**과 **기온 RAW(일별)**을 준비하세요.")
@@ -438,17 +452,21 @@ def period_avg(label_m: pd.Timestamp) -> float:
mask = (temp_raw["일자"]>=s)&(temp_raw["일자"]<=e)
return temp_raw.loc[mask,"기온"].mean()

        # 학습 데이터
        # 학습 데이터(선택 연도) — 기간평균기온 계산
train_sales = sales_df[sales_df["연"].isin(years_sel)].copy()
rows = [{"판매월":m, "기간평균기온": period_avg(m)} for m in train_sales["판매월"].unique()]
        sj = pd.merge(train_sales[["판매월","판매량"]], pd.DataFrame(rows), on="판매월", how="left")
        sj = pd.merge(train_sales[["판매월","판매량","연","월"]], pd.DataFrame(rows), on="판매월", how="left")
miss = sj["기간평균기온"].isna()
        if miss.any(): sj.loc[miss,"기간평균기온"] = sj.loc[miss,"판매월"].dt.month.map(fallback_by_M)
        if miss.any(): sj.loc[miss,"기간평균기온"] = sj.loc[miss,"월"].map(fallback_by_M)
sj = sj.dropna(subset=["기간평균기온","판매량"])

        x_train = sj["기간평균기온"].astype(float).values
        # 회귀 학습(그래프용 회귀선도 여기서 계산)
        x_train = sj["기간평균기온"].astype(float).values.reshape(-1,1)
y_train = sj["판매량"].astype(float).values
        _fit, r2_fit = fit_poly3_and_predict(x_train, y_train, x_train)
        poly = PolynomialFeatures(degree=3, include_bias=False)
        Xtr = poly.fit_transform(x_train)
        reg = LinearRegression().fit(Xtr, y_train)
        r2_fit = reg.score(Xtr, y_train)

# 예측 입력
f_start = pd.Timestamp(year=int(start_y), month=int(start_m), day=1)
@@ -463,33 +481,38 @@ def period_avg(label_m: pd.Timestamp) -> float:
mask = (temp_raw["일자"]>=s)&(temp_raw["일자"]<=e)
avg_period = temp_raw.loc[mask,"기온"].mean()
avg_month  = monthly_cal.loc[(monthly_cal["연"]==m.year)&(monthly_cal["월"]==m.month),"기온"].mean()
            rows.append({"연":int(m.year),"월":int(m.month),"기간평균기온":avg_period,"당월평균기온":avg_month})
            rows.append({"연":int(m.year), "월":int(m.month), "기간평균기온":avg_period, "당월평균기온":avg_month})
pred = pd.DataFrame(rows)
        for c in ["기간평균기온","당월평균기온"]]:
        for c in ["기간평균기온", "당월평균기온"]:
miss = pred[c].isna()
            if miss.any(): pred.loc[miss,c] = pred.loc[miss,"월"].map(fallback_by_M)
            if miss.any():
                pred.loc[miss, c] = pred.loc[miss, "월"].map(fallback_by_M)

        x_future = pred["기간평균기온"].astype(float).values
        y_future, _ = fit_poly3_and_predict(x_train, y_train, x_future)
        # 예측(회귀 모델 재사용)
        x_future = pred["기간평균기온"].astype(float).values.reshape(-1,1)
        y_future = reg.predict(poly.transform(x_future))
pred["예측판매량"] = np.clip(np.rint(y_future).astype(np.int64), a_min=0, a_max=None)

        # 실제/오차 (검증용)
        # 검증용(실제/오차/오차율)
actual = sales_df[["연","월","판매량"]].rename(columns={"판매량":"실제판매량"})
        out = pd.merge(pred, actual, on=["연","월"], how="left")
        out["오차"] = (out["예측판매량"] - out["실제판매량"]).astype("Int64")
        out["오차율(%)"] = np.where(
            out["실제판매량"].notna(),
            (out["오차"] / out["실제판매량"]) * 100.0,
        pred_all = pd.merge(pred, actual, on=["연","월"], how="left")
        pred_all["오차"] = (pred_all["예측판매량"] - pred_all["실제판매량"]).astype("Int64")
        pred_all["오차율(%)"] = np.where(
            pred_all["실제판매량"].notna() & (pred_all["실제판매량"] != 0),
            (pred_all["오차"].astype(float) / pred_all["실제판매량"].astype(float)) * 100.0,
np.nan
)

st.session_state["sales_result"] = {
"forecast_start": f_start,
"years_all": years_all,
"hist": sales_df.rename(columns={"판매량":"val"})[["연","월","val"]],
            "pred": out,
            "train_points": sj[["기간평균기온","판매량"]].rename(columns={"판매량":"냉방용판매량"}),
            "r2": r2_fit
            "pred_only": pred,            # 상단(요약)용
            "verify": pred_all,           # 검증표용
            "r2": r2_fit,
            "train_xy": (sj["기간평균기온"].values, sj["판매량"].values),
            "reg_poly": poly,
            "reg_model": reg
}
if "sales_years_view" not in st.session_state:
default_years = years_all[-5:] if len(years_all)>=5 else years_all
@@ -506,14 +529,14 @@ def period_avg(label_m: pd.Timestamp) -> float:
default=st.session_state.get("sales_years_view", res["years_all"][-5:]),
key="sales_years_view")

    # 월별 실적/예측 추이 그래프 (최근 5개년 실적 + 예측)
    # 실적/예측 라인 그래프(최근 5개년 + 예측)
months = list(range(1,13))
fig = plt.figure(figsize=(9,3.6)); ax = plt.gca()
for y in sorted([int(v) for v in years_view]):
s = (res["hist"][res["hist"]["연"]==y].set_index("월")["val"]).reindex(months)
ax.plot(months, s.values, label=f"{y} 실적")
pred_vals, y, m = [], int(res["forecast_start"].year), int(res["forecast_start"].month)
    P = res["pred"][["연","월","예측판매량"]].copy(); P["연"]=P["연"].astype(int); P["월"]=P["월"].astype(int)
    P = res["pred_only"][["연","월","예측판매량"]].copy(); P["연"]=P["연"].astype(int); P["월"]=P["월"].astype(int)
for _ in range(12):
row = P[(P["연"]==y)&(P["월"]==m)]
pred_vals.append(row.iloc[0]["예측판매량"] if len(row) else np.nan)
@@ -525,55 +548,42 @@ def period_avg(label_m: pd.Timestamp) -> float:
ax.set_title(f"냉방용 — Poly-3 (Train R²={res['r2']:.3f})"); ax.legend(loc="best")
plt.tight_layout(); st.pyplot(fig, clear_figure=True)

    # ── 표 1: 예측 요약(예측만 표시)
    show_pred_only = res["pred"][["연","월","당월평균기온","기간평균기온","예측판매량"]].copy()
    show_pred_only = show_pred_only.rename(columns={"기간평균기온":"기간평균기온 (m-1, 16일 ~ m15일)"})
    show_pred_only["연"] = show_pred_only["연"].astype(int).astype(str)
    show_pred_only["월"] = show_pred_only["월"].astype("Int64")

    # 1) 상단 요약표 — 예측판매량까지만
    show_pred = res["pred_only"].rename(columns={"기간평균기온":"기간평균기온 (m-1, 16일 ~ m15일)"})
    show_pred["연"] = show_pred["연"].astype(int).astype(str); show_pred["월"] = show_pred["월"].astype("Int64")
st.subheader("판매량 예측(요약)")
render_centered_table(
        show_pred_only,
        show_pred[["연","월","당월평균기온","기간평균기온 (m-1, 16일 ~ m15일)","예측판매량"]],
float1_cols=["당월평균기온","기간평균기온 (m-1, 16일 ~ m15일)"],
        int_cols=["예측판매량"],
        index=False
        int_cols=["예측판매량"], index=False
)
st.download_button(
"판매량 예측 CSV 다운로드",
        data=show_pred_only.to_csv(index=False).encode("utf-8-sig"),
        data=show_pred.to_csv(index=False).encode("utf-8-sig"),
file_name="cooling_sales_forecast.csv", mime="text/csv"
)

    # ── 표 2: 예측 검증(실제·오차·오차율)
    # 2) 검증표 — 실제/오차/오차율
st.subheader("판매량 예측 검증")
    val = res["pred"].dropna(subset=["실제판매량"]).copy()
    val["연"] = val["연"].astype(int).astype(str)
    val["월"] = val["월"].astype("Int64")
    # 오차율 표시(문자열로 %)
    val["오차율(%)"] = val["오차율(%)"].map(lambda x: "" if pd.isna(x) else f"{x:.1f}%")
    ver = res["verify"].copy()
    ver["연"] = ver["연"].astype(int).astype(str); ver["월"] = ver["월"].astype("Int64")
    # 오차율 표현(%.1f)
    ver_disp = ver.copy()
    ver_disp["오차율(%)"] = ver_disp["오차율(%)"].round(1).map(lambda x: "" if pd.isna(x) else f"{x:.1f}")
render_centered_table(
        val[["연","월","실제판매량","오차","오차율(%)"]],
        int_cols=["실제판매량","오차"],
        index=False
        ver_disp[["연","월","실제판매량","예측판매량","오차","오차율(%)"]],
        int_cols=["실제판매량","예측판매량","오차"], index=False
)

    # ── 하단: 상관관계(R²) 시각화 — 기온 vs 냉방용 실적(학습데이터)
    st.subheader("기온-냉방용 실적 상관관계 (학습 데이터)")
    tp = res["train_points"].copy()
    x = tp["기간평균기온"].astype(float).values.reshape(-1,1)
    y = tp["냉방용판매량"].astype(float).values
    poly = PolynomialFeatures(degree=3, include_bias=False)
    Xtr = poly.fit_transform(x)
    mdl = LinearRegression().fit(Xtr, y)
    r2 = mdl.score(Xtr, y)
    x_grid = np.linspace(np.nanmin(x), np.nanmax(x), 200).reshape(-1,1)
    y_grid = mdl.predict(poly.transform(x_grid))

    fig2 = plt.figure(figsize=(8.5,4)); ax2 = plt.gca()
    ax2.scatter(tp["기간평균기온"], tp["냉방용판매량"], alpha=0.6, label="학습 데이터")
    ax2.plot(x_grid.ravel(), y_grid, linestyle="--", label="3차 다항 회귀")
    ax2.set_xlabel("기간평균기온 (m-1, 16일 ~ m15일)")
    ax2.set_ylabel("냉방용 판매량")
    ax2.set_title(f"냉방용 판매량 ~ 기온 (R²={r2:.3f})")
    ax2.legend(loc="best")
    # 3) R² 시각화 — 기간평균기온 vs 냉방용 판매량(산점도 + 회귀곡선)
    st.subheader("기온–냉방용 판매량 상관관계(학습데이터)")
    x_tr, y_tr = res["train_xy"]
    poly, model = res["reg_poly"], res["reg_model"]
    xx = np.linspace(np.nanmin(x_tr), np.nanmax(x_tr), 200).reshape(-1,1)
    yy = model.predict(poly.transform(xx))
    fig2 = plt.figure(figsize=(8,4)); ax2 = plt.gca()
    ax2.scatter(x_tr, y_tr, alpha=0.5, label="학습 샘플")
    ax2.plot(xx.ravel(), yy, linestyle="--", label=f"3차 회귀선 (R²={res['r2']:.3f})")
    ax2.set_xlabel("기간평균기온(°C)"); ax2.set_ylabel("판매량")
    ax2.legend(loc="best"); ax2.grid(True, alpha=0.2)
plt.tight_layout(); st.pyplot(fig2, clear_figure=True)
