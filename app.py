
run_btn = st.button("예측 시작", type="primary")

    # 계산은 버튼 클릭 시에만 (기본 미래 월/기온의 ‘베이스’를 만든 후, 아래에서 Δ°C만 즉시 반영)
    # 계산은 버튼 클릭 시에만
if run_btn:
base = df.dropna(subset=["날짜"]).sort_values("날짜").reset_index(drop=True)
train_df = base[base["연"].isin(years_sel)].copy()
@@ -284,11 +284,9 @@ def _forecast_table_for_delta(delta: float) -> pd.DataFrame:

# 각 상품 예측
pred_rows = []
        r2_dict = {}
for col in prods:
y_train = train_df[col].astype(float).values
            y_future, r2 = fit_poly3_and_predict(x_train, y_train, x_future)
            r2_dict[col] = r2
            y_future, _ = fit_poly3_and_predict(x_train, y_train, x_future)
tmp = fut_base[["연","월"]].copy()
tmp["월평균기온"] = x_future
tmp["상품"] = col
@@ -315,7 +313,7 @@ def _forecast_table_for_delta(delta: float) -> pd.DataFrame:

return pivot_total

    # ── 출력(표: Normal → Best → Conservative)
    # ── 표 출력( Normal → Best → Conservative )
st.markdown("### Normal")
tbl_n = _forecast_table_for_delta(d_norm)
render_centered_table(tbl_n, float1_cols=["월평균기온"], int_cols=[c for c in tbl_n.columns if c not in ["연","월","월평균기온"]], index=False)
@@ -331,6 +329,51 @@ def _forecast_table_for_delta(delta: float) -> pd.DataFrame:
st.download_button("예측 결과 CSV 다운로드 (Normal)", data=tbl_n.to_csv(index=False).encode("utf-8-sig"),
file_name="citygas_supply_forecast_normal.csv", mime="text/csv")

    # ── 그래프( Normal Δ°C 기준 )
    st.markdown("### 그래프 (Normal 기준)")
    years_all_for_plot = sorted([int(v) for v in base["연"].dropna().unique()])
    default_years = years_all_for_plot[-5:] if len(years_all_for_plot) >= 5 else years_all_for_plot
    years_view = st.multiselect(
        "표시할 실적 연도",
        options=years_all_for_plot,
        default=st.session_state.get("supply_years_view", default_years),
        key="supply_years_view"
    )

    # Normal Δ°C로 예측(그래프용)
    x_future_norm = (fut_base["temp"] + float(d_norm)).astype(float).values

    for prod in prods:
        # 학습/예측
        y_train = train_df[prod].astype(float).values
        y_future_norm, r2_train = fit_poly3_and_predict(x_train, y_train, x_future_norm)

        P = fut_base[["연","월"]].copy()
        P["pred"] = np.clip(np.rint(y_future_norm).astype(np.int64), a_min=0, a_max=None)

        fig = plt.figure(figsize=(9,3.6)); ax = plt.gca()

        # 실적
        for y in sorted([int(v) for v in years_view]):
            s = (base.loc[base["연"]==y, ["월", prod]].set_index("월")[prod]).reindex(months)
            ax.plot(months, s.values, label=f"{y} 실적")

        # 예측(12개월 시퀀스)
        pred_vals = []
        y, m = int(mats["start_ts"].year), int(mats["start_ts"].month)
        P2 = P.copy(); P2["연"]=P2["연"].astype(int); P2["월"]=P2["월"].astype(int)
        for _ in range(12):
            row = P2[(P2["연"]==y)&(P2["월"]==m)]
            pred_vals.append(row.iloc[0]["pred"] if len(row) else np.nan)
            if m==12: y+=1; m=1
            else: m+=1
        ax.plot(months, pred_vals, linestyle="--", label="예측(Normal)")

        ax.set_xlim(1,12); ax.set_xticks(months); ax.set_xticklabels([f"{mm}월" for mm in months])
        ax.set_xlabel("월"); ax.set_ylabel("양")
        ax.set_title(f"{prod} — Poly-3 (Train R²={r2_train:.3f})"); ax.legend(loc="best")
        plt.tight_layout(); st.pyplot(fig, clear_figure=True)

# =============== B) 판매량 분석(냉방용) =====================================
else:
st.header("판매량 분석(냉방용) — 전월 16일 ~ 당월 15일 평균기온 기준")
