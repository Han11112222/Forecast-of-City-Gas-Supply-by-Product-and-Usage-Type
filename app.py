# ===========================================================
# D) 추천 학습연도 — 상품 선택 → R²/상관도 분석 → 추천 연도 저장
# ===========================================================
def render_recommend_learn_years():
    title_with_icon("🎯", "추천 학습연도 도우미", "h2")
    st.caption("상품(용도)별로 **기온↔공급량** 관계를 월단위로 분석해서 R²·상관도(r)를 보여주고, 기준을 넘어서는 연도를 추천해줄게.")

    # ── 좌측: 데이터 로딩(실적만 필요)
    with st.sidebar:
        title_with_icon("📥", "데이터 불러오기", "h3", small=True)
        src = st.radio("📦 방식", ["Repo 내 파일 사용", "파일 업로드"], index=0, key="reco_src")
        df = None
        if src == "Repo 내 파일 사용":
            data_dir = Path("data"); data_dir.mkdir(exist_ok=True)
            repo_files = sorted([str(p) for p in data_dir.glob("*.xlsx")])
            if repo_files:
                default_idx = next((i for i, p in enumerate(repo_files)
                                    if ("상품별공급량" in Path(p).stem) or ("공급량" in Path(p).stem)), 0)
                file_choice = st.selectbox("📄 실적 파일(Excel)", repo_files, index=default_idx,
                                           format_func=lambda p: Path(p).name, key="reco_file")
                df = read_excel_sheet(file_choice, prefer_sheet="데이터")
            else:
                st.info("📂 data 폴더에 엑셀 파일이 없습니다. 업로드로 진행하세요.")
        else:
            up = st.file_uploader("📄 실적 엑셀 업로드(xlsx) — '데이터' 시트", type=["xlsx"], key="reco_up")
            if up is not None:
                df = read_excel_sheet(up, prefer_sheet="데이터")

    if df is None or df.empty:
        st.info("👈 좌측에서 실적 파일을 선택/업로드하면 분석을 시작할 수 있어.")
        st.stop()

    df = df.dropna(subset=["연","월"]).copy()
    df["연"] = df["연"].astype(int); df["월"] = df["월"].astype(int)
    temp_col = detect_temp_col(df)
    if temp_col is None:
        st.error("🌡️ 실적 파일에서 '평균기온/기온' 열을 찾지 못했어. 데이터에 기온 열을 넣어줘.")
        st.stop()

    product_cols = guess_product_cols(df)
    title_with_icon("📦", "상품(용도) 선택 — 단추로 여러 개 고르기", "h3", small=True)

    # 버튼(토글)로 선택
    selected_products = []
    cols = st.columns(4)
    for i, p in enumerate(product_cols):
        with cols[i % 4]:
            on = st.toggle(p, value=False, key=f"reco_toggle_{p}")
        if on: selected_products.append(p)

    st.markdown(" ".join([f"<span class='badge'>{p}</span>" for p in selected_products]), unsafe_allow_html=True)
    run = st.button("✅ 선택 완료 · 분석 시작", type="primary", key="reco_run")

    if not run or not selected_products:
        st.stop()

    # 기준값(추천 기준)
    st.markdown("##### 추천 기준")
    c1, c2 = st.columns(2)
    with c1:
        thr_r = st.slider("연도별 |r| 임계값 (절대상관)", 0.0, 1.0, 0.6, 0.05)
    with c2:
        min_pts = st.slider("연도별 최소 월 데이터 개수", 6, 12, 10, 1)

    base = df.sort_values(["연","월"]).reset_index(drop=True)

    def year_stats_for(prod:str):
        rows = []
        for y, g in base[["연","월",temp_col,prod]].dropna().groupby("연"):
            if len(g) < min_pts:
                rows.append((y, len(g), np.nan, np.nan))
                continue
            x = g[temp_col].astype(float).values
            v = g[prod].astype(float).values
            # 선형 상관계수 r
            r = float(np.corrcoef(x, v)[0,1]) if np.std(x)>0 and np.std(v)>0 else np.nan
            # Poly-3 R² (연도별 적합)
            try:
                _, r2, _, _ = fit_poly3_and_predict(x, v, x)
            except Exception:
                r2 = np.nan
            rows.append((int(y), int(len(g)), r, r2))
        out = pd.DataFrame(rows, columns=["연","표본수","상관계수 r","R²(Poly-3)"]).sort_values("연")
        return out

    # 선택된 상품들 Loop
    recommended_union = set()
    for prod in selected_products:
        title_with_icon("📊", f"{prod} — 전체 상관/적합(월단위)", "h3", small=True)

        all_ok = base[[temp_col, prod]].dropna()
        x_all = all_ok[temp_col].astype(float).values
        y_all = all_ok[prod].astype(float).values

        # 전체 Poly-3 곡선 + R², r
        xx = np.linspace(np.nanmin(x_all) - 1, np.nanmax(x_all) + 1, 200)
        yhat, r2_all, model_all, _ = fit_poly3_and_predict(x_all, y_all, xx)
        r_all = float(np.corrcoef(x_all, y_all)[0,1]) if np.std(x_all)>0 and np.std(y_all)>0 else np.nan

        if go is None:
            fig, ax = plt.subplots(figsize=(10,4.6))
            ax.scatter(x_all, y_all, s=28, alpha=.7)
            ax.plot(xx, yhat, lw=2.5, color="#1f77b4", label=f"Poly-3 (R²={r2_all:.3f})")
            ax.set_xlabel("기온 (℃)"); ax.set_ylabel("공급량")
            ax.grid(alpha=.25); ax.legend(loc="best")
            ax.text(0.02, 0.06, f"r={r_all:.3f}\n{poly_eq_text(model_all)}",
                    transform=ax.transAxes, fontsize=10,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=.8))
            st.pyplot(fig)
        else:
            # 연도별로 색 구분한 산점도
            fig = go.Figure()
            for y, g in base[["연", temp_col, prod]].dropna().groupby("연"):
                fig.add_trace(go.Scatter(
                    x=g[temp_col], y=g[prod], mode="markers", name=str(int(y)),
                    hovertemplate="연 %{text} | x=%{x:.2f}℃ | y=%{y:,}",
                    text=[int(y)]*len(g)
                ))
            fig.add_trace(go.Scatter(x=xx, y=yhat, mode="lines",
                                     name=f"Poly-3 (R²={r2_all:.3f}, r={r_all:.3f})"))
            fig.update_layout(xaxis_title="기온 (℃)", yaxis_title="공급량",
                              legend=dict(orientation="h", yanchor="bottom", y=-0.18, xanchor="left", x=0),
                              margin=dict(t=60,b=110,l=40,r=20))
            st.plotly_chart(fig, use_container_width=True)

        # 연도별 통계표 + 바차트
        stats_df = year_stats_for(prod)
        st.markdown("###### 연도별 상관/적합 표")
        render_centered_table(stats_df, float1_cols=["상관계수 r","R²(Poly-3)"], int_cols=["표본수"], index=False)

        if go is None:
            figb, axb = plt.subplots(figsize=(10,3.2))
            axb.bar(stats_df["연"].astype(int), stats_df["R²(Poly-3)"])
            axb.set_ylim(0,1); axb.set_ylabel("R²"); axb.set_xlabel("연")
            st.pyplot(figb)
        else:
            figb = go.Figure()
            figb.add_trace(go.Bar(x=stats_df["연"], y=stats_df["R²(Poly-3)"], name="R²"))
            figb.add_trace(go.Scatter(x=stats_df["연"], y=stats_df["상관계수 r"], name="r", mode="lines+markers"))
            figb.update_layout(yaxis=dict(range=[0,1]), legend=dict(orientation="h", y=-0.2),
                               margin=dict(t=30,b=100,l=40,r=20), xaxis_title="연")
            st.plotly_chart(figb, use_container_width=True)

        # 추천 로직
        cand = stats_df[(stats_df["표본수"] >= min_pts) & (stats_df["상관계수 r"].abs() >= thr_r)]["연"].astype(int).tolist()
        if not cand:
            # 보수: |r| Top 5
            tmp = stats_df.dropna(subset=["상관계수 r"]).copy()
            tmp["absr"] = tmp["상관계수 r"].abs()
            cand = tmp.sort_values("absr", ascending=False)["연"].astype(int).head(5).tolist()

        st.markdown(f"**👉 {prod} 추천 학습연도:** " + ", ".join(map(str, cand)) if cand else "추천 가능 연도가 충분치 않아.")
        recommended_union.update(cand)

    # 전체 선택 적용
    st.markdown("---")
    reco_sorted = sorted(set(int(y) for y in recommended_union))
    st.markdown("#### 모든 선택용도 합산 추천 연도")
    st.markdown(" ".join([f"<span class='badge'>{y}</span>" for y in reco_sorted]) or "없음", unsafe_allow_html=True)

    if st.button("💾 공급량 예측 탭의 기본 학습연도로 **적용하기**", type="primary", key="apply_reco"):
        if reco_sorted:
            st.session_state["reco_years"] = reco_sorted
            st.success(f"적용 완료: {', '.join(map(str, reco_sorted))}")
        else:
            st.info("추천 목록이 비어 있어 적용할 값이 없어.")

