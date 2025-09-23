    def _monthly_series_for(prod: str) -> pd.Series:
        """
        선택한 상품의 월별 시계열을 '연속 월(MS)'로 강제하고 수치형으로 반환.
        """
        s = base[["연","월",prod]].dropna(subset=["연","월"])
        s["날짜"] = pd.to_datetime(s["연"].astype(int).astype(str) + "-" + s["월"].astype(int).astype(str) + "-01")
        s = s.sort_values("날짜").set_index("날짜")[prod].astype(float)

        # 연속 월로 강제 (빈 달 생성) + 숫자형 유지
        s = s.asfreq("MS")

        return s

    def _prepare_train_series(ts: pd.Series, years_sel: list[int]) -> pd.Series:
        """
        학습연도만 필터링 후 결측을 제거:
        1) 시간보간(interpolate('time')) → 2) ffill/bfill → 3) 0 대체
        마지막에 float 시리즈로 반환.
        """
        train = ts[ts.index.year.isin(years_sel)]

        if train.isna().any():
            train = (train.interpolate(method="time", limit_direction="both")
                          .fillna(method="ffill")
                          .fillna(method="bfill")
                          .fillna(0.0))

        # 혹시 전부 0이거나 길이가 너무 짧을 때 방어
        if len(train.dropna()) < 6:  # 최소 6개월 미만이면 시계열 모델 피함
            return pd.Series(dtype=float)

        return train.astype(float)

    def _fore_arima_yearsum(prod: str, target_years: list[int]) -> dict:
        if not _HAS_SM:
            return {y: np.nan for y in target_years}

        ts = _monthly_series_for(prod)
        train = _prepare_train_series(ts, years_sel)
        if train.empty:
            return {y: np.nan for y in target_years}

        # 간단 후보 중 AIC 최소 선택
        candidates = [(1,1,0), (0,1,1), (1,1,1)]
        best_mdl, best_aic = None, np.inf
        for order in candidates:
            try:
                mdl = ARIMA(train, order=order).fit()
                if mdl.aic < best_aic:
                    best_aic, best_mdl = mdl.aic, mdl
            except Exception:
                continue
        if best_mdl is None:
            return {y: np.nan for y in target_years}

        last_train_year = int(train.index[-1].year)
        steps = 12 * max(1, (max(target_years) - last_train_year))
        f = best_mdl.forecast(steps=steps)

        fut = f.copy()
        fut.index = pd.date_range(start=train.index[-1] + pd.offsets.MonthBegin(1),
                                  periods=len(fut), freq="MS")
        df_year = fut.groupby(fut.index.year).sum()
        return {y: float(df_year.get(y, np.nan)) for y in target_years}

    def _fore_sarima_yearsum(prod: str, target_years: list[int]) -> dict:
        if not _HAS_SM:
            return {y: np.nan for y in target_years}

        ts = _monthly_series_for(prod)
        train = _prepare_train_series(ts, years_sel)
        if train.empty:
            return {y: np.nan for y in target_years}

        try:
            mdl = SARIMAX(
                train,
                order=(1,1,1),
                seasonal_order=(1,1,1,12),
                enforce_stationarity=False,
                enforce_invertibility=False,
            ).fit(disp=False)
        except Exception:
            return {y: np.nan for y in target_years}

        last_train_year = int(train.index[-1].year)
        steps = 12 * max(1, (max(target_years) - last_train_year))
        f = mdl.forecast(steps=steps)

        fut = f.copy()
        fut.index = pd.date_range(start=train.index[-1] + pd.offsets.MonthBegin(1),
                                  periods=len(fut), freq="MS")
        df_year = fut.groupby(fut.index.year).sum()
        return {y: float(df_year.get(y, np.nan)) for y in target_years}
