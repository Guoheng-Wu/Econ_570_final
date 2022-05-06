def acf_plot(data):
    plot_acf(data)
    plt.title('Autocorrelation')
    plt.xlabel('Lag')
    plt.ylabel('ACF')


def pacf_plot(data):
    if len(data) <= 40:
        plot_pacf(data, lags=(len(data) / 2 - 1))
    else:
        plot_pacf(data)
    plt.title('Partial Autocorrelation')
    plt.xlabel('Lag')
    plt.ylabel('PACF')


def searchARMA(data, exog, max_p, max_q):
    aic = np.zeros((max_p, max_q))
    bic = np.zeros((max_p, max_q))
    aic_test = 1000000
    p = 0
    q = 0
    for i in range(max_p):
        for j in range(max_q):
            try:
                if len(exog) == len(data):
                    model = ARMA(data, (i, j), exog=exog)
                else:
                    model = ARMA(data, (i, j))
                res = model.fit(disp=0, trend='nc')
                aic[i, j] = res.aic
                if aic_test > aic[i, j]:
                    aic_test = aic[i, j]
                    p = i
                    q = j
                bic[i, j] = res.bic
                print('p:', i, ' q:', j, ' aic:', aic[i, j], ' bic:', bic[i, j])
            except:
                continue
    print(p, q)


def get_forecast(ARMA_res, log):
    l = len(test)
    start_index = length
    end_index = start_index + l
    f = ARMA_res.predict(start=start_index, end=end_index)
    if log == False:
        forecast = test[0]
        F = [forecast]
        for i in range(1, l):
            forecast = forecast + f[i]
            F.append(forecast)
            i = i + 1
    else:
        forecast = np.log(test[0])
        F = [forecast]
        for i in range(1, l):
            forecast = forecast + f[i]
            F.append(forecast)
            i = i + 1

    return F


def plot_diagnostics(data):
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 6))
    ax1.hist(data, bins=40, color='m', density=True)
    mu = data.mean()
    sigma = data.std()
    x = np.linspace(mu - 3 * sigma, mu + 3 * sigma, 2419)
    ax1.plot(x, stats.norm.pdf(x, mu, sigma), linewidth=5)
    ax1.grid()
    ax1.set_title("Hist Resid")

    ax2.plot(data)
    ax2.grid()
    ax2.set_title("Resid")

    fig = plot_acf(data, lags=40, zero=False, ax=ax3, use_vlines=True)
    ax3.grid()

    fig = sm.qqplot(data, line='q', ax=ax4)
    ax4.grid()

    plt.tight_layout()


def createXY(dataset, n_past):
    dataX = []
    dataY = []
    for i in range(n_past, len(dataset)):
        dataX.append(dataset[i - n_past:i, 0:dataset.shape[1]])
        dataY.append(dataset[i, 0])
    return np.array(dataX), np.array(dataY)