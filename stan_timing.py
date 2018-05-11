"""Timing how long it takes to extract generated quantities in pystan."""
import matplotlib.pyplot as plt
import numpy as np
import pystan

import timeit

STAN_CODE = """
data {
    int<lower=0> n;
}
generated quantities {
    vector[n] theta;
    for (j in 1:n) theta[j] = normal_rng(0., 1.);
}
"""
MODEL = pystan.StanModel(model_code=STAN_CODE)


def time_fit(n=10):
    """Time actually generating the data."""
    code = r"""fit = MODEL.sampling(
        data={'n': %s},
        seed=42,
        iter=1,
        chains=1,
        algorithm="Fixed_param"
    )
    """ % n
    t = timeit.timeit(
        code,
        number=1,
        globals=globals()
    )
    return t


def time_extract(n=10):
    """Time the extract method for varying n."""
    setup_code = r"""fit = MODEL.sampling(
        data={'n': %s},
        seed=42,
        iter=1,
        chains=1,
        algorithm="Fixed_param"
    )
    """ % n
    t = timeit.timeit(
        "fit.extract()",
        setup=setup_code,
        number=1,
        globals=globals(),
    )
    return t


def time_hack(n=10):
    """Time how long it takes to get the data out using hack."""
    setup_code = r"""fit = MODEL.sampling(
        data={'n': %s},
        seed=42,
        iter=1,
        chains=1,
        algorithm="Fixed_param"
    )
    """ % n
    t = timeit.timeit(
        "[float(i) for i in fit.sim['samples'][0].chains.values()][:-1]",
        setup=setup_code,
        number=1,
        globals=globals()
    )
    return t


def time_not_permuted(n=10):
    """Time how long it takes to get the data out using suggested fix."""
    setup_code = r"""fit = MODEL.sampling(
        data={'n': %s},
        seed=42,
        iter=1,
        chains=1,
        algorithm="Fixed_param"
    )
    """ % n
    t = timeit.timeit(
        "fit.extract('theta', permuted=False)",
        setup=setup_code,
        number=1,
        globals=globals()
    )
    return t


def main():
    """Make a graph comparing the different methods."""
    n = np.arange(1000, 55000, 5000)
    tf = [time_fit(ni) for ni in n]
    te = [time_extract(ni) for ni in n]
    th = [time_hack(ni) for ni in n]
    tnp = [time_not_permuted(ni) for ni in n]
    fig, ax = plt.subplots()
    ax.plot(n, tf, label="sampling time")
    ax.plot(n, te, label="`fit.extract()` time")
    ax.plot(n, th, label="[float(i) for i in fit.sim['samples'][0].chains.values()][:-1] time")
    ax.plot(n, tnp, label="fit.extract('theta', permuted=False) time")
    ax.set_ylim((1e-4, 1e4))
    ax.set_xlabel("size of generated vector")
    ax.set_ylabel("time (seconds)")
    ax.set_yscale("log")
    ax.legend(loc='upper left')
    fig.savefig("pystan_timings.pdf")


if __name__ == "__main__":
    main()
