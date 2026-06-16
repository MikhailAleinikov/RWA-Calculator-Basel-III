import numpy as np, pandas as pd

def generate_portfolio(nuber_of_loans: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed) # We use the same seed throughout the generation process to make it reproducible
    N = nuber_of_loans
    types = rng.choice(
        ['retail', 'corporate', 'individual', 'institution', 'sovereign'],
        size=N, p=[0.40, 0.30, 0.18, 0.08, 0.04]) # individual - residential-mortgage, retail - personal loans

    mu_by_type = {'retail': 9.0, 'individual': 12.3, 'corporate': 13.0,
                  'institution': 15.5, 'sovereign': 17.0}   # natural logarithms of means in EUR
    sigma_by_type = {'retail': 0.8, 'individual': 0.5, 'corporate': 1.1,
                     'institution': 1.0, 'sovereign': 1.2}
    mu  = np.array([mu_by_type[t]  for t in types])
    sig = np.array([sigma_by_type[t] for t in types])
    ead = rng.lognormal(mu, sig)        # the log-normal distribution gives fat tails on the right

    grades = ['AAA','AA','A','BBB','BB','B','CCC']
    rating_dist = {
        'sovereign':   [.30,.30,.20,.12,.05,.02,.01],
        'institution': [.10,.25,.30,.20,.10,.04,.01],
        'corporate':   [.03,.08,.20,.32,.22,.10,.05],
    }
    rating = np.full(N, None, dtype=object)
    for t, dist in rating_dist.items():
        m = types == t
        rating[m] = rng.choice(grades, size=m.sum(), p=dist)
    # retail + individual stay None -> unrated; also knock a few rated ones to None to make the initial data impaired
    unrated_noise = rng.random(N) < 0.05
    rating[unrated_noise & np.isin(types, ['corporate','institution'])] = None

    maturity = np.empty(N)
    maturity[types=='individual']  = rng.uniform(15, 30, (types=='individual').sum())
    maturity[types=='retail']      = rng.uniform(0.5, 5, (types=='retail').sum())
    maturity[types=='corporate']   = rng.gamma(2, 1.8, (types=='corporate').sum()).clip(0.5, 15)
    maturity[types=='institution'] = rng.uniform(0.25, 7, (types=='institution').sum())
    maturity[types=='sovereign']   = rng.uniform(1, 30, (types=='sovereign').sum())

    property_value = np.full(N, np.nan)
    loan_amount    = np.full(N, np.nan)
    re_mask = types == 'individual'
    n_re = re_mask.sum()
    pv = rng.lognormal(12.6, 0.45, n_re)                     # property values
    ltv = rng.beta(6, 2, n_re) * 0.9 + 0.2                   # LTV centred ~0.75
    property_value[re_mask] = pv
    loan_amount[re_mask]    = pv * ltv

    collateral_type  = np.full(N, None, dtype=object)
    collateral_value = np.full(N, np.nan)
    has_coll = (rng.random(N) < 0.35) & ~re_mask
    collateral_type[has_coll]  = rng.choice(['cash','securities','guarantee'], has_coll.sum())
    collateral_value[has_coll] = ead[has_coll] * rng.uniform(0.2, 1.3, has_coll.sum())

    dpd = np.zeros(N, dtype=int)
    delinquent = rng.random(N) < 0.06
    dpd[delinquent] = rng.integers(1, 200, delinquent.sum())
    default_flag = dpd > 90
    # we corrupt ~15% of the genuine 90+ rows to break the invariant on purpose
    break_mask = (dpd > 90) & (rng.random(N) < 0.15)
    default_flag[break_mask] = False

    pd_anchor = {'AAA':0.0003,'AA':0.0008,'A':0.002,'BBB':0.006,
                 'BB':0.02,'B':0.07,'CCC':0.20, None:0.03}
    PD = np.array([pd_anchor[r] for r in rating]) * rng.lognormal(0, 0.4, N)
    PD = PD.clip(1e-5, 0.999)
    LGD = np.where(has_coll, 0.30, 0.45) * rng.lognormal(0, 0.15, N)
    LGD = LGD.clip(0.01, 0.99)
    PD[default_flag] = 1.0     # defaulted names: PD = 1
    PD[rng.uniform(range(N))] = rng.uniform(1.2, 5)

    df = pd.DataFrame({
        'exposure_id': [f'EXP{i:06d}' for i in range(N)],
        'counterparty_type': types,
        'exposure_amount': ead.round(2),
        'external_rating': rating,
        'maturity_years': maturity.round(2),
        'property_value': property_value.round(2),
        'loan_amount': loan_amount.round(2),
        'collateral_type': collateral_type,
        'collateral_value': collateral_value.round(2),
        'days_past_due': dpd,
        'default_flag': default_flag,
        'PD': PD.round(6),
        'LGD': LGD.round(4),
    })
    return df