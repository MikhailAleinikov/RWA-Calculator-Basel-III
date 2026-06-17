import pandas as pd
def validate(df):
    f = pd.DataFrame(index=df.index)
    f['bad_ead']          = (df.exposure_amount <= 0) | df.exposure_amount.isna()
    re = df.counterparty_type == 'individual'
    ltv = df.loan_amount / df.property_value
    f['re_missing_value'] = re & df.property_value.isna()
    f['ltv_gt_1']         = re & (ltv > 1)
    f['pd_oob']           = df.PD.notna()  & ~df.PD.between(0, 1)
    f['lgd_oob']          = df.LGD.notna() & ~df.LGD.between(0, 1)
    f['bad_maturity']     = df.maturity_years <= 0
    f['unrated']          = df.external_rating.isna()
    f['dpd_default_mismatch'] = (df.days_past_due > 90) & ~df.default_flag

    df = df.join(f)
    df.loc[df.dpd_default_mismatch, 'default_flag'] = True          # correct
    df['excluded'] = f[['bad_ead','re_missing_value',
                        'ltv_gt_1','bad_maturity']].any(axis=1)     # exclude
    return df

def quality_summary(df, checks):
    return (pd.DataFrame([
        {'check': c, 'n': int(df[c].sum()),
         'eur': df.loc[df[c], 'exposure_amount'].clip(lower=0).sum()}
        for c in checks])
        .sort_values('eur', ascending=False))