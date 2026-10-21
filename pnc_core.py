import re
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

CATEGORICAL_RAW = ['v024', 'v025', 'v130', 'v151', 'v106', 'v190', 'v701', 'v714',
                   'v157', 'v158', 'v159', 'v743a', 'v467d', 'b4_01', 'm15_1']
NUMERIC_RAW = ['v012', 'v136', 'bord_01', 'b19_01']
RAW_INPUT_COLS = ['survey_year'] + NUMERIC_RAW + ['m14_1'] + CATEGORICAL_RAW

LABEL_FIXES = {
    'v024': {'barisal': 'barishal', 'chittagong': 'chattogram'},
    'v130': {'buddhism': 'buddhist'},
    'm15_1': {"respondent's home": 'home', 'her home': 'home',
              'upazilla health complex': 'upazila health complex',
              'other private medical sector': 'other private medical',
              'specialized government hospital': 'specialized govt hospital'},
    'v701': {"don't know": None, 'dk': None},
}
EDU = {'no education': 0, 'primary': 1, 'secondary': 2, 'higher': 3}
WEALTH = {'poorest': 0, 'poorer': 1, 'middle': 2, 'richer': 3, 'richest': 4}
MEDIA = {'not at all': 0, 'less than once a week': 1, 'at least once a week': 2, 'almost every day': 3}
BINARY_SPEC = {'survey_year': (2022, 2017), 'v025': ('rural', 'urban'), 'v714': ('yes', 'no'),
               'v467d': ('big problem', 'not a big problem'), 'v151': ('female', 'male'),
               'b4_01': ('female', 'male')}

FE_COLS = ['fe_age_at_birth', 'fe_teen_birth', 'fe_high_parity', 'fe_anc4plus', 'fe_no_anc',
           'fe_place_group', 'fe_facility_delivery', 'fe_facility_x_2022', 'fe_media_index', 'fe_media_any',
           'fe_decision_involved', 'fe_edu_gap', 'fe_secondary_plus', 'fe_poor', 'fe_rural_poor',
           'fe_home_rural', 'fe_vulnerability']
NUMERIC_FEATS = ['v012', 'v136', 'bord_01', 'm14_1', 'fe_age_at_birth', 'fe_media_index',
                 'fe_edu_gap', 'fe_vulnerability']
WINSOR_FEATS = ['v012', 'v136', 'bord_01', 'm14_1']
ORDINAL_FEATS = {'v106': EDU, 'v701': EDU, 'v190': WEALTH, 'v157': MEDIA, 'v158': MEDIA, 'v159': MEDIA}
FLAG_FEATS = ['fe_teen_birth', 'fe_high_parity', 'fe_anc4plus', 'fe_no_anc', 'fe_facility_delivery',
              'fe_facility_x_2022', 'fe_media_any', 'fe_decision_involved', 'fe_secondary_plus', 'fe_poor',
              'fe_rural_poor', 'fe_home_rural']
NOMINAL_FEATS = ['v024', 'v130', 'v743a', 'm15_1', 'fe_place_group']

FEATURE_GROUPS = {
    'Survey year': ['survey_year', 'fe_facility_x_2022'],
    'Place of delivery': ['m15_1', 'fe_place_group', 'fe_facility_delivery', 'fe_home_rural'],
    'Antenatal care': ['m14_1', 'fe_anc4plus', 'fe_no_anc'],
    'Socioeconomic': ['v106', 'v701', 'v190', 'v714', 'fe_secondary_plus', 'fe_poor', 'fe_rural_poor', 'fe_edu_gap'],
    'Media exposure': ['v157', 'v158', 'v159', 'fe_media_index', 'fe_media_any'],
    'Demographic/reproductive': ['v012', 'v136', 'bord_01', 'b19_01', 'b4_01', 'v130', 'v151',
                                 'fe_age_at_birth', 'fe_teen_birth', 'fe_high_parity'],
    'Geography': ['v024', 'v025'],
    'Empowerment/access': ['v743a', 'v467d', 'fe_decision_involved', 'fe_vulnerability'],
}

VAR_NAMES = {
    'survey_year': 'Survey year (2022)', 'v012': "Mother's age", 'v024': 'Division', 'v025': 'Rural residence',
    'v130': 'Religion', 'v136': 'Household size', 'v151': 'Female household head', 'v106': "Mother's education",
    'v190': 'Wealth index', 'v701': "Husband's education", 'v714': 'Mother working', 'v157': 'Newspaper',
    'v158': 'Radio', 'v159': 'TV', 'v743a': 'Decision on own health care', 'v467d': 'Distance a big problem',
    'b4_01': 'Child female', 'bord_01': 'Birth order', 'b19_01': 'Child age (months)', 'm14_1': 'ANC visits',
    'm15_1': 'Place of delivery', 'fe_age_at_birth': 'Age at birth (FE)', 'fe_teen_birth': 'Teen birth <20 (FE)',
    'fe_high_parity': 'Birth order >=4 (FE)', 'fe_anc4plus': 'ANC >=4 (FE)', 'fe_no_anc': 'No ANC (FE)',
    'fe_place_group': 'Delivery place group (FE)', 'fe_facility_delivery': 'Facility delivery (FE)',
    'fe_facility_x_2022': 'Facility delivery x 2022 (FE)', 'fe_media_index': 'Media index 0-9 (FE)',
    'fe_media_any': 'Any media (FE)', 'fe_decision_involved': 'Involved in health decision (FE)',
    'fe_edu_gap': 'Education gap wife-husband (FE)', 'fe_secondary_plus': 'Secondary+ education (FE)',
    'fe_poor': 'Poor/poorest (FE)', 'fe_rural_poor': 'Rural and poor (FE)', 'fe_home_rural': 'Home birth and rural (FE)',
    'fe_vulnerability': 'Vulnerability score 0-7 (FE)',
}


def group_of(var):
    for g, members in FEATURE_GROUPS.items():
        if var in members:
            return g
    return 'Other'


def norm_label(x):
    try:
        if pd.isna(x):
            return np.nan
    except (TypeError, ValueError):
        pass
    s = re.sub(r'\s+', ' ', str(x).strip().lower().replace('\u2019', "'"))
    return np.nan if s in ('', 'nan', 'none') else s


def parse_anc_visits(x):
    if isinstance(x, (int, float, np.integer, np.floating)):
        v = float(x)
        return np.nan if (np.isnan(v) or v < 0 or v >= 90) else v
    s = norm_label(x)
    if not isinstance(s, str) or 'know' in s or s == 'dk':
        return np.nan
    if s.startswith('no ') or s in ('none', 'no visits'):
        return 0.0
    m = re.search(r'\d+', s)
    return (np.nan if float(m.group()) >= 90 else float(m.group())) if m else np.nan


def harmonize_df(X):
    X = pd.DataFrame(X).copy()
    for c in CATEGORICAL_RAW:
        if c in X.columns:
            fx = LABEL_FIXES.get(c, {})
            s = X[c].astype(object).map(norm_label)
            X[c] = s.map(lambda v: fx.get(v, v) if isinstance(v, str) else v)
    if 'm14_1' in X.columns:
        X['m14_1'] = pd.to_numeric(X['m14_1'].astype(object).map(parse_anc_visits), errors='coerce')
    for c in NUMERIC_RAW + ['survey_year']:
        if c in X.columns:
            X[c] = pd.to_numeric(X[c], errors='coerce')
    return X


def place_group(label):
    if not isinstance(label, str):
        return np.nan
    if 'home' in label:
        return 'home'
    if 'ngo' in label:
        return 'ngo'
    if 'private' in label:
        return 'private'
    if label.startswith('other'):
        return 'other'
    return 'public'


def _flag(cond, valid):
    return pd.Series(np.where(np.asarray(valid), np.asarray(cond).astype(float), np.nan), index=cond.index)


def engineer_features(X):
    X = pd.DataFrame(X).copy()
    idx = X.index

    def col(c):
        return X[c] if c in X.columns else pd.Series(np.nan, index=idx, dtype=object)

    edu = pd.to_numeric(col('v106').map(EDU), errors='coerce')
    hedu = pd.to_numeric(col('v701').map(EDU), errors='coerce')
    wealth = pd.to_numeric(col('v190').map(WEALTH), errors='coerce')
    age = pd.to_numeric(col('v012'), errors='coerce')
    child_m = pd.to_numeric(col('b19_01'), errors='coerce')
    bord = pd.to_numeric(col('bord_01'), errors='coerce')
    anc = pd.to_numeric(col('m14_1'), errors='coerce')
    year = pd.to_numeric(col('survey_year'), errors='coerce')

    X['fe_age_at_birth'] = age - child_m / 12.0
    X['fe_teen_birth'] = _flag(X['fe_age_at_birth'] < 20, X['fe_age_at_birth'].notna())
    X['fe_high_parity'] = _flag(bord >= 4, bord.notna())
    X['fe_anc4plus'] = _flag(anc >= 4, anc.notna())
    X['fe_no_anc'] = _flag(anc == 0, anc.notna())
    pg = col('m15_1').map(place_group)
    X['fe_place_group'] = pg
    fac = pg.isin(['public', 'private', 'ngo'])
    X['fe_facility_delivery'] = _flag(fac, pg.notna())
    X['fe_facility_x_2022'] = _flag(fac & (year == 2022), pg.notna() & year.notna())
    media = pd.concat([pd.to_numeric(col(c).map(MEDIA), errors='coerce') for c in ('v157', 'v158', 'v159')], axis=1)
    X['fe_media_index'] = media.sum(axis=1, min_count=1)
    X['fe_media_any'] = _flag(X['fe_media_index'] > 0, X['fe_media_index'].notna())
    dec = col('v743a')
    X['fe_decision_involved'] = _flag(dec.isin(['respondent alone', 'respondent and husband/partner']), dec.notna())
    X['fe_edu_gap'] = edu - hedu
    X['fe_secondary_plus'] = _flag(edu >= 2, edu.notna())
    X['fe_poor'] = _flag(wealth <= 1, wealth.notna())
    rural = col('v025').eq('rural')
    rv = col('v025').notna()
    X['fe_rural_poor'] = _flag(rural & (wealth <= 1), rv & wealth.notna())
    X['fe_home_rural'] = _flag(rural & pg.eq('home'), rv & pg.notna())
    comps = pd.concat([edu.eq(0), wealth.le(1), rural, X['fe_teen_birth'].eq(1), X['fe_high_parity'].eq(1),
                       col('v467d').eq('big problem'), X['fe_media_index'].eq(0)], axis=1).astype(float)
    X['fe_vulnerability'] = comps.sum(axis=1)
    return X


def sanitize(name):
    return re.sub(r'[^0-9a-zA-Z_]+', '_', str(name)).strip('_')


def _mode(s, default=0.0):
    s = pd.Series(s).dropna()
    return s.mode().iloc[0] if len(s) else default


class Harmonizer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return harmonize_df(X)


class FeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self, enabled=True):
        self.enabled = enabled

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return engineer_features(X) if self.enabled else pd.DataFrame(X).copy()


class Preprocessor(BaseEstimator, TransformerMixin):
    """শুধু TRAIN থেকে শেখে: numeric median + IQR winsor; ordinal; binary 0/1; nominal one-hot (<min_freq → __other__)।"""

    def __init__(self, min_freq=0.01):
        self.min_freq = min_freq

    @staticmethod
    def _bin(s, pn):
        pos, neg = pn
        v = pd.to_numeric(s, errors='coerce') if isinstance(pos, (int, float, np.integer)) else s.astype(object)
        return pd.Series(np.where(v == pos, 1.0, np.where(v == neg, 0.0, np.nan)), index=s.index)

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        self.num_ = [c for c in NUMERIC_FEATS if c in X.columns]
        self.medians_, self.fences_ = {}, {}
        for c in self.num_:
            s = pd.to_numeric(X[c], errors='coerce')
            self.medians_[c] = float(s.median()) if s.notna().any() else 0.0
            if c in WINSOR_FEATS and s.notna().sum() > 3:
                q1, q3 = s.quantile(0.25), s.quantile(0.75)
                self.fences_[c] = (float(q1 - 1.5 * (q3 - q1)), float(q3 + 1.5 * (q3 - q1)))
        self.ord_ = {c: m for c, m in ORDINAL_FEATS.items() if c in X.columns}
        self.ord_modes_ = {c: float(_mode(X[c].map(m))) for c, m in self.ord_.items()}
        self.bin_ = {c: pn for c, pn in BINARY_SPEC.items() if c in X.columns}
        self.bin_modes_ = {c: float(_mode(self._bin(X[c], pn))) for c, pn in self.bin_.items()}
        self.flags_ = [c for c in FLAG_FEATS if c in X.columns]
        self.flag_modes_ = {c: float(_mode(pd.to_numeric(X[c], errors='coerce'))) for c in self.flags_}
        self.nom_ = [c for c in NOMINAL_FEATS if c in X.columns]
        self.nom_modes_, self.nom_levels_ = {}, {}
        for c in self.nom_:
            s = X[c].astype(object)
            self.nom_modes_[c] = _mode(s, default='missing')
            freq = s.where(s.notna(), self.nom_modes_[c]).value_counts(normalize=True)
            keep = sorted(freq[freq >= self.min_freq].index.astype(str).tolist())
            if (freq < self.min_freq).any():
                keep.append('__other__')
            self.nom_levels_[c] = keep
        self.columns_, self.source_, self.dummies_ = [], {}, []
        for c in self.num_ + list(self.ord_) + list(self.bin_) + self.flags_:
            self.columns_.append(c)
            self.source_[c] = c
        for c in self.nom_:
            for lv in self.nom_levels_[c]:
                name = f'{c}__{sanitize(lv)}'
                while name in self.source_:
                    name += '_'
                self.columns_.append(name)
                self.source_[name] = c
                self.dummies_.append((name, c, lv))
        return self

    def transform(self, X):
        X = pd.DataFrame(X)
        idx = X.index

        def col(c):
            return X[c] if c in X.columns else pd.Series(np.nan, index=idx, dtype=object)

        out = {}
        for c in self.num_:
            s = pd.to_numeric(col(c), errors='coerce').fillna(self.medians_[c])
            out[c] = s.clip(*self.fences_[c]) if c in self.fences_ else s
        for c, m in self.ord_.items():
            out[c] = pd.to_numeric(col(c).map(m), errors='coerce').fillna(self.ord_modes_[c])
        for c, pn in self.bin_.items():
            out[c] = self._bin(col(c), pn).fillna(self.bin_modes_[c])
        for c in self.flags_:
            out[c] = pd.to_numeric(col(c), errors='coerce').fillna(self.flag_modes_[c])
        prepared = {}
        for c in self.nom_:
            s = col(c).astype(object)
            s = s.where(s.notna(), self.nom_modes_[c]).astype(str)
            if '__other__' in self.nom_levels_[c]:
                s = s.where(s.isin(set(self.nom_levels_[c]) - {'__other__'}), '__other__')
            prepared[c] = s
        for name, c, lv in self.dummies_:
            out[name] = (prepared[c] == lv).astype(float)
        return pd.DataFrame(out, index=idx)[self.columns_].astype(float)


class ColumnPicker(BaseEstimator, TransformerMixin):
    def __init__(self, cols=None):
        self.cols = cols

    def fit(self, X, y=None):
        self.cols_ = list(X.columns) if self.cols is None else [c for c in self.cols if c in X.columns]
        return self

    def transform(self, X):
        return X[self.cols_]


def make_pipeline(model, cols=None, fe=True, smote=False, scale=False, random_state=42):
    """harmonize → FE → preprocess → ফিচার বাছাই → (SMOTE, শুধু training fold-এ) → (scale) → মডেল"""
    from imblearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    steps = [('harm', Harmonizer()), ('fe', FeatureEngineer(enabled=fe)), ('prep', Preprocessor()),
             ('pick', ColumnPicker(cols))]
    if smote:
        from imblearn.over_sampling import SMOTE
        steps.append(('smote', SMOTE(random_state=random_state)))
    if scale:
        steps.append(('scale', StandardScaler()))
    steps.append(('model', model))
    return Pipeline(steps)


def model_input(pipe, X):
    """pipeline-এর মডেলে ঠিক কোন ডেটা ঢোকে (SMOTE বাদে) — SHAP-এর জন্য।"""
    Z = X
    for _, step in pipe.steps[:-1]:
        if hasattr(step, 'fit_resample'):
            continue
        Z = step.transform(Z)
    return pd.DataFrame(np.asarray(Z, dtype=float), columns=pipe.named_steps['pick'].cols_, index=pd.DataFrame(X).index)
