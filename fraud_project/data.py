"""Dataset acquisition, integrity audit, strict chronological split and features."""
import hashlib, json, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from .config import SEED, V_COLS, REQUIRED_COLS, FEATURE_NAMES, DATASET_URL, ORIGINAL_DATASET_PAGE

def sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(2 ** 20), b""):
            h.update(chunk)
    return h.hexdigest()

def acquire_data(path: str | Path, try_download: bool = True) -> Path:
    """Uses an explicitly declared public dataset mirror, validates file later.

    If network is blocked, obtain original CSV from Kaggle and put at path.
    """
    path = Path(path)
    if path.exists() and path.stat().st_size > 0:
        print(f"Using dataset: {path} ({path.stat().st_size:,} bytes)")
        return path
    if not try_download:
        raise FileNotFoundError(f"Please place original creditcard.csv at {path} (source: {ORIGINAL_DATASET_PAGE}).")
    import requests
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix('.csv.partial')
    print(f"Downloading public mirror of original ULB dataset to {path} ...")
    try:
        with requests.get(DATASET_URL, stream=True, timeout=(25, 180), allow_redirects=True) as resp:
            resp.raise_for_status()
            with open(temp_path, 'wb') as handle:
                for chunk in resp.iter_content(chunk_size=2 ** 20):
                    if chunk:
                        handle.write(chunk)
        if temp_path.stat().st_size < 1_000_000:
            raise RuntimeError("Downloaded content is unexpectedly small and may not be the CSV.")
        temp_path.replace(path)
        print(f"Saved {path.stat().st_size:,} bytes; SHA256 {sha256(path)}")
        return path
    except Exception as exc:
        temp_path.unlink(missing_ok=True)
        raise RuntimeError(
            f"Automatic download failed: {exc}. Download from {ORIGINAL_DATASET_PAGE} "
            f"and save the CSV as {path}. Original data is not packaged in this project."
        ) from exc

def load_audit_data(path: str | Path, output_dir: str | Path | None = None) -> pd.DataFrame:
    """Verify schema, log provenance, remove byte-for-byte repeated records."""
    path = Path(path)
    df = pd.read_csv(path)
    missing = sorted(set(REQUIRED_COLS) - set(df.columns))
    extra = sorted(set(df.columns) - set(REQUIRED_COLS))
    if missing or extra:
        raise ValueError(f"Dataset schema mismatch. Missing={missing}, additional={extra}")
    df = df[REQUIRED_COLS]
    for c in REQUIRED_COLS:
        df[c] = pd.to_numeric(df[c], errors='raise')
    if not np.isfinite(df.to_numpy(dtype=np.float64)).all():
        raise ValueError("Found missing, NaN or infinite values; audit before continuing.")
    if not df['Class'].isin([0,1]).all():
        raise ValueError("Class must be binary 0/1.")
    if (df['Amount'] < 0).any() or (df['Time'] < 0).any():
        raise ValueError("Unexpected negative amount or time.")
    collision = df.groupby(['Time', *V_COLS, 'Amount'], dropna=False)['Class'].nunique()
    if (collision > 1).any():
        raise ValueError("Identical input features have conflicting labels: investigate.")
    audit = {
        'source': ORIGINAL_DATASET_PAGE,
        'mirror': DATASET_URL,
        'file_sha256': sha256(path),
        'rows_raw': len(df),
        'fraud_raw': int(df.Class.sum()),
        'rows_exact_duplicate': int(df.duplicated().sum()),
        'rows_with_nulls': int(df.isnull().any(axis=1).sum()),
        'time_min': float(df.Time.min()), 'time_max':float(df.Time.max()),
        'schema': REQUIRED_COLS,
    }
    if len(df) != 284807 or int(df.Class.sum()) != 492:
        warnings.warn("Dataset does not match published ULB reference totals; inspect audit.")
    df = df.drop_duplicates(keep='first').sort_values('Time', kind='mergesort').reset_index(drop=True)
    audit.update({'rows_deduplicated':len(df), 'fraud_deduplicated':int(df.Class.sum()),
                  'fraud_rate_deduplicated':float(df.Class.mean())})
    if output_dir is not None:
        p = Path(output_dir); p.mkdir(parents=True, exist_ok=True)
        (p / 'dataset_audit.json').write_text(json.dumps(audit, indent=2))
    print(json.dumps(audit, indent=2))
    return df

def temporal_split(df: pd.DataFrame, train_fraction=0.60, valid_fraction=0.20):
    """Chronological 60/20/20 split; never divide identical timestamps."""
    if not (0 < train_fraction < 1 and 0 < valid_fraction < 1 and train_fraction+valid_fraction < 1):
        raise ValueError('Invalid split fractions.')
    if not df.Time.is_monotonic_increasing:
        raise ValueError('Data must first be sorted by Time.')
    q1, q2 = df.Time.quantile([train_fraction, train_fraction + valid_fraction]).tolist()
    train = df.loc[df.Time <= q1].copy().reset_index(drop=True)
    valid = df.loc[(df.Time > q1) & (df.Time <= q2)].copy().reset_index(drop=True)
    test = df.loc[df.Time > q2].copy().reset_index(drop=True)
    for name, part in [('train',train),('val',valid)]:
        if part.empty or part.Class.nunique() < 2:
            raise ValueError(f'{name} partition empty or contains only one class.')
    if test.empty:
        raise ValueError('Empty test partition.')
    assert train.Time.max() < valid.Time.min() < test.Time.min()
    assert len(train) + len(valid) + len(test) == len(df)
    split_info = {'method': 'strict temporal quantiles with non-overlapping timestamps',
                  'train_fraction_target':train_fraction, 'validation_fraction_target':valid_fraction,
                  'split_thresholds': [float(q1),float(q2)]}
    for name, part in [('train',train),('valid',valid)]:
        split_info[name] = {'n':len(part),'fraud':int(part.Class.sum()),
                            'prevalence':float(part.Class.mean()),
                            'time_min':float(part.Time.min()),'time_max':float(part.Time.max())}
    split_info['test'] = {'n':len(test), 'labels':'SEALED',
                          'time_min':float(test.Time.min()), 'time_max':float(test.Time.max())}
    print(pd.DataFrame({x:split_info[x] for x in ['train','valid','test']}).T.to_string())
    return train, valid, test, split_info

def make_features(df: pd.DataFrame) -> pd.DataFrame:
    # Same fixed, label-free, point-in-time features for every model.
    # Time is seconds since the first observed transaction; absolute Time is
    # omitted because it extrapolates poorly to future-only evaluation.
    out = df[V_COLS].astype('float32').copy()
    out['log_amount'] = np.log1p(df['Amount'].to_numpy(dtype='float64')).astype('float32')
    phase = (df['Time'].to_numpy(dtype='float64') % 86400) * (2 * np.pi / 86400)
    out['time_sin'] = np.sin(phase).astype('float32')
    out['time_cos'] = np.cos(phase).astype('float32')
    assert out.columns.tolist() == FEATURE_NAMES and np.isfinite(out.to_numpy()).all()
    return out
