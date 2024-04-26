import os
import tqdm
from pathlib import Path
from datetime import timedelta

import torch
import numpy as np
import pandas as pd

from chronos import ChronosPipeline

import ipdb
@ipdb.iex
def main(
    data_dir = '/Users/bea/Downloads/depth2',
    # event_csv = '/Users/bea/Downloads/AllEventsData523-6.csv',
    event_csv = 'events_merged.csv',
    output_dir = 'ready',
    x_column = 'depth_filt_mm',
    proj='umap',
    batch_size = 128,
    buffer_mins = 3,
):
    os.makedirs(output_dir, exist_ok=True)

    # --------------------------------- Load events -------------------------------- #
    
    # LABELS = ['flood', 'blip', 'pulse-chain', 'box', 'snow', 'misc']

    event_df = pd.read_csv(event_csv)
    # event_df = event_df.rename(columns={c: c.lower() for c in event_df.columns})
    # event_df['label'] = event_df['class'].apply(lambda i: LABELS[i])
    # event_df = pd.concat([event_df, pd.read_csv('all_events_filt.csv')])
    # if input('>?'):from IPython import embed;embed()

    event_df['start_time'] = pd.to_datetime(event_df['start_time'], format='ISO8601').dt.tz_localize(None)
    event_df['end_time'] = pd.to_datetime(event_df['end_time'], format='ISO8601').dt.tz_localize(None)


    # --------------------------- Load time series data -------------------------- #
    data_df = pd.concat((
        pd.read_csv(f, on_bad_lines='warn') 
        for f in tqdm.tqdm(list(Path(data_dir).glob('*.csv')), desc="Loading...")
    ), ignore_index=True)
    data_df['time'] = pd.to_datetime(data_df['time'], format='ISO8601').dt.tz_localize(None)
    data_df = data_df.set_index(['deployment_id', 'time']).sort_index()

    print(set(event_df.deployment_id.unique()) - set(data_df.index.get_level_values(0).unique()))
    event_df = event_df[event_df.deployment_id.isin(data_df.index.get_level_values(0).unique())].reset_index()
    y_label = event_df.label.values

    emb_cache_npz = f'{x_column}.npz'
    if os.path.isfile(emb_cache_npz):
        d = np.load(emb_cache_npz, allow_pickle=True)
        Z = d['Z']
        stats = d['stats']
    else:

        # ----------------------------------- Embed ---------------------------------- #

        Z, stats = extract_embeddings(event_df, data_df, x_column, batch_size, buffer_mins)
        np.savez(emb_cache_npz, Z=Z, stats=stats)

    print(Z.shape)

    # -------------------------------- Projection -------------------------------- #
    print("Projecting...")

    if proj == 'umap':
        from umap import UMAP
        umap = UMAP()
        Zt = umap.fit_transform(Z)
    elif proj == 'umap':
        from sklearn.decomposition import PCA
        umap = PCA()
        Zt = umap.fit_transform(Z)
    elif proj == 'lda':
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
        umap = LinearDiscriminantAnalysis()
        known_label = ~pd.isna(y_label)
        zp, yp = Z[known_label], y_label[known_label]
        print("known:", zp.shape)
        ix = np.random.permutation(np.arange(len(zp)))[:int(len(zp)*2/3)]
        print("fit:", zp[ix].shape)
        umap.fit(zp[ix], yp[ix])
        Zt = umap.transform(Z)
    else:
        raise ValueError(f'Invalid projection: {proj}')
    
    print(Zt.shape)

    event_df = event_df[['deployment_id', 'start_time', 'end_time', 'label']]

    # Apply coordinates
    event_df['x'] = Zt[:, 0]
    event_df['y'] = Zt[:, 1]
    event_df['color'] = y_label
    event_df = pd.concat([event_df, pd.DataFrame(stats)], axis=1)

    output_csv = f'{output_dir}/ready.csv'
    event_df.to_csv(output_csv, index=False)
    event_df.to_parquet(f'{output_dir}/events.parquet')
    print('wrote to', output_csv)
    output_csv = f'{output_dir}/data.parquet'
    data_df.to_parquet(output_csv)
    print('wrote to', output_csv)


def extract_embeddings(event_df, data_df, x_column, batch_size, buffer_mins):
    if os.path.isfile(f'{x_column}.npz'):
        return np.load(f'{x_column}.npz')['Z']
    
    pipeline = ChronosPipeline.from_pretrained("amazon/chronos-t5-small", device_map="mps")

    Zs = []
    stats = []
    for i in tqdm.tqdm(range(0, len(event_df) + batch_size-1, batch_size), desc="Embedding..."):
        edf = event_df.iloc[i:i+batch_size]

        x_mm = [
            data_df.loc[e.deployment_id, x_column].loc[
                e.start_time - timedelta(minutes=buffer_mins):
                e.end_time + timedelta(minutes=buffer_mins)]
            for _, e in edf.iterrows()
        ]
        if not len(x_mm):
            continue

        Z, tok_state = pipeline.embed([torch.tensor(x.values) for x in x_mm])
        Zs.append(Z.mean(1).cpu().numpy())
        stats.extend([x_stats(x) for x in x_mm])

    Z = np.concatenate(Zs)
    return Z, stats


def x_stats(x):
    dxdt = x.diff()# / (x.index.diff().dt.total_seconds()/60)
    # dx2dt = dxdt.diff() / (x.index.diff().dt.total_seconds()/60)
    return {
        'num_points': len(x),
        'duration': x.index.max() - x.index.min(),
        'max_value': x.max(),
        'min_value': x.max(),
        'dx_spread': dxdt.max() - dxdt.min(),
        'dx_mean': dxdt.mean(),
        # 'ddx_spread': dx2dt.max() - dx2dt.min(),
    }


if __name__ == '__main__':
    import fire
    fire.Fire(main)