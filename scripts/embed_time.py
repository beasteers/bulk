from concurrent.futures import ThreadPoolExecutor
import os
from IPython import embed
import requests
import tqdm
from pathlib import Path
from datetime import timedelta

import torch
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from chronos import ChronosPipeline

TIME_QUERY_CONFIG = {
    # "url": "https://api.dev.floodlabs.nyc/v1/graphql",
    "url": "http://localhost:8080/v1/graphql",
    "query": """
query TimeSeriesData(
        $deployment_id: String!,
        $start_time: timestamptz!,
        $end_time: timestamptz!
) {
    depth_data(where: {
        deployment_id: { _eq: $deployment_id },
        time: { _gt: $start_time, _lt: $end_time },
    }, order_by: {time: asc}) {
        time
        depth_filt_mm
        depth_proc_mm
    }
}
""",
    'response_key': 'depth_data',
}

import ipdb
@ipdb.iex
def main(
    # data_dir = '/Users/bea/Downloads/depth2',
    # event_csv = '/Users/bea/Downloads/AllEventsData523-6.csv',
    event_csv = 'events_merged.csv',
    # event_csv = 'all_events_filt.csv',
    output_dir = 'ready',
    x_column = 'depth_filt_mm',
    proj='umap',
    batch_size = 128,
    buffer_mins = 20,
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
    # data_df = pd.concat((
    #     pd.read_csv(f, on_bad_lines='warn') 
    #     for f in tqdm.tqdm(list(Path(data_dir).glob('*.csv')), desc="Loading...")
    # ), ignore_index=True)
    # data_df['time'] = pd.to_datetime(data_df['time'], format='ISO8601').dt.tz_localize(None)
    # data_df = data_df.set_index(['deployment_id', 'time']).sort_index()

    # print(set(event_df.deployment_id.unique()) - set(data_df.index.get_level_values(0).unique()))
    # event_df = event_df[event_df.deployment_id.isin(data_df.index.get_level_values(0).unique())].reset_index()
    y_label = event_df.label.values

    emb_cache_npz = f'{x_column}.npz'
    if os.path.isfile(emb_cache_npz):
        d = np.load(emb_cache_npz, allow_pickle=True)
        Z = d['Z']
        stats = d['stats']
    else:

        # ----------------------------------- Embed ---------------------------------- #

        Z, stats = extract_embeddings(event_df, x_column, batch_size, buffer_mins)
        np.savez(emb_cache_npz, Z=Z, stats=stats)

    print(Z.shape)

    # -------------------------------- Projection -------------------------------- #

    event_df = event_df[['deployment_id', 'start_time', 'end_time', 'label']]

    print("Projecting...")
    Zt = project(Z, proj, y_label)
    event_df['x'] = Zt[:, 0]
    event_df['y'] = Zt[:, 1]
    event_df['z'] = Zt[:, 1]
    print(proj, Zt.shape)

    Zt = project(Z, 'umap', ndim=3)
    event_df['umap3_x'] = Zt[:, 0]
    event_df['umap3_y'] = Zt[:, 1]
    event_df['umap3_z'] = Zt[:, 2]
    print('umap', Zt.shape)

    event_df = pd.concat([event_df, pd.DataFrame(list(stats))], axis=1)
    event_df2=event_df.copy()
    
    filtered = event_df.label.isna()
    proc_filtered = filtered & (event_df.max_proc_value.fillna(0) == 0)
    false_alarm = event_df.max_value == 0
    print(proc_filtered.mean())
    event_df.loc[proc_filtered, 'label'] = 'misc'
    event_df.loc[proc_filtered & (event_df.num_points < 5), 'label'] = 'blip'
    event_df.loc[false_alarm, 'label'] = 'false-alarm'

    print(event_df.label.fillna('?').value_counts())
    # embed()

    output_csv = f'{output_dir}/full_events.csv'
    event_df.to_csv(output_csv, index=False)
    print('wrote to', output_csv, event_df.shape)

    event_df = event_df[event_df.label != 'false-alarm']

    output_csv = f'{output_dir}/events.csv'
    event_df.to_csv(output_csv, index=False)
    pd.DataFrame(Z).to_parquet("events_embeddings.parquet")
    # event_df.to_parquet(f'{output_dir}/events.parquet')
    print('wrote to', output_csv, event_df.shape)
    # output_csv = f'{output_dir}/data.parquet'
    # data_df.to_parquet(output_csv)
    # print('wrote to', output_csv)


def project(Z, proj, y_label=None, ndim=None, train_ratio=2/3):
    if proj == 'umap':
        from umap import UMAP
        umap = UMAP(n_components=ndim or 2)
        Zt = umap.fit_transform(Z)
    elif proj == 'pca':
        from sklearn.decomposition import PCA
        umap = PCA(n_components=ndim or 2)
        Zt = umap.fit_transform(Z)
    elif proj == 'lda':
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
        umap = LinearDiscriminantAnalysis(n_components=ndim)
        known_label = ~pd.isna(y_label) & (y_label != 'misc')
        zp, yp = Z[known_label], y_label[known_label]
        print("known:", zp.shape)
        ix = np.random.permutation(np.arange(len(zp)))[:int(len(zp)*train_ratio)]
        print("fit:", zp[ix].shape)
        umap.fit(zp[ix], yp[ix])
        Zt = umap.transform(Z)
    else:
        raise ValueError(f'Invalid projection: {proj}')
    return Zt


def query_data(event, x_column, buffer_mins):
    # print("query", event['deployment_id'], event['start_time'])
    URL = TIME_QUERY_CONFIG['url']
    QUERY = TIME_QUERY_CONFIG['query']
    VARIABLES = {
        'deployment_id': event['deployment_id'], 
        'start_time': (event['start_time'] - timedelta(minutes=buffer_mins)).isoformat(), 
        'end_time': (event['end_time'] + timedelta(minutes=buffer_mins)).isoformat(),
    }
    response = requests.post(URL, json={
        'query': QUERY, 
        'variables': VARIABLES,
    }).json()
    if 'errors' in response:
        raise RuntimeError(response['errors'])
    df = pd.DataFrame(response['data'][TIME_QUERY_CONFIG['response_key']])
    df['time'] = pd.to_datetime(df['time'], format='ISO8601')
    df = df.set_index('time').sort_index()
    df = df.astype('Int64')
    return df #[x_column]


def extract_embeddings(event_df, x_column, batch_size, buffer_mins):
    if os.path.isfile(f'{x_column}.npz'):
        return np.load(f'{x_column}.npz')['Z']
    
    pipeline = ChronosPipeline.from_pretrained("amazon/chronos-t5-small", device_map="mps")

    Zs = []
    stats = []
    # with ThreadPoolExecutor(max_workers=1) as executor:
    for i in tqdm.tqdm(range(0, len(event_df) + batch_size-1, batch_size), desc="Embedding..."):
        edf = event_df.iloc[i:i+batch_size]
        events = [e for _, e in edf.iterrows()]
        # futs = [executor.submit(query_data, e, x_column, buffer_mins=buffer_mins) for e in events]
        # x_mm = [f.result() for f in futs]
        dfs = [query_data(e, x_column, buffer_mins=buffer_mins) for e in events]
        # for x, e in zip(x_mm, events):
        #     # print(e)
        #     plt.plot(x.depth_filt_mm)
        #     plt.plot(x.depth_proc_mm)
        #     plt.scatter(x.index, x.depth_filt_mm.values)
        #     plt.title(f"{e['deployment_id']} {e['start_time']}")
        #     plt.axvline(e.start_time)
        #     plt.axvline(e.end_time)
        #     plt.savefig('event.png')
        #     plt.close()
        #     input()
        x_mm = [x[x_column].dropna() for x in dfs]
        # x_mm = [x for x in x_mm]

        # x_mm = [
        #     data_df.loc[e.deployment_id, x_column].loc[
        #         e.start_time - timedelta(minutes=buffer_mins):
        #         e.end_time + timedelta(minutes=buffer_mins)]
        #     for _, e in edf.iterrows()
        # ]
        if not len(x_mm):
            continue

        Z, tok_state = pipeline.embed([torch.tensor(x.values) for x in x_mm])
        Zs.append(Z.mean(1).cpu().numpy())
        stats.extend([x_stats(d) for d in dfs])

    Z = np.concatenate(Zs)
    return Z, stats


def x_stats(df):
    x = df.depth_filt_mm
    x2 = df.depth_proc_mm.fillna(0)
    # dxdt = x.diff()# / (x.index.diff().dt.total_seconds()/60)
    # dx2dt = dxdt.diff() / (x.index.diff().dt.total_seconds()/60)
    return {
        'num_points': len(x[x>0]),
        'duration': x.index.max() - x.index.min(),
        'max_value': x.max(),
        'max_proc_value': 0 if not len(x2) else x2.max(),
        # 'min_value': x.max(),
        # 'dx_spread': dxdt.max() - dxdt.min(),
        # 'dx_mean': dxdt.mean(),
        # 'ddx_spread': dx2dt.max() - dx2dt.min(),
    }


if __name__ == '__main__':
    import fire
    fire.Fire(main)