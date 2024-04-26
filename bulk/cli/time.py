import os
from datetime import timedelta, datetime
import tqdm
from pathlib import Path
import numpy as np
import pandas as pd
from bokeh.io import curdoc
from bokeh.layouts import column, row
from bokeh.models import (
    Button, ColorBar, ColumnDataSource, CustomJS,
    DataTable, TableColumn, TextInput, 
    RadioButtonGroup, DateRangeSlider,
    TabPanel, Tabs, MultiChoice,
    Select, Div,
    LassoSelectTool,
    BooleanFilter, IndexFilter, IntersectionFilter, UnionFilter, AllIndices, InversionFilter, GroupFilter, SymmetricDifferenceFilter,
    CDSView,
)
from bokeh.palettes import Category10, Cividis256
from bokeh.transform import factor_cmap, linear_cmap
from bokeh.plotting import figure
from wasabi import msg

from bulk._bokeh_utils import download_js_code, read_file, save_file


OUTPUT_COLUMNS = [
    'deployment_id', 'start_time', 'end_time', 'label', 
    # 'annotation_session', 'annotated_by'
]

LABELS = ['flood', 'blip', 'pulse-chain', 'box', 'snow', 'misc', 'probable-noise', 'probable-flood', '?']

def debounce(wait):
    def wrap(fn):
        values = []
        def update(attr, old, new):
            values[:] = new
            def delayed_job():
                if values == new:
                    print("update", fn.__name__, attr)
                    fn(attr, old, new)
                else:
                    print("debounce", fn.__name__, attr)
            curdoc().add_timeout_callback(delayed_job, wait*1000)
            print('debounce hit', fn.__name__, attr)
        return update
    return wrap

def get_filter_mask(filt, df):
    if isinstance(filt, CDSView):
        return get_filter_mask(filt.filter, df)
    if isinstance(filt, AllIndices):
        return np.ones(len(df), dtype=bool)
    if isinstance(filt, BooleanFilter):
        return np.asarray(filt.booleans) if filt.booleans is not None else np.ones(len(df), dtype=bool)
    if isinstance(filt, IndexFilter):
        return np.isin(np.arange(len(df)), np.asarray(filt.indices)) if filt.indices is not None else np.ones(len(df), dtype=bool)
    if isinstance(filt, InversionFilter):
        return ~get_filter_mask(filt.operand, df)
    if isinstance(filt, IntersectionFilter):
        return np.all([get_filter_mask(f, df) for f in filt.operands], 0)
    if isinstance(filt, UnionFilter):
        return np.any([get_filter_mask(f, df) for f in filt.operands], 0)
    if isinstance(filt, GroupFilter):
        x = df[filt.column_name].to_numpy()
        print(x.shape, x.dtype, type(x), type(df), type(filt.column_name), filt.column_name, filt.group, type(filt.group))
        print(df[filt.column_name].value_counts())
        x = df.loc[:, filt.column_name]
        x = x.to_numpy()
        x = x.astype(str)
        x = x == filt.group
        x = np.asarray(x)
        return x #np.asarray(.to_numpy().copy().astype(str) == filt.group)
    raise NotImplementedError(f'{type(filt).__name__} is not implemented. {filt}')

def multi_group_filter(column, values):
    return UnionFilter(operands=[GroupFilter(column_name=column, group=x) for x in values]) if len(values) else AllIndices()



def bulk_time(path, data_path, output_path='event_output.csv', labels=LABELS, download=True):
    print("Loading data parquet")
    data_df = pd.read_parquet(data_path)
    # data_df['time'] = pd.to_datetime(data_df['time'], format='ISO8601').dt.tz_localize(None)
    # if input('>?'):from IPython import embed;embed()
    # data_df = data_df.dropna(subset=['deployment_id'])
    print("Loaded.", data_df.shape)

    current_session = datetime.now().strftime("%y-%m-%d")
    def bkapp(doc):
        nonlocal labels
        # event_df, colormap, orig_cols = read_file(path)
        # event_df = pd.read_csv(path)

        full_event_df = pd.read_csv(path, index_col=['deployment_id', 'start_time'])

        event_df = full_event_df
        if os.path.isfile(output_path):
            overlay_df = pd.read_csv(output_path, index_col=['deployment_id', 'start_time'])
            event_df = overlay_df.combine_first(full_event_df)
        event_df = event_df.reset_index()


        event_df['start_time'] = pd.to_datetime(event_df['start_time'], format='ISO8601').dt.tz_localize(None)
        event_df['end_time'] = pd.to_datetime(event_df['end_time'], format='ISO8601').dt.tz_localize(None)
        event_df['label'] = event_df['label'].fillna('?')
        print("starting with", event_df.shape)
        event_df = event_df.dropna(subset=['deployment_id'])
        print("dropping non-deployments", event_df.shape)

        if labels is None:
            labels = event_df.label.unique().tolist()

        source = ColumnDataSource(data=event_df)
        deployment_id_filter = UnionFilter(operands=[AllIndices()])
        date_filter = BooleanFilter()
        select_filter = IndexFilter()
        filter_view = CDSView(filter=deployment_id_filter & date_filter)
        selection_view = CDSView(filter=select_filter & filter_view.filter)
        active_view = CDSView(filter=BooleanFilter())
        
        data_table = DataTable(
            source=source, 
            columns=[
                TableColumn(field=c, title=c)
                for c in event_df.columns
                if c not in ['x', 'y', 'color', 'alpha']
            ], 
            view=selection_view, 
            selectable=False, # TODO
            # width=750 if "color" in event_df.columns else 800,
            sizing_mode='stretch_width',
        )

        highlighted_idx = []
        @debounce(0.5)
        def update_selection(attr, old, new):
            """Callback used for plot update when lasso selecting"""
            print("update selection", len(new))
            select_filter.indices = new if len(new) else None
            highlighted_idx[:] = new
            if len(new):
                refresh_tabs(attr, old, new)
            else:
                tab_group.tabs = starter_tabs

        # def save():
        #     """Callback used to save highlighted data points"""
        #     save_file(
        #         dataf=event_df,
        #         highlighted_idx=highlighted_idx,
        #         filename=text_filename.value,
        #         orig_cols=orig_cols,
        #     )

        def save_df(mod_df):
            mod_df = mod_df.copy()[OUTPUT_COLUMNS]
            mod_df['annotated_by'] =  os.getlogin()
            mod_df['annotation_session'] = current_session
            mod_df['annotated_date'] = datetime.now().isoformat()
            mod_df = mod_df.groupby(['deployment_id', 'start_time'], group_keys=True).last().reset_index()
            id_df = mod_df.set_index(['deployment_id', 'start_time'])
            data = source.data
            patch = []
            for i in tqdm.tqdm(range(len(data['deployment_id']))):
                key = (data['deployment_id'][i], data['start_time'][i])
                if key in id_df.index:
                    label = id_df.loc[key, 'label']
                    print("Updating", key)
                    patch.append((i, label))  # TODO: update all fields not just labels
            source.patch({'label': patch})
            for i, l in patch:
                print(i, l, source.data['label'][i])

            print("saving", mod_df.shape)
            if os.path.isfile(output_path):
                from io import StringIO
                mod_df = pd.read_csv(StringIO(mod_df.to_csv(index=False)), index_col=['deployment_id', 'start_time'])
                existing_df = pd.read_csv(output_path, index_col=['deployment_id', 'start_time'])
                merge_df = mod_df.combine_first(existing_df)
                print(merge_df.shape)
                merge_df = merge_df.groupby(level=merge_df.index.names).last()
                print(existing_df.shape, mod_df.shape, merge_df.shape)
                assert set(mod_df.index).issubset(set(merge_df.index))
                
                mdf = merge_df.loc[mod_df.index, mod_df.columns]
                print(mdf.shape, mod_df.shape)
                print(mdf.columns, mod_df.columns)
                differences = (mdf != mod_df).any(axis=1)
                if differences.any():
                    bkp_path = f'event_output_weird_update_{datetime.now().isoformat()}.csv'
                    print('WARNING: DIFFERENCES AFTER MERGE ?? SAVING BACKUP')
                    print(mdf[differences]) # rows that are not equal from mdf
                    print(mod_df[differences]) # rows that are not equal from mod_df
                    # assert False, "mismatch in saving"
                    mod_df.to_csv(bkp_path, index=True)
                mod_df = merge_df.reset_index()
            
            mod_df.to_csv(output_path, index=False)

        def refresh_tabs(attr, old, new):
            print("refreshing tabs")
            df = source.to_df()
            i = get_filter_mask(active_view.filter & selection_view.filter, df)
            df = df.loc[i]
            tab_group.tabs = [
                TabPanel(child=get_time_series_tab(data_df, fdf, labels, save_df), title=f'{label} ({len(fdf)})')
                for label, fdf in df.groupby('label')
            ] + [TabPanel(child=data_table, title='data')]
        db_refresh_tabs = debounce(0.2)(refresh_tabs)

        def update_date(attr, old, new):
            x = date_range_slider.value_as_datetime
            print("update date", x)
            if x is None: 
                date_filter.booleans = None
                return 
            start_time, end_time = x
            date_filter.booleans = (event_df.start_time >= np.datetime64(start_time)) & (event_df.start_time <= np.datetime64(end_time))
            tab_group.tabs = starter_tabs

        time_min = event_df['start_time'].min().date()
        time_max = (event_df['end_time'].max() + timedelta(days=1)).date()
        date_range_slider = DateRangeSlider(value=(time_min, time_max), start=time_min, end=time_max)
        date_range_slider.on_change('value_throttled', update_date)

        def update_labels(attr, old, new):
            print("update labels", new)
            active_view.filter.operands = [GroupFilter(column_name='label', group=x) for x in new] if len(new) else [AllIndices()]
            # active_view.filter.booleans = event_df.label.isin(new) if len(new) else None
            # db_refresh_tabs(attr, old, new)
            tab_group.tabs = starter_tabs

        label_select = MultiChoice(title="Selectable Labels:", value=['?'], options=[*labels])
        label_select.on_change('value', update_labels)
        active_view.filter = multi_group_filter("label", ['?'])

        starter_tabs = [TabPanel(child=data_table, title='please select points on the scatter plot')]
        tab_group = Tabs(tabs=starter_tabs, sizing_mode='stretch_width', max_height=1000, height=1000)
        # doc.add_timeout_callback(lambda: refresh_tabs(None, None, []), 100)
        # db_refresh_tabs(None, None, [])

        options, p = embedding_plot(event_df, source, filter_view, active_view, labels, update_selection)
        stat_table = stats_table(event_df, deployment_id_filter)
        # text_filename, save_btn = file_download(source, save, download)
        print("Fin.")
        return doc.add_root(column(
            row(*options.children, date_range_slider, label_select),
            row(
                column(
                    p,
                    stat_table,
                    sizing_mode="stretch_width",
                ), 
                tab_group,
                sizing_mode='stretch_width',
            ),
            # Div(text="""
            # <style>
            # .scrollable{
            # overflow: auto;
            # max-height: 100vh;
            # }
            # </style>
            # """),
            sizing_mode='stretch_both',
        ))

    return bkapp


def get_time_series_tab(df, filt_df, labels, save, page_size=35):
    c = column(
        sizing_mode='stretch_width',
        max_height=1000,
        height=1000,
        # css_classes=['scrollable'],
        styles={'overflow': 'auto', 'max-height': '80vh'}
    )

    # filt_df = filt_df.sample(frac=1)
    print(df.shape, filt_df.shape)

    page = 0
    n_total = int(np.ceil(len(filt_df) / page_size))
    # page_df = None
    def set_page(new):
        nonlocal page #, page_df
        page = min(max(0, new), n_total)
        # page_df = filt_df.iloc[page*page_size:(page+1)*page_size].copy()
        next_button.label = f"skip ({page+1}/{n_total})"
        refresh_plots()

    def prev_click():
        set_page(page-1)
    def next_click():
        set_page(page+1)
    def save_click():
        commit_values()
        save(filt_df.iloc[page*page_size:(page+1)*page_size])
        set_page(page+1)

    def refresh_plots():
        c.children = [
            time_series_plot(df, event, labels)
            for _, event in filt_df.iloc[page*page_size:(page+1)*page_size].iterrows()
        ] or [
            Div(text="""Done!""", width=200, height=100)
        ]

    # def update_value(idx, label):
    #     page_df.loc[idx, 'label'] = label
    #     print(idx)

    def commit_values():
        for ci in c.children:
            radio = ci.children[-1]
            i = radio._event_index
            filt_df.loc[i, 'label'] = labels[radio.active]

    def change_all():#attr, old, new
        new = change_all_select.value
        if new != '--':
            # page_df['label'] = new
            for ci in c.children:
                ci.children[-1].active = labels.index(new)
            # change_all_select.value = 0

    prev_button = Button(label="back", button_type="default")
    next_button = Button(label=f"skip ({n_total})", button_type="primary")
    save_button = Button(label="submit >", button_type="success")
    change_all_select = Select(title="Assign Label To All:", value='--', options=['--', *labels])
    assign_button = Button(label="change all", button_type="warning")
    # change_all_select.on_change('value', change_all)
    prev_button.on_click(prev_click)
    next_button.on_click(next_click)
    save_button.on_click(save_click)
    assign_button.on_click(change_all)
    set_page(0)
    refresh_plots()

    # Create a column with the plots for each event
    return column(
        row(change_all_select, assign_button, save_button, prev_button, next_button),
        c,
        sizing_mode='stretch_width',
    )

def time_series_plot(df, event, labels):
    # Filter the time series data for the given event
    start_time = pd.to_datetime(event['start_time']).to_pydatetime() - timedelta(minutes=5)
    end_time = pd.to_datetime(event['end_time']).to_pydatetime() + timedelta(minutes=5)
    dfi = df.loc[event['deployment_id']].loc[start_time:end_time]



    # Create the plot with two lines: depth_filt_mm and depth_proc_mm
    p = figure(
        title=f"{event['deployment_id']}: {event['start_time']} ({event['end_time'] - event['start_time']})",
        width_policy='max',
        min_width=600,
        height=200,
        sizing_mode='stretch_width',
    )
    p.line(dfi.index, dfi['depth_filt_mm'], line_width=2, line_color='red', legend_label='filt')
    p.line(dfi.index, dfi['depth_proc_mm'], line_width=2, line_color='blue', legend_label='proc')
    p.y_range.only_visible = True
    p.legend.click_policy="hide"

    # Radio buttons for selecting event label
    radio_button_group = RadioButtonGroup(labels=labels, active=labels.index(event['label']))
    # def update_radio(attr, old, new):
    #     update_value(event.name, new)
    # radio_button_group.on_change('active', update_radio)
    radio_button_group._event_index = event.name
    
    # Return the plot and the radio button group as a column
    return column(
        p,
        radio_button_group,
        sizing_mode='stretch_width',
    )



def stats_table(df, deployment_id_filter):
    # calculate stats
    stat_df = df.groupby(['deployment_id', 'label']).size().unstack().fillna(0)#.sort_index(axis=1)
    print(stat_df)
    sort_cols = [x for x in ['?', 'flood'] if x in stat_df.columns]
    if sort_cols:
        stat_df.sort_values(sort_cols, ascending=False, inplace=True)

    stat_table = DataTable(
        source=ColumnDataSource(data=stat_df), 
        columns=[TableColumn(field=c, title=c)for c in stat_df.reset_index().columns], 
        width=600, 
        selectable='checkbox', 
        autosize_mode='force_fit', 
        sizing_mode="stretch_width"
    )

    # @debounce(0.2)
    def update(attr, old, new):
        print(attr, old, new)
        # deployment_id_filter.booleans = df.deployment_id.isin(stat_df.index[new])
        deployment_id_filter.operands = [multi_group_filter("deployment_id", stat_df.index[new])]
    stat_table.source.selected.on_change("indices", update)
    return stat_table




def embedding_plot(df, source, view, active_view, labels, update_selection):
    cols = df.columns.tolist()
    x_col = Select(title="X:", value='x' if 'x' in cols else cols[0], options=cols)
    y_col = Select(title="Y:", value='y' if 'y' in cols else cols[1], options=cols)

    p = figure(
        title="",
        width=600,
        height=500,
        # sizing_mode="scale_both",
        # sizing_mode="fixed",
        # sizing_mode="scale_width",
        sizing_mode="stretch_width",
        tools=[
            # "lasso_select",
            # "box_select",
            # "pan",
            # "box_zoom",
            "wheel_zoom",
            "reset",
        ],
        # active_drag="lasso_select",
        output_backend="webgl",
        toolbar_location='below',
        tooltips=[('deployment_id', '@deployment_id'), ('start_time', '@start_time'), ('end_time', '@end_time'), ('label', '@label')]
    )

    colors = list(Category10[len(labels)])
    colors = ["grey" if l == '?' else c for l, c in zip(labels, colors)]
    colormap = factor_cmap(
        field_name="label",
        palette=colors,
        factors=labels,
        nan_color="grey",
    )
    if "color" in df.columns:
        color_bar = ColorBar(color_mapper=colormap["transform"], height=8)
        p.add_layout(color_bar, "below")

    # scatter = p.scatter(**{
    #     "x": "x",
    #     "y": "y",
    #     "size": 7,
    #     "source": source,
    #     "view": view,
    #     "alpha": 0.6,
    #     # "line_color": "yellow",
    #     **circle_kwargs,
    # })
    reference = p.scatter(**{
        "x": "x",
        "y": "y",
        "size": 7,
        # 'fill_color': None,
        "source": source,
        "view": CDSView(filter=(~active_view.filter) & view.filter),
        "alpha": 0.3,
        "color": colormap,
    })
    active = p.scatter(**{
        "x": "x",
        "y": "y",
        "size": 7,
        "source": source,
        "view": CDSView(filter=active_view.filter & view.filter),
        # "view": view,
        "alpha": 0.6,
        # "color": colormap,
        "color": colormap
    })

    def chx(a, o, x): 
        reference.glyph.x=x
        active.glyph.x=x
        # scatter.glyph.x=x
    def chy(a, o, x): 
        reference.glyph.y=x
        active.glyph.y=x
        # scatter.glyph.y=x
    # x_col.js_link('value', reference.glyph, 'x')
    # y_col.js_link('value', reference.glyph, 'y')
    # x_col.js_link('value', active.glyph, 'x')
    # y_col.js_link('value', active.glyph, 'y')
    x_col.on_change('value', chx)
    y_col.on_change('value', chy)

    # scatter.data_source.selected.on_change("indices", update_selection)
    active.data_source.selected.on_change("indices", update_selection)
    # active.data_source.selected.on_change("indices", lambda a,o,n: print(a, len(o), len(n)))
    options = row(x_col, y_col)

    lasso = LassoSelectTool(renderers=[active])
    p.add_tools(lasso)
    p.toolbar.active_drag = lasso
    # p.tools[0].renderers = [active]

    return options, p


def file_download(source, save, download):
    text_filename = TextInput(
        value="out.jsonl" if download else "out.csv",
        title="Filename:",
        name="filename",
    )
    save_btn = Button(label="DOWNLOAD" if download else "SAVE")
    if download:
        save_btn.js_on_click(
            CustomJS(args=dict(source=source), code=download_js_code())
        )
    else:
        save_btn.on_click(save)
    return text_filename, save_btn
