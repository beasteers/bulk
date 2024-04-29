import os
from datetime import timedelta, datetime
from turtle import width
import tqdm
from pathlib import Path
import numpy as np
import pandas as pd
from bokeh.io import curdoc
from bokeh.layouts import column, row
from bokeh.models import (
    Button, ColorBar, ColumnDataSource, CustomJS,
    DataTable, TableColumn, TextInput, DataCube,
    SumAggregator, GroupingInfo, DateFormatter, StringFormatter,
    RadioButtonGroup, DateRangeSlider,
    TabPanel, Tabs, MultiChoice,
    Select, Div,
    LassoSelectTool, RangeTool, PolyAnnotation,
    BooleanFilter, IndexFilter, IntersectionFilter, UnionFilter, AllIndices, InversionFilter, GroupFilter, SymmetricDifferenceFilter,
    CDSView,
    GlobalInlineStyleSheet, InlineStyleSheet,
)
from bokeh.palettes import Category10, Cividis256
from bokeh.transform import factor_cmap, linear_cmap
from bokeh.plotting import figure
from bokeh.themes import Theme
from wasabi import msg

from bulk._bokeh_utils import download_js_code, read_file, save_file
from bulk.cli.custom.graphql_datasource import GraphQLDataSource


OUTPUT_COLUMNS = [
    'deployment_id', 'start_time', 'end_time', 'label', 
    # 'annotation_session', 'annotated_by'
]

LABELS = ['flood', 'blip', 'pulse-chain', 'box', 'snow', 'misc', 'probable-noise', 'probable-flood', '?']

TIME_QUERY_CONFIG = {
    "url": "https://api.dev.floodlabs.nyc/v1/graphql",
    # "url": "http://localhost:8080/v1/graphql",
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
        deployment_id
    }
}
""",
}

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
    data_df = None #pd.read_parquet(data_path)
    # data_df['time'] = pd.to_datetime(data_df['time'], format='ISO8601').dt.tz_localize(None)
    # if input('>?'):from IPython import embed;embed()
    # data_df = data_df.dropna(subset=['deployment_id'])
    # print("Loaded.", data_df.shape)

    current_session = datetime.now().strftime("%y-%m-%d")
    def bkapp(doc):
        doc.title = "FloodNet Annotator"
        css = open(os.path.join(os.path.dirname(__file__), 'theme.css')).read()
        stylesheet = GlobalInlineStyleSheet(css=css)
        doc.theme = Theme(
            json={'attrs': {
                'UIElement': {'stylesheets': [InlineStyleSheet(css=css)]},
                # "Plot": {
                #     "background_fill_color": "#a99fff32",
                #     "border_fill_color": "#15191C",
                #     "outline_line_color": "#00000000",
                #     "outline_line_alpha": 0,
                # },
                # "Legend": {
                #     "label_text_color": "#E0E0E0",
                #     "background_fill_color": "#a99fff32",
                # },
                # "Grid": {
                #     "grid_line_color": "#E0E0E0",
                #     "grid_line_alpha": 0.25,
                # },
                # "Axis": {
                #     "major_tick_line_color": "#E0E0E0",
                #     "minor_tick_line_color": "#E0E0E0",
                #     "axis_line_color": "#E0E0E0",
                #     "major_label_text_color": "#E0E0E0",
                #     "axis_label_text_color": "#E0E0E0",
                # },
                # "BaseColorBar": {
                #     "title_text_color": "#E0E0E0",
                #     "major_label_text_color": "#E0E0E0",
                #     "background_fill_color": "#a99fff32",
                # },
                # "Title": {
                #     "text_color": "#E0E0E0",
                # },
            }},
        )
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
        event_df['count'] = 1
        print("starting with", event_df.shape)
        event_df = event_df.dropna(subset=['deployment_id'])
        print("dropping non-deployments", event_df.shape)

        if labels is None:
            labels = event_df.label.unique().tolist()

        # -------------------------------- Data Source ------------------------------- #

        source = ColumnDataSource(data=event_df)
        # deployment_id_filter = UnionFilter(operands=[AllIndices()])
        deployment_id_filter = BooleanFilter()
        date_filter = BooleanFilter()
        select_filter = IndexFilter()
        label_filter = UnionFilter(operands=[AllIndices()])
        filter_view = CDSView(filter=deployment_id_filter & date_filter & label_filter)
        selection_view = CDSView(filter=select_filter & filter_view.filter)
        active_view = CDSView(filter=BooleanFilter())

        # ---------------------------- Summary Data Table ---------------------------- #
        
        # data_table = DataTable(
        #     source=source, 
        #     columns=[
        #         TableColumn(field=c, title=c)
        #         for c in event_df.columns
        #         if c not in ['x', 'y', 'color', 'alpha']
        #     ], 
        #     view=selection_view, 
        #     selectable=False, # TODO
        #     # width=750 if "color" in event_df.columns else 800,
        #     sizing_mode='stretch_both',
        # )
        columns = [
            TableColumn(field="deployment_id", title="Name"),
            TableColumn(field="label", title="Label"),
            TableColumn(field="start_time", title="Start", formatter=DateFormatter()),
            TableColumn(field="end_time", title="End", formatter=DateFormatter()),
            # TableColumn(field="annotated_by", title="Annotator"),
            # TableColumn(field=c, title=c)
            # for c in event_df.columns
            # if c not in ['x', 'y', 'color', 'alpha']
        ]
        data_table = DataTable(
            source=source, 
            columns=columns + [
                TableColumn(field=c, title=c.replace('_',' ')) for c in event_df.columns if c not in ([c.field for c in columns] + ['x', 'y', 'z', 'color', 'alpha']) and not any(c.endswith(p) for p in ['_x', '_y', '_z'])
            ], 
            view=selection_view, 
            selectable=False, # TODO
            sizing_mode='stretch_both',
            index_position=None,
            
        )
        # data_table = DataCube(
        #     source=source, 
        #     view=selection_view, 
        #     selectable=False, # TODO
        #     sizing_mode='stretch_both',
        #     # columns=[
        #     #     TableColumn(field='deployment_id', title='Name', width=80),
        #     #     # TableColumn(field='label', title='Label', width=40),
        #     # ],
        #     columns=[
        #         TableColumn(field=c, title=c)
        #         for c in event_df.columns
        #         if c not in ['x', 'y', 'color', 'alpha']
        #     ], 
        #     grouping=[
        #         GroupingInfo(getter='deployment_id', aggregators=[SumAggregator(field_='count')]),
        #         GroupingInfo(getter='label', aggregators=[SumAggregator(field_='count')]),
        #     ],
        #     target=ColumnDataSource(data=dict(row_indices=[], labels=[])),
        # )

        # -------------------------------- Date filter ------------------------------- #

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
        date_range_slider = DateRangeSlider(value=(time_min, time_max), start=time_min, end=time_max, styles={'flex-grow': '1'})
        date_range_slider.on_change('value_throttled', update_date)

        # ------------------------------ Label Selection ----------------------------- #

        def update_select_labels(attr, old, new):
            print("update labels", new)
            active_view.filter.operands = [GroupFilter(column_name='label', group=x) for x in new] if len(new) else [AllIndices()]
            tab_group.tabs = starter_tabs

        label_select = MultiChoice(title="Selectable Labels:", value=['?'], options=[*labels])
        label_select.on_change('value', update_select_labels)
        active_view.filter = multi_group_filter("label", ['?'])

        def update_show_labels(attr, old, new):
            print("update labels", new)
            label_filter.operands = [GroupFilter(column_name='label', group=x) for x in new] if len(new) else [AllIndices()]
            tab_group.tabs = starter_tabs

        label_show = MultiChoice(title="Visible Labels:", value=[], options=[*labels])
        label_show.on_change('value', update_show_labels)

        # ----------------------------------- Tabs ----------------------------------- #

        def refresh_tabs(attr, old, new):
            print("refreshing tabs")
            df = source.to_df()
            i = get_filter_mask(active_view.filter & selection_view.filter, df)
            df = df.loc[i]
            tab_group.tabs = [
                TabPanel(child=get_time_series_tab(data_df, fdf, labels, save_df), title=f'{label} ({len(fdf)})')
                for label, fdf in df.groupby('label')
            ] + data_tabs

        data_tabs = [TabPanel(child=column(data_table, sizing_mode='stretch_height', width=600), title='data')]
        starter_tabs = [TabPanel(child=data_table, title='please select points on the scatter plot')]
        tab_group = Tabs(tabs=starter_tabs, sizing_mode='stretch_width', max_height=1000, height=1000)

        # ---------------------------------- Actions --------------------------------- #

        @debounce(0.5)
        def update_selection(attr, old, new):
            """Callback used for plot update when lasso selecting"""
            print("update selection", len(new))
            select_filter.indices = new if len(new) else None
            if len(new):
                refresh_tabs(attr, old, new)
            else:
                tab_group.tabs = starter_tabs

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

        # ---------------------------- Embedding Explorer ---------------------------- #

        options, p = embedding_plot(event_df, source, filter_view, active_view, labels, update_selection)

        # -------------------------------- Stats Table ------------------------------- #

        stat_table = stats_table(event_df, source, deployment_id_filter)
        # text_filename, save_btn = file_download(source, save, download)

        # ----------------------------------- Page ----------------------------------- #

        print("Fin.")
        return doc.add_root(column(
            row(
                *options.children, date_range_slider, label_select, label_show,
                sizing_mode='stretch_width',
                styles={
                    'align-items': 'flex-end',
                    'padding': '0.1em 0.4em',
                    # 'width': 'auto',
                },
            ),
            row(
                column(
                    p,
                    stat_table,
                    sizing_mode="stretch_width",
                    # resizable='width',
                ), 
                tab_group,
                sizing_mode='stretch_both',
                # styles={
                #     'flex': 'stretch',
                # }
            ),
            sizing_mode='stretch_both',
            styles={
                'align-items': 'stretch',
            }, 
            stylesheets=[stylesheet]
        ))

    return bkapp


def get_time_series_tab(df, filt_df, labels, save, page_size=35):
    c = column(
        sizing_mode='stretch_width',
        styles={
            'overflow': 'auto', 
            'flex': '1 1',
        }
    )

    # filt_df = filt_df.sample(frac=1)
    print(filt_df.shape)

    page = 0
    n_total = int(np.ceil(len(filt_df) / page_size))
    def set_page(new):
        nonlocal page #, page_df
        page = min(max(0, new), n_total)
        next_button.label = f"skip ({page+1}/{n_total})"
        refresh_plots()

    def refresh_plots():
        c.children = [
            ajax_time_series_plot(df, event, labels)
            for _, event in filt_df.iloc[page*page_size:(page+1)*page_size].iterrows()
        ] or [Div(text="""Done!""", width=200, height=100)]

    def change_all(new):
        if new != '--':
            for ci in c.children:
                ci.children[-1].active = labels.index(new)

    def commit_values():
        for ci in c.children:
            radio = ci.children[-1]
            i = radio._event_index
            filt_df.loc[i, 'label'] = labels[radio.active]

    def save_click():
        commit_values()
        save(filt_df.iloc[page*page_size:(page+1)*page_size])
        set_page(page+1)

    prev_button = Button(label="back", button_type="default")
    next_button = Button(label=f"skip ({n_total})", button_type="primary")
    save_button = Button(label="submit >", button_type="success")
    change_all_select = Select(title="Assign Label To All:", value='--', options=['--', *labels])
    assign_button = Button(label="change all", button_type="warning")
    change_all_select.on_change('value', lambda a,o,n: change_all(n))
    prev_button.on_click(lambda: set_page(page-1))
    next_button.on_click(lambda: set_page(page+1))
    save_button.on_click(save_click)
    assign_button.on_click(lambda: change_all(change_all_select.value))
    set_page(0)
    refresh_plots()

    # Create a column with the plots for each event
    return column(
        row(
            change_all_select, assign_button, save_button, prev_button, next_button,
            styles={'align-items': 'flex-end'},
        ),
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
    radio_button_group._event_index = event.name
    
    # Return the plot and the radio button group as a column
    return column(
        p,
        radio_button_group,
        sizing_mode='stretch_width',
    )


def ajax_time_series_plot(df, event, labels):
    # Filter the time series data for the given event
    start_time = pd.to_datetime(event['start_time']).to_pydatetime() - timedelta(minutes=10)
    end_time = pd.to_datetime(event['end_time']).to_pydatetime() + timedelta(minutes=10)

    ts_source = GraphQLDataSource(
        data_url=TIME_QUERY_CONFIG['url'],
        query=TIME_QUERY_CONFIG['query'],
        variables={
            "deployment_id": event['deployment_id'],
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
        },
        polling_interval=None, 
        http_headers={'Access-Control-Allow-Origin': '*'},
        adapter=CustomJS(code="""
        let data = cb_data.response.data.depth_data

        // row oriented to column oriented
        data = [...new Set(data.flatMap(Object.keys))].reduce((result, key) => {
            result[key] = data.map(row => row[key]);
            return result;
        }, {});

        data.time = data.time.map(x => Date.parse(x));
        console.log(data);
        return data;
        """)
    )

    # Create the plot with two lines: depth_filt_mm and depth_proc_mm
    p = figure(
        title=f"{event['deployment_id']}: {event['start_time']} ({event['end_time'] - event['start_time']})",
        width_policy='max',
        min_width=600,
        height=200,
        x_axis_type="datetime",
        sizing_mode='stretch_width',
        toolbar_location='above',
        # tooltips=[('depth', '@y'), ('time', '@x')],
        tooltips=[('depth', '@depth'), ('time', '@time')],
    )
    p.line('time', 'depth_filt_mm', source=ts_source, line_width=2, line_color='red', legend_label='filt')
    p.line('time', 'depth_proc_mm', source=ts_source, line_width=2, line_color='blue', legend_label='proc')
    p.ray(x=[event['start_time']], y=[0], length=0, angle=90, line_width=2, line_dash='dotted', line_color='#9370DB', angle_units='deg')
    p.ray(x=[event['end_time']], y=[0], length=0, angle=90, line_width=2, line_dash='dotted', line_color='#9370DB', angle_units='deg')
    p.y_range.only_visible = True
    p.legend.click_policy="hide"
    p.toolbar.logo = None

    # Radio buttons for selecting event label
    radio_button_group = RadioButtonGroup(labels=labels, active=labels.index(event['label']))
    radio_button_group._event_index = event.name
    
    # Return the plot and the radio button group as a column
    return column(
        p,
        radio_button_group,
        sizing_mode='stretch_width',
    )


def stats_table(df, source, deployment_id_filter):
    # calculate stats
    stat_df = df.groupby(['deployment_id', 'label']).size().unstack().fillna(0)
    stat_table = DataTable(
        source=ColumnDataSource(data=stat_df), 
        columns=[TableColumn(field=c, title=c)for c in stat_df.reset_index().columns], 
        width=600, 
        selectable='checkbox', 
        autosize_mode='force_fit', 
        sizing_mode='stretch_width', 
        styles={'flex-shrink': '1'},
        index_position=None,
    )

    # # @debounce(0.2)
    # def update(attr, old, new):
    #     print(attr, old, new)
    #     deployment_id_filter.operands = [multi_group_filter("deployment_id", stat_df.index[new])]
    # stat_table.source.selected.on_change("indices", update)
    stat_table.source.selected.js_on_change('indices',  CustomJS(
        args=dict(source=source, stat_source=stat_table.source, f=deployment_id_filter),
        code="""
        const selected = cb_obj.indices.map(idx => stat_source.data.deployment_id[idx]);
        f.booleans = selected.length ? source.data.deployment_id.map(l => selected.includes(l)) : null;
        """))
    return stat_table


def embedding_plot(df, source, view, active_view, labels, update_selection):
    p = figure(
        title="",
        width=600,
        height=500,
        sizing_mode="stretch_width",
        tools=[
            # "lasso_select",
            # "box_select",
            "pan",
            "box_zoom",
            "wheel_zoom",
            "reset",
        ],
        x_axis_type=None,
        y_axis_type=None,
        output_backend="webgl",
        toolbar_location='above',
        tooltips=[('deployment_id', '@deployment_id'), ('start_time', '@start_time'), ('end_time', '@end_time'), ('label', '@label')]
    )
    p.toolbar.autohide = True

    colors = list(Category10[len(labels)])
    # "#010026"
    colors = ["grey" if l == '?' else c for l, c in zip(labels, colors)]
    colormap = factor_cmap(
        field_name="label",
        palette=colors,
        factors=labels,
        nan_color="grey",
    )
    color_bar = ColorBar(color_mapper=colormap["transform"], height=8)
    p.add_layout(color_bar, "below")

    reference = p.scatter(
        x="x", y="y",
        size=7,
        source=source,
        view=CDSView(filter=(~active_view.filter) & view.filter),
        alpha=0.4,
        color=colormap,
        line_color=None,
    )
    active = p.scatter(
        x="x", y="y",
        size=11,
        source=source,
        view=CDSView(filter=active_view.filter & view.filter),
        alpha=0.4,
        color=colormap,
        # line_color=None,
        line_color='black',
        selection_line_color="#ff1141",
        selection_alpha=0.7,
        selection_line_width=3,
        hover_fill_color="midnightblue", hover_alpha=0.5,
        hover_line_color="white",
    )

    cols = df.columns.tolist()
    x_col = Select(title="X:", value='x' if 'x' in cols else cols[0], options=cols, styles={'max-width': '6em'})
    y_col = Select(title="Y:", value='y' if 'y' in cols else cols[1], options=cols, styles={'max-width': '6em'})
    x_col.js_on_change('value', CustomJS(args=dict(g=reference.glyph),code="g.x = {field: this.value}"))
    x_col.js_on_change('value', CustomJS(args=dict(g=active.glyph),code="g.x = {field: this.value}"))
    y_col.js_on_change('value', CustomJS(args=dict(g=reference.glyph),code="g.y = {field: this.value}"))
    y_col.js_on_change('value', CustomJS(args=dict(g=active.glyph),code="g.y = {field: this.value}"))

    # polygon = PolyAnnotation(fill_color="blue", fill_alpha=0.2)
    # p.add_layout(polygon)

    active.data_source.selected.on_change("indices", update_selection)

    select = figure(
                height=40, 
                x_axis_type=None, y_axis_type=None, output_backend="webgl", lod_factor=50,
                tools="", toolbar_location=None, sizing_mode='stretch_width')
    select.scatter(x="x", y="y", source=source, view=view, alpha=0.6, line_color=None, fill_color=colormap)
    range_tool = RangeTool(x_range=p.x_range, y_range=p.y_range)
    range_tool.overlay.fill_color = "navy"
    range_tool.overlay.fill_alpha = 0.2
    select.add_tools(range_tool)

    # range_tool = RangeTool(x_range=p.x_range)
    # range_tool.overlay.fill_color = "navy"
    # range_tool.overlay.fill_alpha = 0.2
    # p.add_tools(range_tool)
    lasso = LassoSelectTool(
        # overlay=PolyAnnotation(),
        renderers=[active])
    p.add_tools(lasso)
    p.toolbar.active_drag = lasso
    p.toolbar.active_inspect = None
    p.toolbar.logo = None

    options = row(x_col, y_col)
    return options, column(p, select, sizing_mode='stretch_width')


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
