from bokeh.models import AjaxDataSource
from bokeh.util.compiler import TypeScript
from bokeh.core.properties import String, Required, Dict, Any, Nullable


class GraphQLDataSource(AjaxDataSource):
    query = Required(String())
    variables = Nullable(Dict(String, Any))
    # https://stackoverflow.com/questions/59166017/making-credentialed-requests-with-bokeh-ajaxdatasource
    # https://github.com/bokeh/bokeh/blob/bba83ab0db6986f5b8e82ab01e2ad2d09b877cea/bokehjs/src/lib/models/sources/ajax_data_source.ts#L58
    __implementation__ = "graphql_datasource.ts"
