import base64
from pathlib import Path
from typing import Optional, Tuple

import bokeh.transform
import numpy as np
import pandas as pd
from bokeh.palettes import Category10, Cividis256
from bokeh.transform import factor_cmap, linear_cmap
from wasabi import msg


def get_color_mapping(
    df: pd.DataFrame,
) -> Tuple[Optional[bokeh.transform.transform], pd.DataFrame]:
    """Creates a color mapping"""
    if "color" not in df.columns:
        return None, df

    color_datatype = str(df["color"].dtype)
    if color_datatype == "object":
        df["color"] = df["color"].apply(
            lambda x: str(x) if not (type(x) == float and np.isnan(x)) else x
        )
        all_values = list(df["color"].dropna().unique())
        if len(all_values) == 2:
            all_values.extend([""])
        elif len(all_values) > len(Category10) + 2:
            raise ValueError(
                f"Too many classes defined, the limit for visualisation is {len(Category10) + 2}. "
                f"Got {len(all_values)}."
            )
        mapper = factor_cmap(
            field_name="color",
            palette=Category10[len(all_values)],
            factors=all_values,
            nan_color="grey",
        )
    elif color_datatype.startswith("float") or color_datatype.startswith("int"):
        all_values = df["color"].dropna().values
        mapper = linear_cmap(
            field_name="color",
            palette=Cividis256,
            low=all_values.min(),
            high=all_values.max(),
            nan_color="grey",
        )
    else:
        raise TypeError(
            f"We currently only support the following type for 'color' column: 'int*', 'float*', 'object'. "
            f"Got {color_datatype}."
        )
    return mapper, df


def save_file(dataf: pd.DataFrame, highlighted_idx: pd.Series, filename: str) -> None:
    path = Path(filename)
    subset = dataf.iloc[highlighted_idx]
    if path.suffix == ".jsonl":
        subset.to_json(path, orient="records", lines=True)
    else:
        subset.to_csv(path, index=False)
    msg.good(f"Saved {len(subset)} exampes over at {path}.", spaced=True)


def determine_keyword(text, keywords):
    for kw in keywords:
        if kw in text:
            return kw
    return "none"


def encode_image(path):
    if type(path) == str and path.startswith('http'):
        return f'<img style="object-fit: scale-down;" width="100%" height="100%" src="{path}">'
    else:
        with open(path, "rb") as image_file:
            enc_str = base64.b64encode(image_file.read()).decode("utf-8")
        return f'<img style="object-fit: scale-down;" width="100%" height="100%" src="data:image/png;base64,{enc_str}">'
    

def read_file(path: str, keywords=None):
    path = Path(path)
    if path.suffix == ".jsonl":
        dataf = pd.read_json(path, orient="records", lines=True)
    elif path.suffix == ".csv":
        dataf = pd.read_csv(path)
    else:
        msg.fail(
            f"Bulk only supports .csv or .jsonl files, got `{str(path)}`.",
            exits=True,
            spaced=True,
        )
    
    dataf["alpha"] = 0.5
    if keywords:
        dataf["color"] = [determine_keyword(str(t), keywords) for t in dataf["text"]]
        dataf["alpha"] = [0.4 if c == "none" else 1 for c in dataf["color"]]
    if "path" in dataf.columns:
        dataf["image"] = [encode_image(p) for p in dataf["path"]]
    return get_color_mapping(dataf)


def js_funcs():
    return """
function table_to_csv(source) {
    const subset_col = ["text", "path"].filter(_ => source.data[_])[0];
    let subset = {};
    subset[subset_col] = source.data[subset_col]
    console.log(subset_col, subset);
    const columns = Object.keys(subset)
    const nrows = source.get_length()
    const lines = [columns.join(',')]

    for (let i = 0; i < nrows; i++) {
        let row = [];
        for (let j = 0; j < columns.length; j++) {
            const column = columns[j]
            console.log(column, source.data);
            row.push(source.data[column][i].toString())
        }
        lines.push(row)
    }
    return lines.join('\\n').concat('\\n')
}
"""

def download_js_code():
    return js_funcs() + """
const filename = document.getElementsByName("filename")[0].value
const filetext = table_to_csv(source)
const blob = new Blob([filetext], { type: 'text/csv;charset=utf-8;' })

if (navigator.msSaveBlob) {
    navigator.msSaveBlob(blob, filename)
} else {
    const link = document.createElement('a')
    link.href = URL.createObjectURL(blob)
    link.download = filename
    link.target = '_blank'
    link.style.visibility = 'hidden'
    link.dispatchEvent(new MouseEvent('click'))
}
"""
