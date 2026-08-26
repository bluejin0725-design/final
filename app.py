from __future__ import annotations

from html import escape
import gzip
import json
import math
from pathlib import Path
from urllib.parse import quote

import altair as alt
import folium
import pandas as pd
import statsmodels.api as sm
import streamlit as st
from branca.colormap import LinearColormap
from folium.plugins import Fullscreen, LocateControl
from folium.utilities import JsCode
from shapely import voronoi_polygons
from shapely.affinity import scale
from shapely.geometry import MultiPoint, Point, mapping, shape
from shapely.ops import unary_union
from statsmodels.stats.outliers_influence import variance_inflation_factor
from streamlit_folium import st_folium


st.set_page_config(
    page_title="S-DoT 300m 토지피복 분석",
    page_icon="🗺️",
    layout="wide",
    initial_sidebar_state="expanded",
)

PLACES = {
    "SHP 영역": ([37.5625, 126.9875], 13),
    "서울": ([37.5665, 126.9780], 11),
}

BASEMAPS = {
    "OpenStreetMap": ("OpenStreetMap", "© OpenStreetMap contributors"),
    "CartoDB 밝은 지도": ("CartoDB positron", "© OpenStreetMap contributors © CARTO"),
    "CartoDB 어두운 지도": ("CartoDB dark_matter", "© OpenStreetMap contributors © CARTO"),
}

LAND_COVER_SOURCE = "https://aid.mcee.go.kr/api/land.do"
SDOT_SOURCE = "https://data.seoul.go.kr/dataList/OA-22833/A/1/datasetView.do"
DATA_FILE_NAME = "sdot_nature_20260720_20260726.parquet"
SHP_IMAGE_FILE_NAME = "landcover_detail_2024.png"
SHP_META_FILE_NAME = "landcover_detail_2024.json"
BUFFER_FILE_NAME = "sdot_buffers_300m.geojson.gz"
BUFFER_STATS_FILE_NAME = "sdot_buffer300m_landcover.parquet"

TEMPERATURE_COLUMN = "temperature_c"
TEMPERATURE_MIN = -30.0
TEMPERATURE_MAX = 50.0
TEMPERATURE_COLORS = ["#fff5f0", "#fcbba1", "#fb6a4a", "#de2d26", "#7f0000"]

# 사용자가 분석 제외를 요청한 대분류와 그 하위 분류 코드입니다.
# 2=농업지역, 5=습지, 6=나지이며 지도 원본 오버레이에는 남겨 두되
# 관계 지도·표·경향·회귀분석에서는 제외합니다.
EXCLUDED_LARGE_CLASS_PREFIXES = {"2", "5", "6"}
EXCLUDED_LARGE_CLASS_NAMES = {"농업지역", "습지", "나지"}

# 분석에서는 산림지역과 초지를 하나의 '녹지' 변수로 합칩니다.
ANALYSIS_COVER_COMPONENTS = {
    "시가화건조지역": {"1"},
    "녹지": {"3", "4"},
    "수역": {"7"},
}
ANALYSIS_COVER_COLORS = {
    "시가화건조지역(%)": "#d95f02",
    "녹지(%)": "#2f855a",
    "수역(%)": "#3182bd",
}

# 행은 온도(낮음→높음), 열은 선택 피복비율(낮음→높음)입니다.
# 위쪽으로 갈수록 붉어지고, 오른쪽으로 갈수록 명도를 낮춰 어둡게 보입니다.
BIVARIATE_COLORS = [
    ["#fff5f0", "#b8b0ad", "#736e6c"],
    ["#fb6a4a", "#b54c35", "#713021"],
    ["#cb181d", "#921115", "#5b0b0d"],
]
BIVARIATE_REFERENCE = (
    "https://pro.arcgis.com/en/pro-app/3.6/help/mapping/"
    "layer-properties/bivariate-colors.htm"
)


def find_asset(file_name: str) -> Path | None:
    app_dir = Path(__file__).resolve().parent
    candidates = (app_dir / file_name, app_dir / "data" / file_name)
    return next((path for path in candidates if path.is_file()), None)


@st.cache_data(show_spinner="S-DoT 데이터를 불러오는 중입니다…")
def load_sdot_data(path: str) -> pd.DataFrame:
    return pd.read_parquet(path)


@st.cache_data(show_spinner=False)
def load_json(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def load_gzip_json(path: str) -> dict:
    with gzip.open(path, "rt", encoding="utf-8") as source:
        return json.load(source)


@st.cache_data(show_spinner="300m 버퍼 분석결과를 불러오는 중입니다…")
def load_buffer_stats(path: str) -> pd.DataFrame:
    return pd.read_parquet(path)


def class_color(class_code: str) -> str:
    code = str(class_code)
    base_hues = {"1": 355, "2": 28, "3": 50, "4": 135, "5": 175, "6": 210, "7": 265}
    base_hue = base_hues.get(code[:1], 195)
    try:
        detail_offset = (int(code) % 100) * 1.7
    except ValueError:
        detail_offset = sum(ord(character) for character in code) % 40
    return f"hsl({(base_hue + detail_offset) % 360:.0f}, 68%, 46%)"


def class_code_prefix(value: object) -> str:
    """대·중·세분류 코드에서 대분류를 나타내는 첫 자리를 반환합니다."""
    normalized = str(value).strip().split(".")[0]
    return normalized[:1]


def filter_analysis_landcover_rows(data: pd.DataFrame) -> pd.DataFrame:
    """농업지역·습지·나지 및 해당 하위 분류를 분석 데이터에서 제외합니다."""
    prefixes = data["class_code"].map(class_code_prefix)
    names = data["class_name"].astype(str)
    excluded = prefixes.isin(EXCLUDED_LARGE_CLASS_PREFIXES) | names.isin(
        EXCLUDED_LARGE_CLASS_NAMES
    )
    return data.loc[~excluded].copy()


def build_analysis_composition(
    stats: pd.DataFrame,
    min_coverage: float,
) -> pd.DataFrame:
    """센서별 시가화건조·녹지·수역 비율을 원래 피복면적 기준으로 계산합니다."""
    large_all = stats[
        (stats["level"] == "대분류")
        & (pd.to_numeric(stats["coverage_pct"], errors="coerce") >= min_coverage)
    ].copy()
    if large_all.empty:
        return pd.DataFrame(
            columns=[*ANALYSIS_COVER_COMPONENTS, "SHP 피복률(%)"]
        )

    large_all["serial"] = large_all["serial"].astype(str)
    large_all["share_pct"] = pd.to_numeric(large_all["share_pct"], errors="coerce").fillna(0)
    coverage = large_all.groupby("serial")["coverage_pct"].first()
    kept = filter_analysis_landcover_rows(large_all)
    kept["large_prefix"] = kept["class_code"].map(class_code_prefix)

    composition = pd.DataFrame(index=coverage.index)
    for label, prefixes in ANALYSIS_COVER_COMPONENTS.items():
        component = (
            kept.loc[kept["large_prefix"].isin(prefixes)]
            .groupby("serial")["share_pct"]
            .sum()
        )
        composition[label] = component.reindex(composition.index, fill_value=0.0)

    # 제외 대상만 존재하는 센서는 분석 표본에서 제거합니다.
    composition = composition.loc[
        composition[list(ANALYSIS_COVER_COMPONENTS)].sum(axis=1) > 0
    ].copy()
    composition["SHP 피복률(%)"] = coverage.reindex(composition.index)
    return composition


def tertile_thresholds(values: pd.Series) -> tuple[float, float]:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    return float(clean.quantile(1 / 3)), float(clean.quantile(2 / 3))


def tertile_class(value: float, thresholds: tuple[float, float]) -> int:
    if value <= thresholds[0]:
        return 0
    if value <= thresholds[1]:
        return 1
    return 2


def tertile_label(class_index: int) -> str:
    return ("낮음", "중간", "높음")[class_index]


def pie_chart_data_url(
    row: pd.Series,
    composition_columns: list[str],
    colors: dict[str, str],
) -> str:
    """센서별 토지피복 구성비를 표 안에서 표시할 SVG 파이차트로 변환합니다."""
    segments = []
    accumulated = 0.0
    for column in composition_columns:
        raw_share = pd.to_numeric(row.get(column, 0), errors="coerce")
        share = 0.0 if pd.isna(raw_share) else float(raw_share)
        if share <= 0:
            continue
        segments.append(
            '<circle cx="36" cy="36" r="18" fill="none" '
            f'stroke="{colors[column]}" stroke-width="36" pathLength="100" '
            f'stroke-dasharray="{share:.4f} {100 - share:.4f}" '
            f'stroke-dashoffset="{-accumulated:.4f}" transform="rotate(-90 36 36)"/>'
        )
        accumulated += share
    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" width="72" height="72" viewBox="0 0 72 72">'
        '<circle cx="36" cy="36" r="36" fill="#e5e7eb"/>'
        + "".join(segments)
        + '</svg>'
    )
    return "data:image/svg+xml;utf8," + quote(svg, safe="")


def clean_temperature(data: pd.DataFrame) -> pd.DataFrame:
    cleaned = data.copy()
    temperature = pd.to_numeric(cleaned[TEMPERATURE_COLUMN], errors="coerce")
    cleaned[TEMPERATURE_COLUMN] = temperature.where(
        temperature.between(TEMPERATURE_MIN, TEMPERATURE_MAX)
    )
    return cleaned


def build_base_map(center: list[float], zoom: int, basemap_name: str) -> folium.Map:
    tiles, attribution = BASEMAPS[basemap_name]
    map_object = folium.Map(
        location=center,
        zoom_start=zoom,
        tiles=tiles,
        attr=attribution,
        control_scale=True,
        prefer_canvas=True,
    )
    Fullscreen(position="topright", title="전체 화면", title_cancel="전체 화면 종료").add_to(map_object)
    LocateControl(position="topright", strings={"title": "내 위치"}).add_to(map_object)
    return map_object


def add_shp_overlay(
    map_object: folium.Map,
    image_path: Path,
    metadata: dict,
    opacity: float,
) -> None:
    folium.raster_layers.ImageOverlay(
        image=str(image_path),
        bounds=metadata["bounds"],
        name="2024 세분류 토지피복 SHP",
        opacity=opacity,
        interactive=True,
        cross_origin=False,
        zindex=2,
        pixelated=False,
        show=True,
    ).add_to(map_object)

    legend_rows = "".join(
        f'<div><span style="display:inline-block;width:11px;height:11px;'
        f'background:{item["color"]};margin-right:6px;border-radius:2px"></span>'
        f'{escape(str(item["name"]))}</div>'
        for item in metadata["legend"]
    )
    legend_html = f"""
    <div style="position:fixed;left:12px;bottom:28px;z-index:9999;background:rgba(255,255,255,.93);
                color:#18201e;padding:9px 11px;border-radius:7px;border:1px solid #cdd6d2;
                font:12px/1.5 sans-serif;box-shadow:0 1px 5px rgba(0,0,0,.18)">
      <b>2024 SHP 대분류</b>{legend_rows}
    </div>
    """
    map_object.get_root().html.add_child(folium.Element(legend_html))


def add_buffer_layer(
    parent_group: folium.FeatureGroup,
    buffer_geojson: dict,
    stats: pd.DataFrame,
    level: str,
    min_coverage: float,
    allowed_serials: set[str],
    selected_serial: str | None,
) -> int:
    level_stats = stats[stats["level"] == level].copy()
    dominant_indexes = level_stats.groupby("serial")["share_pct"].idxmax()
    dominant = level_stats.loc[
        dominant_indexes, ["serial", "class_code", "class_name", "share_pct"]
    ].set_index("serial")
    composition_by_serial = {
        str(serial): [
            {
                "name": escape(str(row.class_name)),
                "share": round(float(row.share_pct), 2),
                "color": class_color(str(row.class_code)),
            }
            for row in group.sort_values("share_pct", ascending=False).itertuples(index=False)
            if float(row.share_pct) > 0
        ]
        for serial, group in level_stats.groupby("serial", sort=False)
    }

    features = []
    for source_feature in buffer_geojson["features"]:
        properties = dict(source_feature["properties"])
        serial = str(properties["serial"])
        coverage = float(properties.get("coverage_pct", 0))
        if (
            coverage < min_coverage
            or serial not in dominant.index
            or serial not in allowed_serials
        ):
            continue
        dominant_row = dominant.loc[serial]
        properties.update(
            {
                "dominant_code": str(dominant_row["class_code"]),
                "dominant_name": str(dominant_row["class_name"]),
                "dominant_share": round(float(dominant_row["share_pct"]), 2),
                "coverage_pct": round(coverage, 2),
                "composition_level": level,
                "composition": composition_by_serial.get(serial, []),
                "selected": serial == selected_serial,
            }
        )
        features.append(
            {
                "type": "Feature",
                "properties": properties,
                "geometry": source_feature["geometry"],
            }
        )

    def style_function(feature: dict) -> dict:
        properties = feature["properties"]
        selected = properties["selected"]
        color = class_color(properties["dominant_code"])
        return {
            "color": "#111827" if selected else color,
            "fillColor": color,
            "weight": 4 if selected else 2,
            "fillOpacity": 0.34 if selected else 0.17,
            "dashArray": None if properties["coverage_pct"] >= 99 else "5 4",
        }

    pie_tooltip = JsCode(
        """
        function(feature, layer) {
            const properties = feature.properties || {};
            const items = properties.composition || [];
            let accumulated = 0;
            const colorStops = items.map(function(item) {
                const start = accumulated;
                accumulated = Math.min(100, accumulated + Number(item.share || 0));
                return item.color + " " + start.toFixed(2) + "% " + accumulated.toFixed(2) + "%";
            });
            const pieBackground = colorStops.length
                ? "conic-gradient(" + colorStops.join(",") + ")"
                : "#d1d5db";
            const legendRows = items.map(function(item) {
                return '<div style="display:grid;grid-template-columns:12px 1fr auto;gap:7px;'
                    + 'align-items:center;margin:3px 0">'
                    + '<span style="width:11px;height:11px;border-radius:2px;background:'
                    + item.color + '"></span>'
                    + '<span style="overflow:hidden;text-overflow:ellipsis;white-space:nowrap">'
                    + item.name + '</span>'
                    + '<b style="font-variant-numeric:tabular-nums">'
                    + Number(item.share).toFixed(1) + '%</b></div>';
            }).join("");
            const tooltipHtml = '<div style="width:310px;color:#18201e;font:12px/1.4 sans-serif">'
                + '<div style="font-size:13px;font-weight:700;margin-bottom:7px">'
                + properties.serial + ' · 300m ' + properties.composition_level + ' 구성비</div>'
                + '<div style="display:grid;grid-template-columns:112px 1fr;gap:12px;align-items:center">'
                + '<div style="width:108px;height:108px;border-radius:50%;background:'
                + pieBackground + ';box-shadow:inset 0 0 0 1px rgba(0,0,0,.12)"></div>'
                + '<div style="max-height:180px;overflow-y:auto;padding-right:3px">'
                + legendRows + '</div></div>'
                + '<div style="margin-top:7px;padding-top:6px;border-top:1px solid #d7dfdc;'
                + 'color:#59635f">SHP 피복률 '
                + Number(properties.coverage_pct).toFixed(1) + '%</div></div>';
            layer.bindTooltip(tooltipHtml, {
                sticky: true,
                direction: "auto",
                opacity: 0.97,
                maxWidth: 350,
                className: "landcover-pie-tooltip"
            });
        }
        """
    )

    folium.GeoJson(
        {"type": "FeatureCollection", "features": features},
        name="300m 버퍼",
        style_function=style_function,
        highlight_function=lambda _: {"weight": 4, "fillOpacity": 0.30},
        show=True,
        control=False,
        on_each_feature=pie_tooltip,
        popup=folium.GeoJsonPopup(
            fields=["serial", "address", "coverage_pct"],
            aliases=["센서", "주소", "SHP 피복률(%)"],
            localize=True,
        ),
    ).add_to(parent_group)
    return len(features)





def partition_overlapping_buffers_as_union(features: list[dict]) -> list[dict]:
    """중첩 버퍼를 최근접 센서 영역으로 분할해 합집합을 한 번만 채웁니다."""
    if len(features) <= 1:
        return features

    records = []
    for feature in features:
        properties = feature["properties"]
        geometry = shape(feature["geometry"])
        if geometry.is_empty:
            continue
        longitude = pd.to_numeric(properties.get("longitude"), errors="coerce")
        latitude = pd.to_numeric(properties.get("latitude"), errors="coerce")
        if pd.isna(longitude) or pd.isna(latitude):
            center = geometry.centroid
            longitude, latitude = center.x, center.y
        records.append(
            {
                "feature": feature,
                "geometry": geometry,
                "point": Point(float(longitude), float(latitude)),
            }
        )
    if len(records) <= 1:
        return [record["feature"] for record in records]

    coordinate_count = len(
        {(record["point"].x, record["point"].y) for record in records}
    )
    union_geometry = unary_union([record["geometry"] for record in records])

    def ordered_non_overlapping_partition() -> list[dict]:
        """동일 좌표 등으로 보로노이 분할이 불가능할 때도 중복 채색을 막습니다."""
        assigned = None
        fallback_features = []
        for record in records:
            geometry = record["geometry"]
            piece = geometry if assigned is None else geometry.difference(assigned)
            if piece.is_empty:
                continue
            fallback_feature = dict(record["feature"])
            fallback_feature["geometry"] = mapping(piece)
            fallback_features.append(fallback_feature)
            assigned = (
                geometry
                if assigned is None
                else unary_union([assigned, geometry])
            )
        return fallback_features

    if coordinate_count != len(records):
        return ordered_non_overlapping_partition()

    # 서울 위도에서 경도 1도의 실제 거리가 더 짧은 점을 보정한 뒤 보로노이를 만듭니다.
    mean_latitude = sum(record["point"].y for record in records) / len(records)
    longitude_scale = max(
        float(math.cos(math.radians(mean_latitude))),
        0.5,
    )
    projected_points = [
        scale(
            record["point"],
            xfact=longitude_scale,
            yfact=1.0,
            origin=(0.0, 0.0),
        )
        for record in records
    ]
    projected_union = scale(
        union_geometry,
        xfact=longitude_scale,
        yfact=1.0,
        origin=(0.0, 0.0),
    )

    try:
        cell_collection = voronoi_polygons(
            MultiPoint(projected_points),
            extend_to=projected_union.envelope,
        )
        unused_cells = list(cell_collection.geoms)
        partitioned_features = []
        for record, projected_point in zip(records, projected_points):
            cell_index = min(
                range(len(unused_cells)),
                key=lambda index: unused_cells[index].distance(projected_point),
            )
            projected_cell = unused_cells.pop(cell_index)
            cell = scale(
                projected_cell,
                xfact=1.0 / longitude_scale,
                yfact=1.0,
                origin=(0.0, 0.0),
            )
            piece = record["geometry"].intersection(cell)
            if piece.is_empty:
                continue
            partitioned_feature = dict(record["feature"])
            partitioned_feature["geometry"] = mapping(piece)
            partitioned_features.append(partitioned_feature)
        return partitioned_features
    except Exception:
        return ordered_non_overlapping_partition()

def add_bivariate_buffer_layer(
    parent_group: folium.FeatureGroup,
    map_object: folium.Map,
    buffer_geojson: dict,
    stats: pd.DataFrame,
    snapshot: pd.DataFrame,
    min_coverage: float,
    allowed_serials: set[str],
    selected_serial: str | None,
    cover_name: str,
) -> int:
    """선택 피복비율과 온도의 3×3 이변량 관계 지도를 추가합니다."""
    composition = build_analysis_composition(stats, min_coverage)
    if composition.empty or cover_name not in composition.columns:
        return 0

    temperature = snapshot[["serial", TEMPERATURE_COLUMN]].copy()
    temperature["serial"] = temperature["serial"].astype(str)
    temperature[TEMPERATURE_COLUMN] = pd.to_numeric(
        temperature[TEMPERATURE_COLUMN], errors="coerce"
    )
    temperature = (
        temperature.dropna(subset=[TEMPERATURE_COLUMN])
        .drop_duplicates("serial", keep="last")
        .set_index("serial")
    )

    analysis = composition[[cover_name]].join(temperature, how="inner")
    analysis = analysis.loc[
        analysis.index.astype(str).isin(allowed_serials)
    ].copy()
    if analysis.empty:
        return 0

    cover_thresholds = tertile_thresholds(analysis[cover_name])
    temperature_thresholds = tertile_thresholds(analysis[TEMPERATURE_COLUMN])
    analysis["cover_class"] = analysis[cover_name].map(
        lambda value: tertile_class(float(value), cover_thresholds)
    )
    analysis["temperature_class"] = analysis[TEMPERATURE_COLUMN].map(
        lambda value: tertile_class(float(value), temperature_thresholds)
    )

    features = []
    for source_feature in buffer_geojson["features"]:
        properties = dict(source_feature["properties"])
        serial = str(properties["serial"])
        if serial not in analysis.index:
            continue
        row = analysis.loc[serial]
        cover_class = int(row["cover_class"])
        temperature_class = int(row["temperature_class"])
        properties.update(
            {
                "cover_name": cover_name,
                "cover_share": round(float(row[cover_name]), 2),
                "temperature_c": round(float(row[TEMPERATURE_COLUMN]), 2),
                "cover_level": tertile_label(cover_class),
                "temperature_level": tertile_label(temperature_class),
                "bivariate_color": BIVARIATE_COLORS[temperature_class][cover_class],
                "coverage_pct": round(float(properties.get("coverage_pct", 0)), 2),
                "selected": serial == selected_serial,
            }
        )
        features.append(
            {
                "type": "Feature",
                "properties": properties,
                "geometry": source_feature["geometry"],
            }
        )

    # 겹친 면적은 최근접 센서 한 곳에만 배정해 전체 버퍼 합집합을 한 번만 칠합니다.
    features = partition_overlapping_buffers_as_union(features)

    def style_function(feature: dict) -> dict:
        properties = feature["properties"]
        return {
            "color": "#111827" if properties["selected"] else "transparent",
            "fillColor": properties["bivariate_color"],
            "weight": 4 if properties["selected"] else 0,
            "fillOpacity": 0.82 if properties["selected"] else 0.72,
            "dashArray": None,
        }

    folium.GeoJson(
        {"type": "FeatureCollection", "features": features},
        name=f"{cover_name} × 온도",
        style_function=style_function,
        highlight_function=lambda _: {
            "weight": 3,
            "color": "#111827",
            "fillOpacity": 0.84,
        },
        tooltip=folium.GeoJsonTooltip(
            fields=[
                "serial",
                "address",
                "cover_share",
                "temperature_c",
                "cover_level",
                "temperature_level",
                "coverage_pct",
            ],
            aliases=[
                "센서",
                "주소",
                f"{cover_name} 비율(%)",
                "온도(℃)",
                f"{cover_name} 삼분위",
                "온도 삼분위",
                "SHP 피복률(%)",
            ],
            localize=True,
            sticky=True,
        ),
        popup=folium.GeoJsonPopup(
            fields=["serial", "cover_share", "temperature_c"],
            aliases=["센서", f"{cover_name} 비율(%)", "온도(℃)"],
            localize=True,
        ),
        show=True,
        control=False,
    ).add_to(parent_group)

    legend_cells = []
    for temperature_class in (2, 1, 0):
        legend_cells.append(
            f'<span style="font-size:10px;text-align:right;padding-right:4px">'
            f'{tertile_label(temperature_class)}</span>'
        )
        for cover_class in (0, 1, 2):
            color = BIVARIATE_COLORS[temperature_class][cover_class]
            legend_cells.append(
                f'<span style="width:29px;height:29px;background:{color};'
                f'display:block;border:1px solid rgba(255,255,255,.7)"></span>'
            )
    legend_html = f"""
    <div style="position:fixed;right:14px;bottom:38px;z-index:9999;
                background:rgba(255,255,255,.95);color:#18201e;padding:10px 12px;
                border-radius:7px;border:1px solid #cdd6d2;font:11px/1.35 sans-serif;
                box-shadow:0 1px 5px rgba(0,0,0,.18)">
      <b>{escape(cover_name)} 비율 × 온도</b>
      <div style="margin-top:6px;display:grid;grid-template-columns:38px repeat(3,29px);
                  gap:1px;align-items:center">
        <span style="text-align:right;padding-right:4px;font-weight:700">온도 ↑</span>
        <span></span><span></span><span></span>
        {''.join(legend_cells)}
        <span></span><span style="text-align:center">낮음</span>
        <span style="text-align:center">중간</span><span style="text-align:center">높음</span>
      </div>
      <div style="margin:4px 0 0 39px;text-align:center">{escape(cover_name)} 비율 →</div>
      <div style="margin-top:6px;color:#59635f;border-top:1px solid #d7dfdc;padding-top:5px">
        붉을수록 고온 · 어두울수록 피복비율 높음<br>
        각 변수 삼분위 · n={len(features)}<br>
        피복 경계 {cover_thresholds[0]:.1f}, {cover_thresholds[1]:.1f}% ·
        온도 경계 {temperature_thresholds[0]:.1f}, {temperature_thresholds[1]:.1f}℃
      </div>
    </div>
    """
    map_object.get_root().html.add_child(folium.Element(legend_html))
    return len(features)


def add_temperature_points(
    parent_group: folium.FeatureGroup,
    map_object: folium.Map,
    data: pd.DataFrame,
    show_legend: bool = True,
) -> None:
    data = data.dropna(subset=[TEMPERATURE_COLUMN]).copy()
    measured = data[TEMPERATURE_COLUMN].dropna()
    if measured.empty:
        value_min, value_max = 0.0, 1.0
    else:
        value_min = float(measured.quantile(0.05))
        value_max = float(measured.quantile(0.95))
        if value_min == value_max:
            value_max = value_min + 1.0
    color_scale = LinearColormap(TEMPERATURE_COLORS, vmin=value_min, vmax=value_max)

    for row in data.itertuples(index=False):
        value = getattr(row, TEMPERATURE_COLUMN)
        has_value = pd.notna(value)
        color = color_scale(float(value)) if has_value else "#8b949e"
        value_text = f"{float(value):,.2f} ℃" if has_value else "유효 측정값 없음"
        popup_html = f"""
        <div style="font-family:sans-serif;min-width:230px;line-height:1.55">
          <strong>{escape(str(row.serial))}</strong><br>
          {escape(str(row.address))}<hr style="margin:.45rem 0">
          <b>온도:</b> {escape(value_text)}<br>
          <b>측정시각:</b> {row.measured_at:%Y-%m-%d %H:%M}<br>
          <b>지역:</b> {escape(str(row.district))} · {escape(str(row.neighborhood))}
        </div>
        """
        folium.CircleMarker(
            location=[float(row.latitude), float(row.longitude)],
            radius=5.5,
            color="#ffffff",
            weight=1,
            fill=True,
            fill_color=color,
            fill_opacity=0.92,
            tooltip=f"{row.serial} · {value_text}",
            popup=folium.Popup(popup_html, max_width=330),
        ).add_to(parent_group)

    if show_legend and not measured.empty:
        gradient = ",".join(TEMPERATURE_COLORS)
        legend_html = f"""
        <div style="position:fixed;right:14px;bottom:42px;z-index:9999;width:210px;
                    background:rgba(255,255,255,.94);color:#18201e;padding:9px 11px;
                    border-radius:7px;border:1px solid #cdd6d2;font:12px/1.35 sans-serif;
                    box-shadow:0 1px 5px rgba(0,0,0,.18)">
          <b>S-DoT 온도 (℃)</b>
          <div style="height:11px;margin:6px 0 2px;background:linear-gradient(to right,{gradient});"></div>
          <div style="display:flex;justify-content:space-between">
            <span>{value_min:.1f}</span><span>5–95 백분위</span><span>{value_max:.1f}</span>
          </div>
        </div>
        """
        map_object.get_root().html.add_child(folium.Element(legend_html))


def build_sensor_table(
    stats: pd.DataFrame,
    snapshot: pd.DataFrame,
    min_coverage: float,
) -> tuple[pd.DataFrame, list[str]]:
    """분석 대상 대분류만 사용해 센서별 피복 구성표를 만듭니다."""
    snapshot = snapshot.dropna(subset=[TEMPERATURE_COLUMN]).copy()
    snapshot["serial"] = snapshot["serial"].astype(str)

    analysis_composition = build_analysis_composition(stats, min_coverage)
    if analysis_composition.empty:
        return pd.DataFrame(), []

    cover_labels = list(ANALYSIS_COVER_COMPONENTS)
    composition = analysis_composition[cover_labels].rename(
        columns={label: f"{label}(%)" for label in cover_labels}
    )
    composition_columns = list(composition.columns)
    class_colors = {
        column: ANALYSIS_COVER_COLORS[column] for column in composition_columns
    }

    dominant_label = composition.idxmax(axis=1).str.removesuffix("(%)")
    dominant_share = composition.max(axis=1)
    dominant = pd.DataFrame(
        {
            "serial": composition.index.astype(str),
            "우세 용도": dominant_label.to_numpy(),
            "우세 비율(%)": dominant_share.to_numpy(),
            "SHP 피복률(%)": analysis_composition["SHP 피복률(%)"].to_numpy(),
        }
    )
    composition = composition.reset_index().rename(columns={"index": "serial"})
    composition["serial"] = composition["serial"].astype(str)

    sensor_columns = [
        "serial",
        "district",
        "neighborhood",
        "address",
        TEMPERATURE_COLUMN,
    ]
    table = (
        snapshot[sensor_columns]
        .merge(dominant, on="serial", how="inner")
        .merge(composition, on="serial", how="left")
        .rename(
            columns={
                "serial": "센서",
                "district": "자치구",
                "neighborhood": "행정동",
                "address": "주소",
                TEMPERATURE_COLUMN: "온도(℃)",
            }
        )
    )
    numeric_columns = [
        "온도(℃)",
        "SHP 피복률(%)",
        "우세 비율(%)",
        *composition_columns,
    ]
    table[numeric_columns] = table[numeric_columns].apply(
        pd.to_numeric, errors="coerce"
    ).round(2)
    pie_column_position = table.columns.get_loc("온도(℃)") + 1
    table.insert(
        pie_column_position,
        "피복 구성비",
        table.apply(
            pie_chart_data_url,
            axis=1,
            composition_columns=composition_columns,
            colors=class_colors,
        ),
    )
    return table, composition_columns


def build_trend_table(sensor_table: pd.DataFrame, composition_columns: list[str]) -> pd.DataFrame:
    valid = sensor_table.dropna(subset=["온도(℃)"]).copy()
    rows = []
    for column in composition_columns:
        shares = valid[column].astype(float)
        temperatures = valid["온도(℃)"].astype(float)
        sample_size = len(valid)
        group_size = max(1, sample_size // 4)
        high_indexes = shares.nlargest(group_size).index
        low_indexes = shares.nsmallest(group_size).index
        correlation = shares.corr(temperatures) if shares.nunique() > 1 else float("nan")
        high_temperature = temperatures.loc[high_indexes].mean()
        low_temperature = temperatures.loc[low_indexes].mean()
        temperature_gap = high_temperature - low_temperature
        if pd.isna(correlation) or abs(correlation) < 0.2:
            tendency = "뚜렷하지 않음"
        elif correlation > 0:
            tendency = "구성비 증가 시 온도 상승 경향"
        else:
            tendency = "구성비 증가 시 온도 하락 경향"
        rows.append(
            {
                "토지피복 용도": column.removesuffix("(%)"),
                "분석 센서수": sample_size,
                "해당 용도 포함 센서수": int((shares > 0).sum()),
                "평균 구성비(%)": shares.mean(),
                "온도 상관계수": correlation,
                "구성비 상위25% 평균온도(℃)": high_temperature,
                "구성비 하위25% 평균온도(℃)": low_temperature,
                "온도차(℃)": temperature_gap,
                "경향": tendency,
            }
        )
    trend = pd.DataFrame(rows)
    numeric = [
        "평균 구성비(%)",
        "온도 상관계수",
        "구성비 상위25% 평균온도(℃)",
        "구성비 하위25% 평균온도(℃)",
        "온도차(℃)",
    ]
    trend[numeric] = trend[numeric].round(3)
    return trend


def build_landcover_multiple_regression(
    sensor_table: pd.DataFrame,
) -> tuple[pd.DataFrame | None, pd.DataFrame | None, dict[str, object]]:
    """시가화건조·녹지·수역 비율로 온도를 설명하는 다중 OLS를 추정합니다."""
    predictor_columns = [
        "시가화건조지역(%)",
        "녹지(%)",
        "수역(%)",
    ]
    required_columns = ["센서", "자치구", *predictor_columns, "온도(℃)"]
    missing_columns = [
        column for column in required_columns if column not in sensor_table.columns
    ]
    if missing_columns:
        return None, None, {
            "reason": f"회귀분석에 필요한 열이 없습니다: {', '.join(missing_columns)}"
        }

    analysis = sensor_table[required_columns].copy()
    numeric_columns = [*predictor_columns, "온도(℃)"]
    analysis[numeric_columns] = analysis[numeric_columns].apply(
        pd.to_numeric, errors="coerce"
    )
    analysis = analysis.dropna(subset=numeric_columns)
    for column in predictor_columns:
        analysis = analysis[analysis[column].between(0, 100)]

    varying_predictors = [
        column
        for column in predictor_columns
        if analysis[column].nunique() >= 2 and float(analysis[column].std()) > 0
    ]
    dropped_predictors = [
        column for column in predictor_columns if column not in varying_predictors
    ]
    minimum_rows = max(10, len(varying_predictors) + 3)
    if len(analysis) < minimum_rows:
        return None, None, {
            "reason": f"유효한 센서가 최소 {minimum_rows}개 필요합니다."
        }
    if len(varying_predictors) < 2:
        return None, None, {
            "reason": "선택 조건에서 변화가 있는 피복 변수가 2개 미만이라 다중회귀를 계산할 수 없습니다."
        }

    design = sm.add_constant(
        analysis[varying_predictors].astype(float),
        has_constant="add",
    )
    model = sm.OLS(analysis["온도(℃)"].astype(float), design).fit(
        cov_type="HC3",
        use_t=True,
    )
    confidence = model.conf_int(alpha=0.05)
    response_std = float(analysis["온도(℃)"].std())

    design_values = design.to_numpy()
    vif_by_predictor = {
        predictor: float(
            variance_inflation_factor(
                design_values,
                design.columns.get_loc(predictor),
            )
        )
        for predictor in varying_predictors
    }

    rows = []
    for predictor in varying_predictors:
        coefficient_1pct = float(model.params[predictor])
        standard_error_1pct = float(model.bse[predictor])
        lower_1pct = float(confidence.loc[predictor, 0])
        upper_1pct = float(confidence.loc[predictor, 1])
        p_value = float(model.pvalues[predictor])
        standardized_beta = (
            coefficient_1pct * float(analysis[predictor].std()) / response_std
            if response_std > 0
            else float("nan")
        )
        if p_value < 0.05 and coefficient_1pct > 0:
            judgement = "다른 피복비율 통제 후 유의한 온도 상승 관련"
        elif p_value < 0.05:
            judgement = "다른 피복비율 통제 후 유의한 온도 하락 관련"
        else:
            judgement = "통계적으로 유의하지 않음"
        rows.append(
            {
                "독립변수": predictor.removesuffix("(%)"),
                "10%p 증가 시 온도 변화(℃)": coefficient_1pct * 10,
                "표준오차": standard_error_1pct * 10,
                "95% 하한(℃)": lower_1pct * 10,
                "95% 상한(℃)": upper_1pct * 10,
                "표준화 계수": standardized_beta,
                "p값": p_value,
                "VIF": vif_by_predictor[predictor],
                "판정": judgement,
            }
        )

    coefficient_table = pd.DataFrame(rows)
    numeric_output_columns = [
        "10%p 증가 시 온도 변화(℃)",
        "표준오차",
        "95% 하한(℃)",
        "95% 상한(℃)",
        "표준화 계수",
        "p값",
        "VIF",
    ]
    coefficient_table[numeric_output_columns] = coefficient_table[
        numeric_output_columns
    ].round(4)

    analysis = analysis.copy()
    analysis["예측온도(℃)"] = model.fittedvalues
    analysis["잔차(℃)"] = model.resid
    return analysis, coefficient_table, {
        "nobs": int(model.nobs),
        "r_squared": float(model.rsquared),
        "adjusted_r_squared": float(model.rsquared_adj),
        "model_p_value": float(model.f_pvalue),
        "predictors": varying_predictors,
        "dropped_predictors": dropped_predictors,
        "max_vif": max(vif_by_predictor.values()),
    }


st.markdown(
    """
    <style>
      .block-container {padding-top:1.4rem;padding-bottom:1rem;max-width:1500px;}
      [data-testid="stSidebar"] {border-right:1px solid rgba(128,128,128,.18);}
      .map-kicker {color:#2c7a5a;font-weight:700;letter-spacing:.08em;font-size:.78rem;margin-bottom:.25rem;}
      .map-subtitle {color:#64706d;margin-top:-.65rem;margin-bottom:1rem;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="map-kicker">LAND COVER × 300M BUFFER × TEMPERATURE</div>', unsafe_allow_html=True)
st.title("S-DoT 300m 토지피복 · 온도 분석")
st.markdown(
    '<p class="map-subtitle">온도센서 주변 토지피복 구성비를 비교하고 구성비에 따른 온도 경향을 살펴봅니다.</p>',
    unsafe_allow_html=True,
)

sdot_snapshot: pd.DataFrame | None = None
buffer_stats: pd.DataFrame | None = None
buffer_geojson: dict | None = None
sensor_table: pd.DataFrame | None = None
selected_buffer_serial: str | None = None
selected_timestamp: pd.Timestamp | None = None
buffer_level = "대분류"
min_buffer_coverage = 50.0
selected_district = "전체"
analysis_cover_name = "시가화건조지역"

with st.sidebar:
    st.header("지도 설정")
    selected_place = st.selectbox("빠른 이동", list(PLACES), index=0)
    selected_basemap = st.selectbox("배경지도", list(BASEMAPS), index=1)

    st.divider()
    st.subheader("토지피복 SHP")
    show_shp = st.toggle("2024 세분류 토지피복", value=False)
    shp_opacity = st.slider("토지피복 투명도", 0.1, 1.0, 0.72, 0.05, disabled=not show_shp)
    shp_image_path = find_asset(SHP_IMAGE_FILE_NAME)
    shp_meta_path = find_asset(SHP_META_FILE_NAME)
    if show_shp and (shp_image_path is None or shp_meta_path is None):
        st.warning("토지피복 오버레이 파일 2개가 저장소에 없습니다.")

    st.divider()
    st.subheader("온도센서 · 300m 버퍼")
    show_temperature_sensors = st.toggle("S-DoT 온도센서", value=True)

    sdot_path = find_asset(DATA_FILE_NAME)
    buffer_path = find_asset(BUFFER_FILE_NAME)
    buffer_stats_path = find_asset(BUFFER_STATS_FILE_NAME)
    missing_analysis_files = [
        name
        for name, path in [
            (DATA_FILE_NAME, sdot_path),
            (BUFFER_FILE_NAME, buffer_path),
            (BUFFER_STATS_FILE_NAME, buffer_stats_path),
        ]
        if path is None
    ]
    if show_temperature_sensors and missing_analysis_files:
        st.error("분석파일이 저장소에 없습니다: " + ", ".join(missing_analysis_files))
        st.stop()

    if show_temperature_sensors:
        sdot_data = load_sdot_data(str(sdot_path))
        buffer_geojson = load_gzip_json(str(buffer_path))
        buffer_stats = load_buffer_stats(str(buffer_stats_path))

        timestamps = sorted(sdot_data["measured_at"].dropna().unique())
        selected_timestamp = st.select_slider(
            "측정시각",
            options=timestamps,
            value=timestamps[-1],
            format_func=lambda value: pd.Timestamp(value).strftime("%m-%d %H:%M"),
        )
        districts = ["전체", *sorted(sdot_data["district"].dropna().astype(str).unique())]
        selected_district = st.selectbox("자치구", districts)
        analysis_cover_name = st.selectbox(
            "관계 지도 피복변수",
            list(ANALYSIS_COVER_COMPONENTS),
            help="녹지는 산림지역과 초지의 합입니다.",
        )
        buffer_level = st.selectbox(
            "센서 상세보기 분류",
            ["대분류", "중분류", "세분류"],
        )
        min_buffer_coverage = st.slider(
            "최소 SHP 피복률(%)",
            0,
            100,
            50,
            5,
            help="버퍼 전체 면적 중 제공된 SHP가 차지하는 비율입니다.",
        )

        sdot_snapshot = sdot_data[sdot_data["measured_at"] == selected_timestamp].copy()
        if selected_district != "전체":
            sdot_snapshot = sdot_snapshot[
                sdot_snapshot["district"].astype(str) == selected_district
            ]
        sdot_snapshot = clean_temperature(
            sdot_snapshot.drop_duplicates("serial", keep="last")
        )
        # 온도가 비어 있거나 유효범위를 벗어난 센서는 지도·버퍼·표·분석에서 제외합니다.
        sdot_snapshot = sdot_snapshot.dropna(subset=[TEMPERATURE_COLUMN]).copy()
        allowed_serials = set(sdot_snapshot["serial"].astype(str))

        feature_lookup = {
            str(feature["properties"]["serial"]): feature["properties"]
            for feature in buffer_geojson["features"]
            if float(feature["properties"].get("coverage_pct", 0)) >= min_buffer_coverage
            and str(feature["properties"]["serial"]) in allowed_serials
        }
        detail_options = ["선택 안 함", *sorted(feature_lookup)]
        selected_option = st.selectbox(
            "센서 상세보기",
            detail_options,
            format_func=lambda serial: (
                serial
                if serial == "선택 안 함"
                else f"{serial} · {feature_lookup[serial].get('address', '')}"
            ),
        )
        if selected_option != "선택 안 함":
            selected_buffer_serial = selected_option

    st.caption(
        "기본 지도는 고온일수록 붉고 선택 피복비율이 높을수록 어두운 "
        "3×3 이변량 지도입니다."
    )

if show_temperature_sensors and sdot_snapshot is not None:
    valid_temperatures = sdot_snapshot[TEMPERATURE_COLUMN].dropna()
    status1, status2, status3 = st.columns(3)
    status1.metric("표시 센서", f"{len(sdot_snapshot):,}개")
    status2.metric(
        "유효 온도 중앙값",
        f"{valid_temperatures.median():.2f} ℃" if not valid_temperatures.empty else "값 없음",
    )
    status3.metric("측정시각", pd.Timestamp(selected_timestamp).strftime("2026-%m-%d %H:%M"))

center, zoom = PLACES[selected_place]
if selected_buffer_serial and buffer_geojson is not None:
    selected_properties = next(
        feature["properties"]
        for feature in buffer_geojson["features"]
        if str(feature["properties"]["serial"]) == selected_buffer_serial
    )
    center = [float(selected_properties["latitude"]), float(selected_properties["longitude"])]
    zoom = 16

st.subheader(f"{analysis_cover_name} 비율과 온도의 공간적 관계")
st.caption(
    f"붉을수록 온도가 높고, 어두울수록 {analysis_cover_name} 비율이 높습니다. "
    "중첩 영역은 전체 버퍼 합집합 안에서 가장 가까운 센서에 한 번만 배정합니다."
)
map_object = build_base_map(center, zoom, selected_basemap)

if show_shp and shp_image_path is not None and shp_meta_path is not None:
    add_shp_overlay(
        map_object,
        shp_image_path,
        load_json(str(shp_meta_path)),
        shp_opacity,
    )

visible_buffer_count = 0
if (
    show_temperature_sensors
    and sdot_snapshot is not None
    and buffer_geojson is not None
    and buffer_stats is not None
):
    allowed_serials = set(sdot_snapshot["serial"].astype(str))
    relationship_group = folium.FeatureGroup(
        name=f"{analysis_cover_name} 비율 × 온도",
        show=True,
        control=True,
    )
    visible_buffer_count = add_bivariate_buffer_layer(
        relationship_group,
        map_object,
        buffer_geojson,
        buffer_stats,
        sdot_snapshot,
        min_buffer_coverage,
        allowed_serials,
        selected_buffer_serial,
        analysis_cover_name,
    )
    relationship_group.add_to(map_object)

    point_group = folium.FeatureGroup(
        name="S-DoT 측정지점",
        show=True,
        control=True,
    )
    add_temperature_points(
        point_group,
        map_object,
        sdot_snapshot,
        show_legend=False,
    )
    point_group.add_to(map_object)

folium.LayerControl(collapsed=False, position="topright").add_to(map_object)

map_key_parts = [
    selected_place,
    selected_basemap,
    str(show_shp),
    str(shp_opacity),
    str(show_temperature_sensors),
    analysis_cover_name,
    buffer_level,
    str(min_buffer_coverage),
    str(selected_buffer_serial),
    str(selected_timestamp),
    selected_district,
]
st_folium(
    map_object,
    width=None,
    height=680,
    returned_objects=[],
    key="map-" + "-".join(map_key_parts),
)

if show_temperature_sensors and buffer_stats is not None and sdot_snapshot is not None:
    sensor_table, composition_columns = build_sensor_table(
        buffer_stats,
        sdot_snapshot,
        min_buffer_coverage,
    )

    st.subheader("온도센서별 300m 대분류 구성비")
    sort_col1, sort_col2 = st.columns([2, 1])
    sort_options = [
        "온도(℃)",
        "SHP 피복률(%)",
        "우세 비율(%)",
        *composition_columns,
    ]
    with sort_col1:
        sensor_sort_column = st.selectbox("센서표 정렬 기준", sort_options)
    with sort_col2:
        sensor_sort_ascending = st.toggle("오름차순", value=False, key="sensor_sort_order")
    sorted_sensor_table = sensor_table.sort_values(
        sensor_sort_column,
        ascending=sensor_sort_ascending,
        na_position="last",
    )
    st.dataframe(
        sorted_sensor_table,
        hide_index=True,
        width="stretch",
        height=430,
        row_height=64,
        column_config={
            "피복 구성비": st.column_config.ImageColumn(
                "피복 구성비",
                width="small",
                help="각 센서 300m 버퍼의 대분류 구성비입니다. 두 번 클릭하면 확대됩니다.",
                pinned=True,
            ),
            "온도(℃)": st.column_config.NumberColumn(format="%.2f ℃"),
            "SHP 피복률(%)": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%.1f%%"),
            "우세 비율(%)": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%.1f%%"),
            **{
                column: st.column_config.NumberColumn(format="%.1f%%")
                for column in composition_columns
            },
        },
    )
    st.download_button(
        "정렬된 센서표 CSV 다운로드",
        sorted_sensor_table.drop(columns=["피복 구성비"])
        .to_csv(index=False)
        .encode("utf-8-sig"),
        file_name="sdot_300m_landcover_temperature.csv",
        mime="text/csv",
    )

    if selected_buffer_serial:
        selected_composition = buffer_stats[
            (buffer_stats["serial"] == selected_buffer_serial)
            & (buffer_stats["level"] == buffer_level)
        ].copy()
        selected_composition = filter_analysis_landcover_rows(
            selected_composition
        ).sort_values("share_pct", ascending=False)
        if not selected_composition.empty:
            selected_coverage = float(selected_composition["coverage_pct"].iloc[0])
            dominant_row = selected_composition.iloc[0]
            st.subheader(f"{selected_buffer_serial} · 300m 상세 구성비")
            metric1, metric2, metric3 = st.columns(3)
            metric1.metric("SHP 피복률", f"{selected_coverage:.1f}%")
            metric2.metric("우세 용도", str(dominant_row["class_name"]))
            metric3.metric("우세 비율", f"{float(dominant_row['share_pct']):.1f}%")
            chart_data = selected_composition[
                ["class_code", "class_name", "share_pct", "area_m2"]
            ].copy()
            chart_data["color"] = chart_data["class_code"].map(class_color)
            composition_chart = (
                alt.Chart(chart_data)
                .mark_bar(cornerRadiusEnd=4)
                .encode(
                    x=alt.X("share_pct:Q", title="구성비 (%)", scale=alt.Scale(domain=[0, 100])),
                    y=alt.Y("class_name:N", title=None, sort="-x"),
                    color=alt.Color("color:N", scale=None, legend=None),
                    tooltip=[
                        alt.Tooltip("class_name:N", title="용도"),
                        alt.Tooltip("share_pct:Q", title="구성비(%)", format=".2f"),
                        alt.Tooltip("area_m2:Q", title="면적(㎡)", format=",.0f"),
                    ],
                )
                .properties(height=max(180, len(chart_data) * 27))
            )
            st.altair_chart(composition_chart, width="stretch")

    st.subheader("토지피복 구성비에 따른 온도 경향")
    trend_table = build_trend_table(sensor_table, composition_columns)
    trend_col1, trend_col2 = st.columns([2, 1])
    trend_sort_options = [
        "온도차(℃)",
        "온도 상관계수",
        "평균 구성비(%)",
        "해당 용도 포함 센서수",
    ]
    with trend_col1:
        trend_sort_column = st.selectbox("경향표 정렬 기준", trend_sort_options)
    with trend_col2:
        trend_sort_ascending = st.toggle("오름차순", value=False, key="trend_sort_order")
    sorted_trend = trend_table.sort_values(
        trend_sort_column,
        ascending=trend_sort_ascending,
        na_position="last",
    )
    st.dataframe(
        sorted_trend,
        hide_index=True,
        width="stretch",
        column_config={
            "평균 구성비(%)": st.column_config.NumberColumn(format="%.2f%%"),
            "온도 상관계수": st.column_config.NumberColumn(format="%.3f"),
            "구성비 상위25% 평균온도(℃)": st.column_config.NumberColumn(format="%.2f ℃"),
            "구성비 하위25% 평균온도(℃)": st.column_config.NumberColumn(format="%.2f ℃"),
            "온도차(℃)": st.column_config.NumberColumn(format="%+.2f ℃"),
        },
    )
    st.caption(
        f"현재 시각의 유효 온도 센서 {sensor_table['온도(℃)'].notna().sum()}개를 사용했습니다. "
        "습지·농업지역·나지는 제외하고 녹지는 산림지역+초지로 계산했습니다. "
        "상관과 온도차는 탐색적 연관성이며 인과관계를 뜻하지 않습니다."
    )
else:
    st.info("S-DoT 온도센서 토글을 켜면 센서별 구성비와 온도 경향표가 표시됩니다.")

with st.expander("데이터 및 분석 기준"):
    st.markdown(
        f"""
        - **SHP:** 7개 원본 파일, 97,018개 세분류 폴리곤, 약 2m 래스터 해상도
        - **버퍼:** S-DoT 위치 중심 반경 300m
        - **구성비:** 버퍼 내 SHP 피복면적 대비 각 용도 교차면적
        - **분석 대분류:** 시가화건조지역, 녹지(산림지역+초지), 수역
        - **분석 제외:** 습지, 농업지역, 나지와 해당 하위 분류
        - **이변량 지도:** 피복비율과 온도를 각각 삼분위로 나눈 3×3 관계 색상
          ([ArcGIS Bivariate colors 설계 참고]({BIVARIATE_REFERENCE}))
        - **온도 유효범위:** {TEMPERATURE_MIN:.0f}℃ 이상 {TEMPERATURE_MAX:.0f}℃ 이하
        - **S-DoT 기간:** 2026-07-20 00:07 ~ 2026-07-26 23:07
        - **토지피복 출처:** [기후에너지환경부 환경공간정보서비스]({LAND_COVER_SOURCE})
        - **S-DoT 출처:** [서울열린데이터광장]({SDOT_SOURCE})

        SHP가 제공되지 않은 지역은 피복률로 구분합니다. S-DoT 값은 연구·탐색용 자료이며
        통신 지연, 장비 장애 또는 현장 여건에 따른 결측·이상값이 포함될 수 있습니다.
        """
    )

st.caption("자료 출처: 사용자 제공 2024 토지피복 SHP · 서울특별시 서울열린데이터광장")

st.divider()
st.subheader("대분류 피복비율 다중선형회귀분석")
st.caption(
    "X: 시가화건조지역 비율, 녹지 비율(산림지역+초지), 수역 비율 · "
    "Y: 현재 선택 시각의 S-DoT 온도"
)
if show_temperature_sensors and sensor_table is not None and not sensor_table.empty:
    regression_data, regression_table, regression_info = (
        build_landcover_multiple_regression(sensor_table)
    )
    if regression_data is None or regression_table is None:
        st.warning(str(regression_info["reason"]))
    else:
        model_p_value = float(regression_info["model_p_value"])
        metric1, metric2, metric3, metric4 = st.columns(4)
        metric1.metric("분석 센서", f"{int(regression_info['nobs']):,}개")
        metric2.metric("설명력 R²", f"{float(regression_info['r_squared']):.3f}")
        metric3.metric(
            "조정 R²",
            f"{float(regression_info['adjusted_r_squared']):.3f}",
        )
        metric4.metric(
            "모형 p값",
            "<0.0001" if model_p_value < 0.0001 else f"{model_p_value:.4f}",
        )

        dropped_predictors = regression_info["dropped_predictors"]
        if dropped_predictors:
            dropped_text = ", ".join(
                predictor.removesuffix("(%)")
                for predictor in dropped_predictors
            )
            st.warning(
                f"선택 조건에서 값의 변화가 없는 변수는 회귀식에서 제외했습니다: {dropped_text}"
            )

        significant = regression_table[regression_table["p값"] < 0.05]
        if significant.empty:
            st.info(
                "다른 피복비율을 함께 통제한 결과, p<0.05 기준으로 유의한 개별 피복변수는 없었습니다."
            )
        else:
            strongest_index = significant["표준화 계수"].abs().idxmax()
            strongest = significant.loc[strongest_index]
            direction = (
                "높아지는"
                if strongest["10%p 증가 시 온도 변화(℃)"] > 0
                else "낮아지는"
            )
            st.success(
                f"유의 변수 중 표준화 효과가 가장 큰 항목은 {strongest['독립변수']}입니다. "
                f"다른 피복비율이 같을 때 이 비율이 10%p 증가하면 온도가 평균 "
                f"{abs(float(strongest['10%p 증가 시 온도 변화(℃)'])):.3f}℃ "
                f"{direction} 관계로 추정됐습니다."
            )

        if float(regression_info["max_vif"]) >= 5:
            st.warning(
                "일부 변수의 VIF가 5 이상입니다. 토지피복 비율은 서로 연관된 구성자료이므로 "
                "개별 회귀계수 해석에 주의하세요."
            )

        interval = (
            alt.Chart(regression_table)
            .mark_rule(strokeWidth=3)
            .encode(
                y=alt.Y("독립변수:N", title=None),
                x=alt.X(
                    "95% 하한(℃):Q",
                    title="피복비율 10%p 증가 시 온도 변화 (℃)",
                ),
                x2="95% 상한(℃):Q",
                color=alt.value("#718096"),
                tooltip=[
                    alt.Tooltip("독립변수:N"),
                    alt.Tooltip(
                        "10%p 증가 시 온도 변화(℃):Q",
                        title="추정 온도변화(℃)",
                        format="+.3f",
                    ),
                    alt.Tooltip("95% 하한(℃):Q", format="+.3f"),
                    alt.Tooltip("95% 상한(℃):Q", format="+.3f"),
                    alt.Tooltip("p값:Q", format=".4f"),
                    alt.Tooltip("VIF:Q", format=".2f"),
                ],
            )
        )
        estimates = (
            alt.Chart(regression_table)
            .mark_point(filled=True, size=110, color="#d95f02")
            .encode(
                y=alt.Y("독립변수:N", title=None),
                x=alt.X("10%p 증가 시 온도 변화(℃):Q"),
            )
        )
        zero_line = (
            alt.Chart(pd.DataFrame({"기준": [0.0]}))
            .mark_rule(color="#2d3748", strokeDash=[5, 4])
            .encode(x="기준:Q")
        )
        coefficient_chart = (zero_line + interval + estimates).properties(
            height=240,
            title="회귀계수와 95% 신뢰구간",
        )

        observed_min = float(
            min(
                regression_data["온도(℃)"].min(),
                regression_data["예측온도(℃)"].min(),
            )
        )
        observed_max = float(
            max(
                regression_data["온도(℃)"].max(),
                regression_data["예측온도(℃)"].max(),
            )
        )
        identity_data = pd.DataFrame(
            {
                "예측온도(℃)": [observed_min, observed_max],
                "관측온도(℃)": [observed_min, observed_max],
            }
        )
        prediction_points = (
            alt.Chart(regression_data)
            .mark_circle(size=65, opacity=0.62, color="#4d8073")
            .encode(
                x=alt.X(
                    "예측온도(℃):Q",
                    title="모형 예측온도 (℃)",
                    scale=alt.Scale(zero=False),
                ),
                y=alt.Y(
                    "온도(℃):Q",
                    title="관측온도 (℃)",
                    scale=alt.Scale(zero=False),
                ),
                tooltip=[
                    alt.Tooltip("센서:N"),
                    alt.Tooltip("자치구:N"),
                    alt.Tooltip("예측온도(℃):Q", format=".2f"),
                    alt.Tooltip("온도(℃):Q", title="관측온도(℃)", format=".2f"),
                    alt.Tooltip("잔차(℃):Q", format="+.2f"),
                ],
            )
        )
        identity_line = (
            alt.Chart(identity_data)
            .mark_line(color="#718096", strokeDash=[5, 4])
            .encode(x="예측온도(℃):Q", y="관측온도(℃):Q")
        )
        prediction_chart = (identity_line + prediction_points).properties(
            height=300,
            title="관측온도와 모형 예측온도",
        )

        chart_col1, chart_col2 = st.columns(2)
        with chart_col1:
            st.altair_chart(coefficient_chart, width="stretch")
        with chart_col2:
            st.altair_chart(prediction_chart, width="stretch")

        st.dataframe(
            regression_table,
            hide_index=True,
            width="stretch",
            column_config={
                "10%p 증가 시 온도 변화(℃)": st.column_config.NumberColumn(
                    format="%+.3f ℃"
                ),
                "표준오차": st.column_config.NumberColumn(format="%.3f"),
                "95% 하한(℃)": st.column_config.NumberColumn(format="%+.3f ℃"),
                "95% 상한(℃)": st.column_config.NumberColumn(format="%+.3f ℃"),
                "표준화 계수": st.column_config.NumberColumn(format="%+.3f"),
                "p값": st.column_config.NumberColumn(format="%.4f"),
                "VIF": st.column_config.NumberColumn(format="%.2f"),
            },
        )
        st.caption(
            "OLS에 HC3 강건 표준오차를 적용했습니다. 습지·농업지역·나지는 모든 분석에서 제외했고, "
            "녹지는 산림지역과 초지의 합으로 계산했습니다. 비율은 제외 후 재정규화하지 않아 "
            "원래 300m 버퍼 내 피복면적 기준을 유지합니다. p<0.05는 조건부 연관성을 뜻하며 "
            "인과효과를 의미하지 않습니다."
        )
else:
    st.info("S-DoT 온도센서 토글을 켜면 대분류 피복비율 다중회귀분석이 표시됩니다.")
