from pydantic import BaseModel
from typing import List, Tuple, Dict, Any, Union, Optional

class Axis(BaseModel):
    label: str
    max_value: float

class KMAxes(BaseModel):
    x: Axis
    y: Axis

class KMArm(BaseModel):
    treatment_label: str
    coordinates: List[Tuple[float, float]] # List of [Time, Survival_Prob] at each step drop
    censoring_ticks: List[float] # List of X-axis time points where censoring occurs

class KMChartSchema(BaseModel):
    chart_type: str = "kaplan_meier"
    axes: KMAxes
    arms: List[KMArm]

class ForestStudy(BaseModel):
    study_label: str
    ratio_value: float
    ci_lower: float
    ci_upper: float

class ForestChartSchema(BaseModel):
    chart_type: str = "forest_plot"
    axes: Dict[str, Any] # e.g., {"x": {"label": "Hazard Ratio (95% CI)"}}
    studies: List[ForestStudy]
    overall_effect: ForestStudy

class WaterfallBar(BaseModel):
    label: str
    value: float

class WaterfallChartSchema(BaseModel):
    chart_type: str = "waterfall_plot"
    axes: Dict[str, Any] # e.g., {"y": {"label": "Change from baseline (%)"}}
    bars: List[WaterfallBar]

class AnchorDataPoint(BaseModel):
    x: Union[str, float]
    y: float

class AnchorSeries(BaseModel):
    series_name: str
    series_type: str # 'bar', 'line', 'scatter'
    data: List[AnchorDataPoint]

class AnchorChartSchema(BaseModel):
    chart_type: str # 'simple_bar', 'stacked_bar', 'multi_line', 'dual_axis_combo', 'scatter'
    axes: Dict # e.g., {"x": {"label": "..."}, "y1": {"label": "..."}, "y2": {"label": "..."}}
    series: List[AnchorSeries]
