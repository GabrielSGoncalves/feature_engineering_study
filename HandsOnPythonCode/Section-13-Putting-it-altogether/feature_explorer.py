from typing import List, Dict, Tuple, Optional
import pandas as pd
import numpy as np
from scipy import stats
from plotly.subplots import make_subplots
from feature_engine import variable_transformers as vt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import plotly.graph_objects as go
import plotly.express as px

# Define Color pallete
colors = px.colors.qualitative.G10


def _get_numerical_feature_info(dataframe: pd.DataFrame, column: str) -> Dict:
    """Generate statistical metrics for a feature in dataframe."""
    dict_info = {
        'Missing Values': dataframe[column].isnull().mean().round(2),
        'Mean': dataframe[column].mean().round(2),
        'Median': dataframe[column].median().round(2),
        'Std': dataframe[column].std().round(2),
        'Skew': dataframe[column].skew().round(2),
        'Kurtosis': dataframe[column].kurtosis().round(2),
    }
    return dict_info


def _get_table_trace(dict_feature_info: Dict, distribution: str) -> go.Table:
    """Generate table plotly trace from a feature dictionary."""
    trace = go.Table(
        header=dict(values=[distribution], font=dict(color='navy', size=16),),
        cells=dict(
            values=[
                list(dict_feature_info.keys()),
                list(dict_feature_info.values()),
            ],
            font=dict(color='black', size=12),
        ),
    )

    return trace


def _get_qqplot_trace(
    df: pd.DataFrame, column: str, **kwargs
) -> Tuple[go.Scatter, go.Scatter]:
    """Generate QQ-Plots traces."""
    qq = stats.probplot(df[column], dist='lognorm', sparams=(1))
    x = np.array([qq[0][0][0], qq[0][0][-1]])

    trace_markers = go.Scatter(
        x=qq[0][0], y=qq[0][1], mode='markers', **kwargs
    )
    trace_line = go.Scatter(
        x=x, y=qq[1][1] + qq[1][0] * x, mode='lines', **kwargs
    )
    return trace_markers, trace_line


def _get_histogram_trace(
    df: pd.DataFrame, column: str, **kwargs
) -> go.Histogram:
    """Generate histogram traces for feature on dataframe."""
    return go.Histogram(x=df[column], **kwargs)  # nbinsx=nbins, )


def _transform_numerical_feature(
    df: pd.DataFrame, feature: str
) -> pd.DataFrame:
    """Perform numerical transformations on a feature from dataframe."""
    # Validate for numeric
    df_filtered = df[df[feature].notnull()][[feature]]
    df_out = df_filtered.copy()

    # Perform transformations
    transformer_log = vt.LogTransformer(variables=[feature])
    transformer_rec = vt.ReciprocalTransformer(variables=[feature])
    transformer_pow = vt.PowerTransformer(variables=[feature], exp=2)
    transformer_sqrt = vt.PowerTransformer(variables=[feature], exp=0.5)
    transformer_bc = vt.BoxCoxTransformer(variables=[feature])
    transformer_yj = vt.YeoJohnsonTransformer(variables=[feature])

    # Perform transformations
    try:
        df_out[f'{feature}_log'] = transformer_log.fit_transform(df_filtered)
    except ValueError:
        df_out[f'{feature}_log'] = transformer_log.fit_transform(
            df_filtered[df_filtered[feature] > 0]
        )

    try:
        df_out[f'{feature}_reciprocal'] = transformer_rec.fit_transform(
            df_filtered
        )
    except ValueError:
        df_out[f'{feature}_reciprocal'] = transformer_rec.fit_transform(
            df_filtered[df_filtered[feature] > 0]
        )

    df_out[f'{feature}_power'] = transformer_pow.fit_transform(df_filtered)
    df_out[f'{feature}_sqrt'] = transformer_sqrt.fit_transform(df_filtered)
    try:
        df_out[f'{feature}_boxcox'] = transformer_bc.fit_transform(df_filtered)

    except ValueError:
        df_out[f'{feature}_boxcox'] = transformer_bc.fit_transform(
            df_filtered[df_filtered[feature] > 0]
        )

    df_out[f'{feature}_yeojohnson'] = transformer_yj.fit_transform(df_filtered)

    return df_out


def _create_feature_subplots_new(dataframe, feature, plot_size=(1200, 800)):
    #
    width, height = plot_size
    df_tranformed = _transform_numerical_feature(dataframe, feature)

    # Initialize figure with subplots
    fig = make_subplots(
        rows=6,
        cols=4,
        column_widths=[0.4, 0.2, 0.2, 0.2],
        # row_heights=[0.4, 0.6],
        specs=[
            [
                {"type": "table"},
                {"type": "table"},
                {"type": "table"},
                {"type": "table"},
            ],
            [
                {"type": "histogram", "rowspan": 2},
                {"type": "histogram"},
                {"type": "histogram"},
                {"type": "histogram"},
            ],
            [
                None,
                {"type": "scatter"},
                {"type": "scatter"},
                {"type": "scatter"},
            ],
            [
                {"type": "scatter", "rowspan": 2},
                {"type": "table"},
                {"type": "table"},
                {"type": "table"},
            ],
            [
                None,
                {"type": "histogram"},
                {"type": "histogram"},
                {"type": "histogram"},
            ],
            [
                None,
                {"type": "scatter"},
                {"type": "scatter"},
                {"type": "scatter"},
            ],
        ],
    )

    fig.add_trace(
        _get_table_trace(
            _get_numerical_feature_info(df_tranformed, feature), 'Original'
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        _get_table_trace(
            _get_numerical_feature_info(df_tranformed, f'{feature}_log'), 'Log'
        ),
        row=1,
        col=2,
    )
    fig.add_trace(
        _get_table_trace(
            _get_numerical_feature_info(df_tranformed, f'{feature}_power'),
            'Power 2',
        ),
        row=1,
        col=3,
    )
    fig.add_trace(
        _get_table_trace(
            _get_numerical_feature_info(df_tranformed, f'{feature}_sqrt'),
            'SquareRoot',
        ),
        row=1,
        col=4,
    )
    fig.add_trace(
        _get_table_trace(
            _get_numerical_feature_info(
                df_tranformed, f'{feature}_reciprocal'
            ),
            'Reciprocal',
        ),
        row=4,
        col=2,
    )
    fig.add_trace(
        _get_table_trace(
            _get_numerical_feature_info(df_tranformed, f'{feature}_boxcox'),
            'Box-Cox',
        ),
        row=4,
        col=3,
    )
    fig.add_trace(
        _get_table_trace(
            _get_numerical_feature_info(
                df_tranformed, f'{feature}_yeojohnson'
            ),
            'Yeo-Johnson',
        ),
        row=4,
        col=4,
    )

    fig.add_trace(
        _get_histogram_trace(
            df_tranformed, feature, name='Original', marker_color=colors[0]
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        _get_histogram_trace(
            df_tranformed, f'{feature}_log', name='Log', marker_color=colors[1]
        ),
        row=2,
        col=2,
    )
    fig.add_trace(
        _get_histogram_trace(
            df_tranformed,
            f'{feature}_power',
            name='Power',
            marker_color=colors[2],
        ),
        row=2,
        col=3,
    )
    fig.add_trace(
        _get_histogram_trace(
            df_tranformed,
            f'{feature}_sqrt',
            name='SquareRoot',
            marker_color=colors[3],
        ),
        row=2,
        col=4,
    )
    fig.add_trace(
        _get_histogram_trace(
            df_tranformed,
            f'{feature}_reciprocal',
            name='Reciprocal',
            marker_color=colors[4],
        ),
        row=5,
        col=2,
    )

    fig.add_trace(
        _get_histogram_trace(
            df_tranformed,
            f'{feature}_boxcox',
            name='Box-Cox',
            marker_color=colors[5],
        ),
        row=5,
        col=3,
    )
    fig.add_trace(
        _get_histogram_trace(
            df_tranformed,
            f'{feature}_yeojohnson',
            name='Yeo-Johnson',
            marker_color=colors[6],
        ),
        row=5,
        col=4,
    )

    fig.add_trace(
        _get_qqplot_trace(
            df_tranformed,
            feature,
            name='Q-Q Plot Original',
            marker_color=colors[0],
        )[0],
        row=4,
        col=1,
    )
    fig.add_trace(
        _get_qqplot_trace(df_tranformed, feature, marker_color='black',)[1],
        row=4,
        col=1,
    )
    fig.add_trace(
        _get_qqplot_trace(
            df_tranformed,
            f'{feature}_log',
            name='Q-Q Plot Log',
            marker_color=colors[1],
        )[0],
        row=3,
        col=2,
    )
    fig.add_trace(
        _get_qqplot_trace(
            df_tranformed, f'{feature}_log', marker_color='black',
        )[1],
        row=3,
        col=2,
    )

    fig.add_trace(
        _get_qqplot_trace(
            df_tranformed,
            f'{feature}_power',
            name='Q-Q Plot Power 2',
            marker_color=colors[2],
        )[0],
        row=3,
        col=3,
    )
    fig.add_trace(
        _get_qqplot_trace(
            df_tranformed, f'{feature}_power', marker_color='black',
        )[1],
        row=3,
        col=3,
    )
    fig.add_trace(
        _get_qqplot_trace(
            df_tranformed,
            f'{feature}_sqrt',
            name='Q-Q Plot SquareRoot',
            marker_color=colors[3],
        )[0],
        row=3,
        col=4,
    )
    fig.add_trace(
        _get_qqplot_trace(
            df_tranformed, f'{feature}_sqrt', marker_color='black',
        )[1],
        row=3,
        col=4,
    )
    fig.add_trace(
        _get_qqplot_trace(
            df_tranformed,
            f'{feature}_reciprocal',
            name='Q-Q Plot Reciprocal',
            marker_color=colors[4],
        )[0],
        row=6,
        col=2,
    )
    fig.add_trace(
        _get_qqplot_trace(
            df_tranformed, f'{feature}_reciprocal', marker_color='black',
        )[1],
        row=6,
        col=2,
    )
    fig.add_trace(
        _get_qqplot_trace(
            df_tranformed,
            f'{feature}_boxcox',
            name='Q-Q Plot Box-Cox',
            marker_color=colors[5],
        )[0],
        row=6,
        col=3,
    )
    fig.add_trace(
        _get_qqplot_trace(
            df_tranformed, f'{feature}_boxcox', marker_color='black',
        )[1],
        row=6,
        col=3,
    )
    fig.add_trace(
        _get_qqplot_trace(
            df_tranformed,
            f'{feature}_yeojohnson',
            name='Q-Q Plot Yeo-Johnson',
            marker_color=colors[6],
        )[0],
        row=6,
        col=4,
    )
    fig.add_trace(
        _get_qqplot_trace(
            df_tranformed, f'{feature}_yeojohnson', marker_color='black',
        )[1],
        row=6,
        col=4,
    )

    fig.update_layout(
        title_text=f"Normality Analysis for feature: {feature}",
        title_font_family="Arial",
        title_font_size=28,
        width=width,
        height=height,
        showlegend=False,
    )

    return fig


def _get_transformation(col_name: str) -> str:
    transf_type = col_name.split('_')
    try:
        return transf_type[1]
    except:
        return 'original'


class DatasetTransformer:
    def __init__(self, X, target, target_type, dict_dtypes):

        self.X = X
        self.dict_dtypes = dict_dtypes
        self.dict_transformed_feats = {}
        self.X_train = None
        self.X_test = None
        self.X_train_tranformed = None
        self.X_test_tranformed = None
        self.transformation_pipeline = None

    def plot_numerical_feature_info(
        self,
        feature: str,
        scaler: Optional[str] = None,
        plot_size: Tuple = (1200, 800),
    ):
        """Plot general info and transformations for an specific feature.

        Parameters
        ----------

        featu
        """
        df_transformed = _transform_numerical_feature(self.X, feature)

        if scaler:
            df_transformed = pd.DataFrame(
                scaler.fit_transform(df_transformed),
                columns=df_transformed.columns,
            )

        # Initialize figure with subplots
        fig = _create_feature_subplots_new(df_transformed, feature, plot_size)
        fig.show()

    def plot_categorical_features(self):
        pass

    def transformed_and_select_features(self):
        pass
