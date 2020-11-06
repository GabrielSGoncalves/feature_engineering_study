import pandas as pd
import numpy as np
from scipy import stats
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def _get_feature_info(dataframe, column):
    dict_info = {
        'col1': {
            #'Cardinality': len(dataframe[column].unique()),
            'Missing Values': dataframe[column].isnull().sum().round(2),
            'Mean': dataframe[column].mean().round(2),
            'Median': dataframe[column].median().round(2),
        },
        'col2': {
            'Std': dataframe[column].std().round(2),
            'Skew': dataframe[column].skew().round(2),
            'Kurtosis': dataframe[column].kurtosis().round(2),
        },
    }
    return dict_info
    # return {**{'Feature': column}, **dict_info}


def _get_table_trace(dict_feature_info, distribution):
    trace = go.Table(
        header=dict(values=[distribution], font=dict(color='blue', size=12),),
        cells=dict(
            values=[
                list(dict_feature_info.get('col1').keys()),
                list(dict_feature_info.get('col1').values()),
                list(dict_feature_info.get('col2').keys()),
                list(dict_feature_info.get('col2').values()),
            ],
            font=dict(color='black', size=8),
        ),
    )

    return trace


def _plot_table(dict_feature_info):
    fig = go.Figure(
        data=[
            go.Table(
                header=dict(
                    values=['feature', dict_feature_info.get('feature')]
                ),
                cells=dict(
                    values=[
                        list(dict_feature_info.keys())[1:],
                        list(dict_feature_info.values())[1:],
                    ]
                ),
            )
        ]
    )
    return fig


def _get_qqplot_trace(dataframe, column, **kwargs):
    qq = stats.probplot(dataframe[column], dist='lognorm', sparams=(1))
    x = np.array([qq[0][0][0], qq[0][0][-1]])

    trace_markers = go.Scatter(
        x=qq[0][0], y=qq[0][1], mode='markers', **kwargs
    )
    trace_line = go.Scatter(x=x, y=qq[1][1] + qq[1][0] * x, mode='lines')
    return trace_markers, trace_line


# def _get_histogram_trace(dataframe, column, nbins=40, **kwargs):
def _get_histogram_trace(dataframe, column, **kwargs):
    """
    """
    return go.Histogram(x=dataframe[column], **kwargs)  # nbinsx=nbins, )


from feature_engine import variable_transformers as vt


def _transform_feature(dataframe, feature):
    """
    """
    # Validate for numeric
    df_filtered = dataframe[dataframe[feature].notnull()][[feature]]
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


def _create_feature_subplots_original(
    dataframe, feature, plot_size=(1200, 800)
):
    #
    width, height = plot_size
    df_tranformed = _transform_feature(dataframe, feature)

    # Initialize figure with subplots
    fig = make_subplots(
        rows=2,
        cols=8,
        # column_widths=[0.6, 0.4],
        # row_heights=[0.4, 0.6],
        subplot_titles=(
            "Original",
            "Log",
            "Reciprocal",
            "Power",
            'SquareRoot',
            "Box-Cox",
            "Yeo-Johnson",
            "Q-Q Plots",
        ),
        specs=[
            [
                {"type": "table"},
                {"type": "table"},
                {"type": "table"},
                {"type": "table"},
                {"type": "table"},
                {"type": "table"},
                {"type": "table"},
                {"type": "scatter", "rowspan": 2},
            ],
            [
                {"type": "histogram"},
                {"type": "histogram"},
                {"type": "histogram"},
                {"type": "histogram"},
                {"type": "histogram"},
                {"type": "histogram"},
                {"type": "histogram"},
                None,
            ],
        ],
    )

    fig.add_trace(
        _get_table_trace(
            _get_feature_info(df_tranformed, feature), 'Original'
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        _get_table_trace(
            _get_feature_info(df_tranformed, f'{feature}_log'), 'Log'
        ),
        row=1,
        col=2,
    )
    fig.add_trace(
        _get_table_trace(
            _get_feature_info(df_tranformed, f'{feature}_power'), 'Power'
        ),
        row=1,
        col=3,
    )
    fig.add_trace(
        _get_table_trace(
            _get_feature_info(df_tranformed, f'{feature}_sqrt'), 'SquareRoot'
        ),
        row=1,
        col=4,
    )
    fig.add_trace(
        _get_table_trace(
            _get_feature_info(df_tranformed, f'{feature}_reciprocal'),
            'Reciprocal',
        ),
        row=1,
        col=5,
    )
    fig.add_trace(
        _get_table_trace(
            _get_feature_info(df_tranformed, f'{feature}_boxcox'), 'Box-Cox'
        ),
        row=1,
        col=6,
    )
    fig.add_trace(
        _get_table_trace(
            _get_feature_info(df_tranformed, f'{feature}_yeojohnson'),
            'Yeo-Johnson',
        ),
        row=1,
        col=7,
    )

    fig.add_trace(
        _get_histogram_trace(df_tranformed, feature, name='Original'),
        row=2,
        col=1,
    )
    fig.add_trace(
        _get_histogram_trace(df_tranformed, f'{feature}_log', name='Log'),
        row=2,
        col=2,
    )
    fig.add_trace(
        _get_histogram_trace(df_tranformed, f'{feature}_power', name='Power'),
        row=2,
        col=5,
    )
    fig.add_trace(
        _get_histogram_trace(
            df_tranformed, f'{feature}_sqrt', name='SquareRoot'
        ),
        row=2,
        col=3,
    )
    fig.add_trace(
        _get_histogram_trace(
            df_tranformed, f'{feature}_reciprocal', name='Reciprocal'
        ),
        row=2,
        col=4,
    )

    fig.add_trace(
        _get_histogram_trace(
            df_tranformed, f'{feature}_boxcox', name='Box-Cox'
        ),
        row=2,
        col=6,
    )
    fig.add_trace(
        _get_histogram_trace(
            df_tranformed, f'{feature}_yeojohnson', name='Yeo-Johnson'
        ),
        row=2,
        col=7,
    )

    fig.add_trace(
        _get_qqplot_trace(dataframe, feature, name='Q-Q Plot')[0], row=1, col=8
    )

    fig.add_trace(_get_qqplot_trace(dataframe, feature)[1], row=1, col=8)

    fig.update_layout(
        # title_text="Correlation Matrix",
        # title_font_family="Arial",
        # title_font_size=28,
        # title_x=0.5,
        width=width,
        height=height,
        # xaxis_showgrid=True,
        # yaxis_showgrid=True,
        # yaxis_autorange="reversed",
        legend=dict(
            orientation="h", yanchor="top"
        ),  # , y=1.02, xanchor="right", x=1
    )

    return fig


def _create_feature_subplots(dataframe, feature, plot_size=(1200, 800)):
    #
    width, height = plot_size
    df_tranformed = _transform_feature(dataframe, feature)

    # Initialize figure with subplots
    fig = make_subplots(
        rows=7,
        cols=3,
        column_widths=[0.2, 0.2, 0.6],
        # row_heights=[0.4, 0.6],
        specs=[
            [
                {"type": "table"},
                {"type": "histogram"},
                {"type": "scatter", "rowspan": 7},
            ],
            [{"type": "table"}, {"type": "histogram"}, None],
            [{"type": "table"}, {"type": "histogram"}, None],
            [{"type": "table"}, {"type": "histogram"}, None],
            [{"type": "table"}, {"type": "histogram"}, None],
            [{"type": "table"}, {"type": "histogram"}, None],
            [{"type": "table"}, {"type": "histogram"}, None],
        ],
    )

    fig.add_trace(
        _get_table_trace(
            _get_feature_info(df_tranformed, feature), 'Original'
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        _get_table_trace(
            _get_feature_info(df_tranformed, f'{feature}_log'), 'Log'
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        _get_table_trace(
            _get_feature_info(df_tranformed, f'{feature}_power'), 'Power 2'
        ),
        row=3,
        col=1,
    )
    fig.add_trace(
        _get_table_trace(
            _get_feature_info(df_tranformed, f'{feature}_sqrt'), 'SquareRoot'
        ),
        row=4,
        col=1,
    )
    fig.add_trace(
        _get_table_trace(
            _get_feature_info(df_tranformed, f'{feature}_reciprocal'),
            'Reciprocal',
        ),
        row=5,
        col=1,
    )
    fig.add_trace(
        _get_table_trace(
            _get_feature_info(df_tranformed, f'{feature}_boxcox'), 'Box-Cox'
        ),
        row=6,
        col=1,
    )
    fig.add_trace(
        _get_table_trace(
            _get_feature_info(df_tranformed, f'{feature}_yeojohnson'),
            'Yeo-Johnson',
        ),
        row=7,
        col=1,
    )

    fig.add_trace(
        _get_histogram_trace(df_tranformed, feature, name='Original'),
        row=1,
        col=2,
    )
    fig.add_trace(
        _get_histogram_trace(df_tranformed, f'{feature}_log', name='Log'),
        row=2,
        col=2,
    )
    fig.add_trace(
        _get_histogram_trace(df_tranformed, f'{feature}_power', name='Power'),
        row=3,
        col=2,
    )
    fig.add_trace(
        _get_histogram_trace(
            df_tranformed, f'{feature}_sqrt', name='SquareRoot'
        ),
        row=4,
        col=2,
    )
    fig.add_trace(
        _get_histogram_trace(
            df_tranformed, f'{feature}_reciprocal', name='Reciprocal'
        ),
        row=5,
        col=2,
    )

    fig.add_trace(
        _get_histogram_trace(
            df_tranformed, f'{feature}_boxcox', name='Box-Cox'
        ),
        row=6,
        col=2,
    )
    fig.add_trace(
        _get_histogram_trace(
            df_tranformed, f'{feature}_yeojohnson', name='Yeo-Johnson'
        ),
        row=7,
        col=2,
    )

    fig.add_trace(
        _get_qqplot_trace(df_tranformed, feature, name='Q-Q Plot Original')[0],
        row=1,
        col=3,
    )
    fig.add_trace(
        _get_qqplot_trace(
            df_tranformed, f'{feature}_log', name='Q-Q Plot Log'
        )[0],
        row=1,
        col=3,
    )

    fig.add_trace(_get_qqplot_trace(df_tranformed, feature)[1], row=1, col=3)
    fig.add_trace(
        _get_qqplot_trace(df_tranformed, f'{feature}_log')[1], row=1, col=3
    )

    fig.update_layout(
        # title_text="Correlation Matrix",
        # title_font_family="Arial",
        # title_font_size=28,
        # title_x=0.5,
        width=width,
        height=height,
        # xaxis_showgrid=True,
        # yaxis_showgrid=True,
        # yaxis_autorange="reversed",
        legend=dict(
            orientation="h", yanchor="top"
        ),  # , y=1.02, xanchor="right", x=1
    )

    return fig


class FeatureExplorer:
    def __init__(self, dataframe, dict_dtypes):

        self.dataframe = dataframe
        self.dict_dtypes = dict_dtypes
        self.dict_transformed_feats = {}

    def plot_feature_info(self, feature, plot_size):
        df_tranformed = _transform_feature(self.dataframe, feature)

        # Initialize figure with subplots
        fig = _create_feature_subplots(self.dataframe, feature, plot_size)
        fig.show()
