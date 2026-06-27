def test_import_package():
    import portopt

    assert portopt.__version__


def test_import_subpackages():
    from portopt import backtest, data, evaluation, models, strategies, utils

    assert all(m is not None for m in (data, strategies, models, backtest, evaluation, utils))
