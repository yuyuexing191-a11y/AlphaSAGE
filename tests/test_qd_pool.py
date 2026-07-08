from qd_pool import (
    classify_feature_group,
    classify_operator_group,
    classify_window_group,
    qd_bucket_key,
)


def test_feature_group_distinguishes_price_volume_and_single_family():
    assert classify_feature_group("TsMean($close,20)") == "price"
    assert classify_feature_group("TsSkew($volume,50)") == "volume"
    assert classify_feature_group("TsCorr($volume,$open,30)") == "price_volume"


def test_operator_group_prefers_specific_operator_families():
    assert classify_operator_group("TsCorr($volume,$open,30)") == "correlation"
    assert classify_operator_group("Rank(TsMean($close,20))") == "rank"
    assert classify_operator_group("TsStd($close,20)") == "volatility"
    assert classify_operator_group("TsMean($close,20)") == "trend"


def test_window_group_uses_largest_explicit_window():
    assert classify_window_group("TsMean($close,10)") == "short"
    assert classify_window_group("TsCorr(TsMean($close,20),$volume,40)") == "medium"
    assert classify_window_group("TsSkew($volume,50)") == "long"
    assert classify_window_group("Add($close,$open)") == "none"


def test_qd_bucket_key_combines_behavior_dimensions():
    assert qd_bucket_key("TsCorr($volume,$open,30)") == (
        "price_volume",
        "correlation",
        "medium",
    )
