use serde_json::json;
use univers_aip_lib_inference_values::{
    decode_numeric_predictions, map_feature_rows, prediction_value, validate_value,
};

#[test]
fn ambiguous_prediction_requires_explicit_numeric_field() {
    let value = json!({"y": 4.2, "confidence": 0.9});
    assert!(prediction_value(&value, None)
        .unwrap_err()
        .to_string()
        .contains("configure output_field"));
    assert_eq!(prediction_value(&value, Some("y")).unwrap(), 4.2);
    assert!(prediction_value(&value, Some("missing")).is_err());
}

#[test]
fn scalar_and_single_numeric_output_preserve_original_behavior() {
    assert_eq!(prediction_value(&json!(2), Some("unused")).unwrap(), 2.0);
    assert_eq!(
        prediction_value(&json!({"y": 4, "label": "ok"}), None).unwrap(),
        4.0
    );
    for value in [
        json!(null),
        json!(true),
        json!("3"),
        json!([]),
        json!({"y": "3"}),
    ] {
        assert!(prediction_value(&value, None).is_err());
    }
}

#[test]
fn prediction_rows_must_match_input_and_never_skip_invalid_rows() {
    assert_eq!(
        decode_numeric_predictions(&json!({"predictions": [3, 2]}), 2, None).unwrap(),
        vec![3.0, 2.0]
    );
    assert!(decode_numeric_predictions(&json!({"predictions": [3]}), 2, None).is_err());
    assert!(
        decode_numeric_predictions(&json!({"predictions": [3, null]}), 2, None)
            .unwrap_err()
            .to_string()
            .contains("row 2")
    );
    assert!(decode_numeric_predictions(&json!({}), 0, None).is_err());
    assert!(
        decode_numeric_predictions(&json!({"predictions": []}), 0, None)
            .unwrap()
            .is_empty()
    );
}

#[test]
fn mapped_features_preserve_nulls_and_raw_values_without_filling_absence() {
    let mapping = json!({"model_a": "temperature", "model_b": "quality"});
    let (rows, names) = map_feature_rows(
        vec![json!({"temperature": 12.0, "quality": null, "unused": 100})],
        mapping.as_object().unwrap(),
    )
    .unwrap();
    assert_eq!(rows, vec![json!({"model_a": 12.0, "model_b": null})]);
    assert_eq!(names, vec!["model_a", "model_b"]);
    assert!(map_feature_rows(
        vec![json!({"temperature": 12.0})],
        mapping.as_object().unwrap()
    )
    .is_err());
    assert!(map_feature_rows(vec![json!(null)], mapping.as_object().unwrap()).is_err());
}

#[test]
fn invalid_mapping_names_are_rejected_even_with_no_rows() {
    for mapping in [
        json!({}),
        json!({" x": "source"}),
        json!({"x": "source "}),
        json!({"x": ""}),
        json!({"x": 1}),
    ] {
        assert!(map_feature_rows(vec![], mapping.as_object().unwrap()).is_err());
    }
}

#[test]
fn duplicate_source_projection_is_explicitly_allowed() {
    let mapping = json!({"a": "source", "b": "source"});
    let (rows, _) = map_feature_rows(
        vec![json!({"source": [1, null]})],
        mapping.as_object().unwrap(),
    )
    .unwrap();
    assert_eq!(rows, vec![json!({"a": [1, null], "b": [1, null]})]);
}

#[test]
fn raw_nonfinite_values_are_rejected() {
    for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        assert!(validate_value(value, "prediction").is_err());
    }
    assert!(validate_value(-1.0, "prediction").is_ok());
}
