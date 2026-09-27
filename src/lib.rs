//! Pure JSON feature projection and numeric inference response decoding.
//! Model selection, transport, scope, confidence and domain policy belong to callers.
use serde_json::{Map, Value};

/// Decode a row-aligned predictions envelope without guessing an output field.
pub fn decode_numeric_predictions(
    response: &Value,
    expected_rows: usize,
    output_field: Option<&str>,
) -> anyhow::Result<Vec<f64>> {
    let predictions = response
        .get("predictions")
        .and_then(Value::as_array)
        .ok_or_else(|| anyhow::anyhow!("response has no predictions array"))?;
    if predictions.len() != expected_rows {
        anyhow::bail!(
            "returned {} predictions for {expected_rows} input rows",
            predictions.len()
        );
    }
    predictions
        .iter()
        .enumerate()
        .map(|(index, prediction)| {
            prediction_value(prediction, output_field).map_err(|error| {
                anyhow::anyhow!("prediction row {} is invalid: {error}", index + 1)
            })
        })
        .collect()
}

/// Project explicit source fields into model input fields, preserving raw values.
pub fn map_feature_rows(
    rows: Vec<Value>,
    mapping: &Map<String, Value>,
) -> anyhow::Result<(Vec<Value>, Vec<String>)> {
    if mapping.is_empty() {
        anyhow::bail!("feature_mapping must be non-empty");
    }
    let mut feature_names = Vec::with_capacity(mapping.len());
    let mut source_names = Vec::with_capacity(mapping.len());
    for (model_feature, source_feature) in mapping {
        if model_feature.trim().is_empty() || model_feature.trim() != model_feature {
            anyhow::bail!("feature_mapping model fields must be non-empty and unpadded");
        }
        let source_feature = source_feature
            .as_str()
            .filter(|value| !value.trim().is_empty() && value.trim() == *value)
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "feature_mapping value for '{model_feature}' must be a non-empty, unpadded string"
                )
            })?;
        feature_names.push(model_feature.clone());
        source_names.push(source_feature.to_owned());
    }
    let mapped = rows
        .into_iter()
        .enumerate()
        .map(|(index, row)| {
            let row = row
                .as_object()
                .ok_or_else(|| anyhow::anyhow!("input row {} must be an object", index + 1))?;
            let mut mapped = Map::new();
            for (model_feature, source_feature) in feature_names.iter().zip(&source_names) {
                let value = row.get(source_feature).cloned().ok_or_else(|| {
                    anyhow::anyhow!(
                        "feature_mapping source '{source_feature}' is absent from input row {}",
                        index + 1
                    )
                })?;
                mapped.insert(model_feature.clone(), value);
            }
            Ok(Value::Object(mapped))
        })
        .collect::<anyhow::Result<Vec<_>>>()?;
    Ok((mapped, feature_names))
}

pub fn prediction_value(prediction: &Value, output_field: Option<&str>) -> anyhow::Result<f64> {
    if let Some(value) = prediction.as_f64() {
        return finite_prediction(value);
    }
    let object = prediction
        .as_object()
        .ok_or_else(|| anyhow::anyhow!("prediction must be a number or object"))?;
    if let Some(field) = output_field {
        let value = object
            .get(field)
            .and_then(Value::as_f64)
            .ok_or_else(|| anyhow::anyhow!("output_field '{field}' is missing or non-numeric"))?;
        return finite_prediction(value);
    }
    let numeric = object
        .iter()
        .filter_map(|(field, value)| value.as_f64().map(|value| (field, value)))
        .collect::<Vec<_>>();
    match numeric.as_slice() {
        [(_, value)] => finite_prediction(*value),
        [] => anyhow::bail!("prediction object has no numeric output field"),
        _ => anyhow::bail!(
            "prediction object has multiple numeric fields ({}); configure output_field",
            numeric
                .iter()
                .map(|(field, _)| field.as_str())
                .collect::<Vec<_>>()
                .join(", ")
        ),
    }
}

fn finite_prediction(value: f64) -> anyhow::Result<f64> {
    if value.is_finite() {
        Ok(value)
    } else {
        anyhow::bail!("prediction must be finite")
    }
}

pub fn validate_value(value: f64, label: &str) -> anyhow::Result<()> {
    if value.is_finite() {
        Ok(())
    } else {
        anyhow::bail!("{label} must be finite")
    }
}
