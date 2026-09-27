# Inference values

`univers-aip-lib-inference-values@0.1.0` extracts reusable JSON mechanics from the
former shared Signal ML adapter. It depends only on `serde_json` and `anyhow`.

- `prediction_value` accepts a finite JSON number, an explicitly selected numeric
  object field, or the only numeric field of an object. Ambiguous objects fail.
  A scalar number retains the legacy behavior of ignoring `output_field`.
- `decode_numeric_predictions` requires a `predictions` array with exactly the
  expected row count. Each output is decoded without coercing strings, booleans
  or null. An empty array is valid only for an expected count of zero; owners
  decide whether an inference request may contain zero rows.
- `map_feature_rows` takes an explicit nonempty object mapping model feature names
  to source field names. Names must be nonempty and unpadded. Every input must be
  an object with each source field present. Nulls and other raw values remain
  unchanged. The caller chooses whether absent mapping means identity projection.
- `validate_value` rejects nonfinite raw floating-point values.

The library does not select models, invoke Ports, fetch metrics, establish World
scope, assign quality/confidence or decide semantic acceptance. A model-level
metric is not transformed into per-prediction confidence. Duplicate source-field
mappings remain allowed, matching the original adapter. Feature order follows
the supplied JSON map iterator; callers must use the returned names with the
returned rows and not assume a separate order.

Run `bash scripts/check.sh`, `bash scripts/build.sh`, and
`bash scripts/publish.sh [--dry-run]` in this independent checkout.
