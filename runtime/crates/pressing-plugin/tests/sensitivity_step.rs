//! The Sensitivity slider snaps to 5 % increments.

use nih_plug::prelude::Param;
use pressing_plugin::{PressingParams, SENSITIVITY_DEFAULT, SENSITIVITY_STEP};

#[test]
fn sensitivity_snaps_to_five_percent() {
    let p = PressingParams::new(2.75);
    assert_eq!(SENSITIVITY_STEP, 5.0);
    assert_eq!(SENSITIVITY_DEFAULT % SENSITIVITY_STEP, 0.0);
    for i in 0..=1000 {
        let v = p.sensitivity.preview_plain(i as f32 / 1000.0);
        assert_eq!(v % 5.0, 0.0, "normalized {} -> {v}", i as f32 / 1000.0);
        assert!((0.0..=100.0).contains(&v));
    }
    assert_eq!(p.sensitivity.preview_plain(0.523), 50.0);
    assert_eq!(p.sensitivity.preview_plain(0.526), 55.0);
    assert_eq!(p.sensitivity.preview_plain(1.0), 100.0);
    // Typed values snap too, and the display shows whole percentages.
    let n = p.sensitivity.string_to_normalized_value("47.3 %").unwrap();
    assert_eq!(p.sensitivity.preview_plain(n), 45.0);
    assert!(p
        .sensitivity
        .normalized_value_to_string(p.sensitivity.preview_normalized(55.0), true)
        .starts_with("55 %"));
}
