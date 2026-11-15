import numpy as np

from pressing_tools.evaluate import EDGE, detection_counts


def test_detection_counts_events_false_alarms_and_dont_care():
    n = 2 * EDGE + 10_000
    lab = np.zeros(n, np.uint8)
    lab[EDGE + 1000:EDGE + 1020] = 1      # repaired click: found
    lab[EDGE + 3000:EDGE + 3020] = 1      # repaired click: missed
    lab[EDGE + 5000:EDGE + 5030] = 2      # detected-only: found (neutral, not a false alarm)
    lab[10:30] = 1                        # inside the ignored edge: not counted
    m = np.zeros(n, bool)
    m[EDGE + 1005:EDGE + 1015] = True
    m[EDGE + 5000:EDGE + 5010] = True
    m[EDGE + 8000:EDGE + 8010] = True     # false alarm
    c = detection_counts(lab, m)
    assert (c.l1_events, c.l1_found) == (2, 1)
    assert (c.l2_events, c.l2_found) == (1, 1)
    assert (c.pred_regions, c.pred_on_l1, c.pred_on_l2_only, c.pred_false) == (3, 1, 1, 1)
    assert c.clean_samples_touched == 10
    s = c.summary()
    assert s["recall_repaired"] == 0.5 and s["precision_vs_any_label"] == 2 / 3
