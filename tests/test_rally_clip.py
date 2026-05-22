from volley_analytics.rally_clip import Rally, segment_rallies


def test_segments_two_rallies_when_separated_by_long_gap():
    # Rally A: frames 0..100, big gap, Rally B: frames 300..420
    frames = list(range(0, 101)) + list(range(300, 421))
    rs = segment_rallies(frames, gap_frames=45, min_rally_frames=50)
    assert len(rs) == 2
    assert (rs[0].start_frame, rs[0].end_frame) == (0, 100)
    assert (rs[1].start_frame, rs[1].end_frame) == (300, 420)


def test_drops_rallies_below_min_length():
    frames = list(range(0, 10)) + list(range(200, 300))
    rs = segment_rallies(frames, gap_frames=45, min_rally_frames=50)
    assert len(rs) == 1
    assert rs[0].start_frame == 200
    assert rs[0].end_frame == 299


def test_small_gaps_do_not_split_rally():
    # Frames have 20-frame gaps (< default gap_frames of 45) — one rally.
    frames = list(range(0, 200, 20))
    rs = segment_rallies(frames, gap_frames=45, min_rally_frames=50)
    assert len(rs) == 1
    assert rs[0].start_frame == 0
    assert rs[0].end_frame == 180


def test_empty_input():
    assert segment_rallies([]) == []


def test_to_seconds():
    r = Rally(start_frame=60, end_frame=180)
    s, e = r.to_seconds(fps=30.0)
    assert s == 2.0
    assert e == 6.0
