from translate_srt import SRTEntry, parse_srt, write_srt, translate_entries


def test_parse_and_write_roundtrip():
    source = """1
00:00:01,000 --> 00:00:03,000
Hello there.

2
00:00:03,500 --> 00:00:05,000
How are you?
"""
    entries = parse_srt(source)
    assert len(entries) == 2
    out = write_srt(entries)
    assert "Hello there." in out
    assert "How are you?" in out


def test_translate_entries_mock():
    entries = [
        SRTEntry(1, "00:00:01,000 --> 00:00:02,000", ["Hello"]),
        SRTEntry(2, "00:00:03,000 --> 00:00:04,000", ["How are you?"]),
    ]

    def fake_translate(_system: str, _user: str) -> str:
        return '[{"id":1,"text":"سلام"},{"id":2,"text":"حالت چطوره؟"}]'

    out = translate_entries(entries, fake_translate, batch_size=10, retries=1, backoff_s=0)
    assert out[0].text_lines == ["سلام"]
    assert out[1].text_lines == ["حالت چطوره؟"]
