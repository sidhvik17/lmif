"""UI sanitizer tests. Import guarded since streamlit import has side effects."""


def _import_safe_filename():
    import importlib
    mod = importlib.import_module("ui.app")
    return mod._safe_filename


def test_strips_path_traversal():
    fn = _import_safe_filename()
    assert "/" not in fn("../../etc/passwd")
    assert "\\" not in fn("..\\..\\windows\\system32\\cmd.exe")


def test_keeps_extension_and_hyphen():
    fn = _import_safe_filename()
    out = fn("my-file_v2.pdf")
    assert out.endswith(".pdf")
    assert "-" in out
    assert "_" in out


def test_falls_back_on_empty():
    fn = _import_safe_filename()
    assert fn("") == "upload.bin"
    assert fn("///") == "upload.bin"
    assert fn("...") == "upload.bin"


def test_replaces_spaces_and_specials():
    fn = _import_safe_filename()
    out = fn("my doc (v2).pdf")
    assert " " not in out
    assert "(" not in out
    assert ")" not in out


def test_caps_length():
    fn = _import_safe_filename()
    out = fn("a" * 500 + ".pdf")
    assert len(out) <= 200
