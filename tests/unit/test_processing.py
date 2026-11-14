"""Text splitting (B5) and chunk metadata (Phase 1)."""
from langchain_core.documents import Document

from controllers.ProcessController import ProcessController
from controllers.BaseController import BaseController
from routes.schemes.data import ProcessRequest

C_PAGE = """Lecture 5: Loops in C
A for loop repeats a block a known number of times.

#include <stdio.h>

int main() {
    int i, sum = 0;
    for (i = 1; i <= 10; i++) {
        sum = sum + i;
        printf("i = %d, sum = %d\\n", i, sum);
    }
    if (sum > 50) {
        printf("big\\n");
    } else {
        printf("small\\n");
    }
    return 0;
}

Common mistake: writing for (i = 1; i <= 10; i++); with a semicolon makes the loop body empty.
""" * 4

ASSET_CONFIG = {"original_name": "Lecture 5 - Loops.pdf", "doc_type": "lecture",
                "course": "CS101", "topic": "loops"}


def make_pc():
    return ProcessController.__new__(ProcessController)  # no directory creation


def test_b5_defaults_and_code_aware_split():
    req = ProcessRequest()
    assert (req.chunk_size, req.overlap_size) == (1000, 200)

    docs = [Document(page_content=C_PAGE, metadata={"page": 3, "source": "/abs/x.pdf"}),
            Document(page_content="Lab 2 exercise", metadata={"page": 4, "source": "/abs/x.pdf"})]
    chunks = make_pc().process_file_content(docs, "abcdefgh1234_x.pdf", req.chunk_size, req.overlap_size)

    original_lines = {l.strip() for l in C_PAGE.splitlines()}
    assert len(chunks) > 2
    for c in chunks:
        assert len(c.page_content) <= 1000
        if c.metadata["page"] == 3:
            for line in c.page_content.splitlines():  # whole source lines only, never cut mid-statement
                assert line.strip() in original_lines, repr(line)
    # chunks never mix pages -> exact page attribution
    assert {c.metadata["page"] for c in chunks} == {3, 4}
    assert "Lab 2 exercise" not in "".join(c.page_content for c in chunks if c.metadata["page"] == 3)


def test_old_tiny_chunking_would_cut_code():
    """Documents why B5 mattered: the old 100/20 defaults split code mid-statement."""
    chunks = make_pc().process_file_content([Document(page_content=C_PAGE, metadata={})], "f.pdf", 100, 20)
    assert any(not c.page_content.rstrip().endswith((";", "{", "}", ".")) for c in chunks)


def test_chunk_metadata_from_asset_config():
    md = make_pc().get_chunk_metadata(
        loader_metadata={"page": 0, "total_pages": 12, "source": "C:/srv/assets/files/1/abc.pdf",
                         "file_path": "C:/srv/assets/files/1/abc.pdf", "author": "Dr. X"},
        file_id="k3j4h5g6f7d8_Lecture5Loops.pdf",
        asset_config=ASSET_CONFIG,
    )
    assert md["document_name"] == "Lecture 5 - Loops.pdf"
    assert md["doc_type"] == "lecture"
    assert md["course"] == "CS101" and md["topic"] == "loops"
    assert md["page"] == 0 and md["page_number"] == 1          # 0-based kept, 1-based added
    assert md["total_pages"] == 12 and md["author"] == "Dr. X"   # loader metadata kept
    assert md["file_id"] == "k3j4h5g6f7d8_Lecture5Loops.pdf"
    assert "source" not in md and "file_path" not in md          # no absolute server paths


def test_chunk_metadata_backward_compatible_without_asset_config():
    md = make_pc().get_chunk_metadata({"page": 4}, "k3j4h5g6f7d8_lab2.pdf", asset_config=None)
    assert md["document_name"] == "lab2.pdf"
    assert md["doc_type"] == "other"
    assert md["page_number"] == 5


def test_text_files_have_no_page_number():
    md = make_pc().get_chunk_metadata({"source": "/x/notes.txt"}, "k3j4h5g6f7d8_notes.txt", {"doc_type": "other"})
    assert md["page_number"] is None


def test_metadata_survives_splitting():
    docs = [Document(page_content=C_PAGE, metadata={"page": 7})]
    chunks = make_pc().process_file_content(docs, "k3j4h5g6f7d8_l5.pdf", 1000, 200, asset_config=ASSET_CONFIG)
    assert all(c.metadata["page_number"] == 8 and c.metadata["doc_type"] == "lecture"
               and c.metadata["document_name"] == "Lecture 5 - Loops.pdf" for c in chunks)


def test_get_original_file_name():
    b = BaseController.__new__(BaseController)
    assert b.get_original_file_name("k3j4h5g6f7d8_Lecture1.pdf") == "Lecture1.pdf"
    assert b.get_original_file_name("C:/a/b/k3j4h5g6f7d8_Lab_2.pdf") == "Lab_2.pdf"
    assert b.get_original_file_name("plain.pdf") == "plain.pdf"
    assert b.get_original_file_name(None) is None
