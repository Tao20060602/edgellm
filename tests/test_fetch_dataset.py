import hashlib
import io
import sys
import unittest
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import fetch_dataset


class FetchDatasetTests(unittest.TestCase):
    def make_archive(self, entries):
        content = io.BytesIO()
        with zipfile.ZipFile(content, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for name, value in entries:
                archive.writestr(name, value)
        content.seek(0)
        return content

    def test_reads_only_named_member_and_verifies_its_bytes(self):
        wanted = b"the pinned test split\n"
        archive = self.make_archive([
            ("../../outside.txt", b"must never be extracted"),
            ("wikitext-2-raw/wiki.test.raw", wanted),
        ])
        dataset = {
            "member": "wikitext-2-raw/wiki.test.raw",
            "member_sha256": hashlib.sha256(wanted).hexdigest(),
            "member_bytes": len(wanted),
        }
        extracted, report = fetch_dataset.read_verified_member(archive, dataset)
        self.assertEqual(extracted, wanted)
        self.assertEqual(report["sha256"], dataset["member_sha256"])

    def test_member_hash_or_length_mismatch_is_rejected(self):
        archive = self.make_archive([("wikitext-2-raw/wiki.test.raw", b"unexpected")])
        dataset = {
            "member": "wikitext-2-raw/wiki.test.raw",
            "member_sha256": "0" * 64,
            "member_bytes": len(b"unexpected"),
        }
        with self.assertRaisesRegex(ValueError, "member verification failed"):
            fetch_dataset.read_verified_member(archive, dataset)

    def test_duplicate_member_is_rejected(self):
        archive = self.make_archive([
            ("wikitext-2-raw/wiki.test.raw", b"first"),
            ("wikitext-2-raw/wiki.test.raw", b"second"),
        ])
        dataset = {"member": "wikitext-2-raw/wiki.test.raw", "member_sha256": "0" * 64, "member_bytes": 5}
        with self.assertRaisesRegex(ValueError, "exactly one"):
            fetch_dataset.read_verified_member(archive, dataset)


if __name__ == "__main__":
    unittest.main()
