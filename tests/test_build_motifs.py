import unittest

import numpy as np

from src.build_motifs import (
    DEFAULT_MOTIF_SPECS,
    build_motif_labels,
    find_motif_spans,
)


class BuildMotifsTest(unittest.TestCase):
    def test_find_dry_npxx_y_and_tgekp(self):
        seq = "AAADRYBBBNPAAYCCTGEKPDDHTGEKPZZ"
        dry = find_motif_spans(seq, "DRY")
        npxxy = find_motif_spans(seq, r"NP..Y")
        tgekp = find_motif_spans(seq, r"H?TGEKP")

        self.assertEqual(dry, [(3, 6)])
        self.assertEqual(npxxy, [(9, 14)])
        # Both TGEKP and HTGEKP
        self.assertEqual(tgekp, [(16, 21), (23, 29)])

    def test_builds_per_residue_labels_and_local_spans(self):
        proteins = [("P1", "XXDRYXXNPAAYXXTGEKPXX")]
        offsets = {"P1": (0, 21)}

        labels, meta = build_motif_labels(
            proteins,
            offsets,
            total_residues=21,
            require_zinc_finger_overlap=False,
        )

        self.assertEqual(
            meta["concepts"],
            ["Motif::DRY", "Motif::NPxxY", "Motif::TGEKP"],
        )
        self.assertTrue(all(c["grain"] == "motif" for c in meta["concept_metadata"]))

        # DRY at local 2:5
        self.assertTrue(labels[2:5, 0].all())
        self.assertFalse(labels[1, 0])
        # NPxxY at local 7:12
        self.assertTrue(labels[7:12, 1].all())
        # TGEKP at local 14:19
        self.assertTrue(labels[14:19, 2].all())

        self.assertEqual(meta["per_protein_domains"]["P1"]["Motif::DRY"], [(2, 5)])
        self.assertEqual(meta["per_protein_domains"]["P1"]["Motif::NPxxY"], [(7, 12)])
        self.assertEqual(meta["per_protein_domains"]["P1"]["Motif::TGEKP"], [(14, 19)])

    def test_tgekp_can_require_zinc_finger_overlap(self):
        proteins = [
            ("P1", "AAAATGEKPAAAA"),  # TGEKP inside ZF span
            ("P2", "BBBBTGEKPBBBB"),  # TGEKP outside ZF / no ZF
        ]
        offsets = {"P1": (0, 13), "P2": (13, 13)}
        annotations = {
            "P1": {
                "features": [
                    {
                        "type": "Zinc finger",
                        "description": "C2H2-type",
                        "location": {"start": {"value": 3}, "end": {"value": 11}},
                    }
                ]
            },
            "P2": {"features": []},
        }

        labels, meta = build_motif_labels(
            proteins,
            offsets,
            total_residues=26,
            annotation_records=annotations,
            require_zinc_finger_overlap=True,
        )

        concepts = meta["concepts"]
        tgekp_idx = concepts.index("Motif::TGEKP")
        # P1 residues 4:9 (0-based) should be True
        self.assertTrue(labels[4:9, tgekp_idx].all())
        # P2 TGEKP should be dropped because no overlapping ZF
        self.assertFalse(labels[13:26, tgekp_idx].any())
        self.assertEqual(meta["per_protein_domains"]["P1"]["Motif::TGEKP"], [(4, 9)])
        self.assertNotIn("Motif::TGEKP", meta["per_protein_domains"].get("P2", {}))

    def test_default_specs_cover_expected_motifs(self):
        names = {spec.name for spec in DEFAULT_MOTIF_SPECS}
        self.assertEqual(names, {"DRY", "NPxxY", "TGEKP"})


if __name__ == "__main__":
    unittest.main()
