import unittest

import numpy as np

from src.build_concepts import (
    DEFAULT_EXCLUDED_FEATURE_TYPES,
    build_reference_feature_labels,
    normalize_description,
)


class BuildConceptsTest(unittest.TestCase):
    def test_normalize_description_collapses_whitespace_and_handles_missing(self):
        self.assertEqual(normalize_description("  Protein   kinase\n domain  "), "Protein kinase domain")
        self.assertEqual(normalize_description(""), "(none)")
        self.assertEqual(normalize_description(None), "(none)")

    def test_builds_any_and_description_grains_for_same_span(self):
        proteins = [("P1", "ABCDEFGHIJ")]
        offsets = {"P1": (0, 10)}
        records = {
            "P1": {
                "features": [
                    {
                        "type": "Domain",
                        "description": "Protein kinase",
                        "location": {"start": {"value": 2}, "end": {"value": 5}},
                    }
                ]
            }
        }

        labels, meta = build_reference_feature_labels(
            proteins,
            records,
            offsets,
            total_residues=10,
            min_domain_instances=1,
            min_positive_residues=1,
        )

        self.assertEqual(meta["concepts"], ["Domain::ANY", "Domain::Protein kinase"])
        self.assertEqual([c["grain"] for c in meta["concept_metadata"]], ["any", "subtype"])
        np.testing.assert_array_equal(labels[:, 0], labels[:, 1])
        self.assertTrue(labels[1:5, 0].all())
        self.assertFalse(labels[0, 0])
        self.assertFalse(labels[5, 0])
        self.assertEqual(meta["per_protein_domains"]["P1"]["Domain::ANY"], [(1, 5)])

    def test_excludes_noisy_feature_types(self):
        proteins = [("P1", "ABCDE")]
        offsets = {"P1": (0, 5)}
        records = {
            "P1": {
                "features": [
                    {
                        "type": "Chain",
                        "description": "Mature chain",
                        "location": {"start": {"value": 1}, "end": {"value": 5}},
                    },
                    {
                        "type": "Region",
                        "description": "Disordered",
                        "location": {"start": {"value": 2}, "end": {"value": 3}},
                    },
                ]
            }
        }

        labels, meta = build_reference_feature_labels(
            proteins,
            records,
            offsets,
            total_residues=5,
            min_domain_instances=1,
            min_positive_residues=1,
        )

        self.assertNotIn("Chain", {c["feature_type"] for c in meta["concept_metadata"]})
        self.assertTrue(set(DEFAULT_EXCLUDED_FEATURE_TYPES).issuperset({"Chain"}))
        self.assertEqual(meta["concepts"], ["Region::ANY", "Region::Disordered"])
        self.assertEqual(labels.shape, (5, 2))

    def test_frequency_filter_keeps_common_any_and_drops_rare_subtype(self):
        proteins = [("P1", "ABCDE"), ("P2", "FGHIJ")]
        offsets = {"P1": (0, 5), "P2": (5, 5)}
        records = {
            "P1": {
                "features": [
                    {
                        "type": "Domain",
                        "description": "Rare A",
                        "location": {"start": {"value": 1}, "end": {"value": 2}},
                    }
                ]
            },
            "P2": {
                "features": [
                    {
                        "type": "Domain",
                        "description": "Rare B",
                        "location": {"start": {"value": 1}, "end": {"value": 2}},
                    }
                ]
            },
        }

        _labels, meta = build_reference_feature_labels(
            proteins,
            records,
            offsets,
            total_residues=10,
            min_domain_instances=2,
            min_positive_residues=10,
        )

        self.assertEqual(meta["concepts"], ["Domain::ANY"])
        self.assertEqual(meta["concept_metadata"][0]["n_domains"], 2)


if __name__ == "__main__":
    unittest.main()
