from __future__ import annotations

import json
import subprocess
import sys
import unittest

from src.foundation_model_registry import (
    get_model_spec,
    preprocess_sequence,
    ranked_candidates,
    residue_token_slice,
)


class FoundationModelRegistryTests(unittest.TestCase):
    def test_esm2_650m_is_highest_priority_nonbaseline_candidate(self):
        ranked = ranked_candidates()
        self.assertGreaterEqual(len(ranked), 2)
        self.assertEqual(ranked[0].key, "esm2_650m")
        self.assertTrue(ranked[0].has_public_sae)
        self.assertTrue(ranked[0].comparable_to_baseline)

    def test_prott5_preprocessing_replaces_rare_amino_acids_and_spaces_residues(self):
        spec = get_model_spec("prot_t5_xl_uniref50")
        self.assertEqual(preprocess_sequence("ACDUZOB", spec), "A C D X X X X")

    def test_residue_token_slice_accounts_for_special_tokens(self):
        self.assertEqual(residue_token_slice(10, 512, "esm"), (1, 11))
        self.assertEqual(residue_token_slice(600, 512, "esm"), (1, 511))
        self.assertEqual(residue_token_slice(10, 512, "suffix_eos"), (0, 10))
        self.assertEqual(residue_token_slice(600, 512, "suffix_eos"), (0, 511))

    def test_registry_cli_json_does_not_import_transformers(self):
        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "src.foundation_model_registry",
                "--model-key",
                "esm2_650m",
                "--format",
                "json",
            ],
            check=True,
            text=True,
            capture_output=True,
        )
        payload = json.loads(proc.stdout)
        self.assertEqual(payload[0]["key"], "esm2_650m")


if __name__ == "__main__":
    unittest.main()
