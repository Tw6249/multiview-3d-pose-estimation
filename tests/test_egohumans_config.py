import json
from pathlib import Path
import unittest

from scripts.download_egohumans import CONFIG_PATH, build_audit_report, load_config


class TestEgoHumansConfig(unittest.TestCase):
    def test_config_and_recommended_subset_contract(self):
        cfg = load_config()
        self.assertEqual(cfg["google_drive_root_id"], "1JD963urzuzV_R_6FOVOtlx8UupwUuknR")
        self.assertEqual(cfg["owner_account"], "egohumans2023@gmail.com")
        rec = cfg["recommended_subset"]["targets"]
        self.assertTrue(len(rec) >= 3)
        for target_name in rec:
            self.assertIn(target_name, cfg["targets"])
            meta = cfg["targets"][target_name]
            self.assertGreater(meta["bytes"], 100_000_000)
            self.assertRegex(meta["md5"], r"^[0-9a-f]{32}$")
            self.assertNotIn(":", meta["rel_path"], "Windows NTFS paths must not contain ':'")

    def test_audit_report_and_junctions(self):
        cfg = load_config()
        storage_root = Path(cfg["storage_root"])
        if storage_root.exists():
            audit = build_audit_report(storage_root, cfg)
            self.assertTrue(audit["project_junctions"]["data/egohumans"])
            self.assertTrue(audit["project_junctions"]["outputs/public_datasets/egohumans"])
            self.assertTrue(audit["local_assets_downloaded"]["official_repository_cloned"])
            self.assertEqual(audit["local_assets_downloaded"]["sequence_yaml_configs_count"], 133)
            self.assertEqual(audit["local_assets_downloaded"]["catalog_total_files"], 142)


if __name__ == "__main__":
    unittest.main()
