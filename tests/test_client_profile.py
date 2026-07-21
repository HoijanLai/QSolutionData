import unittest

from lib.client_profile import client_profile


class ClientProfileTests(unittest.TestCase):
    def test_default_is_steady_profile_from_document(self):
        profile = client_profile()
        self.assertEqual('steady', profile['name'])
        self.assertEqual(0.055, profile['target_return'])
        self.assertEqual((8, 15), profile['holding_count'])
        self.assertEqual(0.10, profile['r5_cap'])

    def test_accepts_chinese_alias(self):
        profile = client_profile('进取型')
        self.assertEqual('aggressive', profile['name'])
        self.assertEqual((0.40, 0.75), profile['asset_class_ranges']['equity'])

    def test_applies_nested_override_without_changing_template(self):
        changed = client_profile(
            'steady',
            overrides={'asset_class_ranges': {'equity': (0.10, 0.30)}},
        )
        unchanged = client_profile('steady')
        self.assertEqual((0.10, 0.30), changed['asset_class_ranges']['equity'])
        self.assertEqual((0.00, 0.30), unchanged['asset_class_ranges']['equity'])

    def test_rejects_invalid_profile(self):
        with self.assertRaisesRegex(ValueError, 'Unknown client profile'):
            client_profile('unknown')

    def test_rejects_invalid_override_range(self):
        with self.assertRaisesRegex(ValueError, '0 <= min'):
            client_profile(
                'steady',
                overrides={'asset_class_ranges': {'equity': (0.50, 0.20)}},
            )


if __name__ == '__main__':
    unittest.main()
