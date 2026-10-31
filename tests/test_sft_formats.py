"""Unit tests of the completion / prompt string formats and the rule-based rewards."""
import unittest
from csw.sft import fmt
from csw.sft import devices as dv
from csw.sft import rewards as R


def lex():
    rows = [{"lang": "yor", "eng": "The village has a high quality yam crop", "m_text": "Abule naa ni irugbin isu", "hm": "Abule naa ni high quality yam crop", "ood": "x",
             "scenario": {"switch": "English"}} for _ in range(5)]
    return dv.Lexicon.build(rows)


class T(unittest.TestCase):
    def test_prompt_formats(self):
        self.assertEqual(fmt.prompt_m_text("yor", "Hi."), "Translate the text to Yoruba:  Hi.")          # two spaces, as the rl2b card requires
        self.assertEqual(fmt.prompt_hm("yor", "eng", "Hi."), "<hm> Yoruba+English:  Hi.")
        self.assertEqual(fmt.prompt_ood("pcm", "Hi."), "<ood> Pidgin:  Hi.")
        self.assertTrue(fmt.gemma_text("p", "c").endswith("<start_of_turn>model\nc<end_of_turn>\n"))

    def test_roundtrip(self):
        c = fmt.completion_hm("yor", "eng", ["high quality", "yam"], "Abule naa ni high quality yam")
        p = fmt.parse("hm", "yor", c)
        self.assertTrue(p["ok"]); self.assertEqual(p["tags"], ["yor", "eng"]); self.assertEqual(p["devices"], ["high quality", "yam"])
        o = fmt.parse("ood", "pcm", fmt.completion_ood("pcm", ["a b", "c"], "text here", "hau"))
        self.assertEqual(o["tags"], ["pcm", "hau"]); self.assertEqual(o["devices"], ["a b", "c"])
        m = fmt.parse("m_text", "yor", "yor E kaaro")
        self.assertEqual((m["tags"], m["text"]), (["yor"], "E kaaro"))
        self.assertFalse(fmt.parse("hm", "yor", "no tags at all")["ok"])

    def test_hm_devices(self):
        r = {"lang": "yor", "scenario": {"switch": "English"}, "m_text": "Abule naa ni irugbin isu", "hm": "Abule naa ni high-quality yam crop, wallahi."}
        d = dv.hm_devices(r, lex())
        self.assertIn("quality yam crop", " ".join(d["devices"]).replace("high ", ""))
        self.assertGreater(d["switch_ratio"], 0.3)

    def test_ood_devices(self):
        r = {"ood": "O fo baka lojo ojo", "devices": ["fo baka (idiom: break like a plate)", "not present (x)"]}
        self.assertEqual(dv.ood_devices(r)["devices"], ["fo baka"])

    def test_rewards_ordering(self):
        s = {"task": "hm", "base": "yor", "switch": "eng", "english": "The village has a high quality yam crop", "n_sent": 1, "must_keep": []}
        good = fmt.completion_hm("yor", "eng", ["high quality", "yam crop"], "Abule naa ni high quality yam crop")
        bad = "no tags"
        self.assertEqual(R.reward(s, bad, lex())["total"], 0.0)
        self.assertGreater(R.reward(s, good, lex())["total"], R.reward(s, good.replace("<yor>", "<xxx>"), lex())["total"])

    def test_must_keep(self):
        mk = R.must_keep("Mama Shade paid 2,500 naira. Then Tunde left.")
        self.assertIn("2,500", mk); self.assertIn("Shade", mk); self.assertIn("Tunde", mk); self.assertNotIn("Then", mk)


if __name__ == "__main__":
    unittest.main()
