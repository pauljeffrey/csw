from csw.rl import rules as R

OK = "<yor><eng><dev>word a | word b</dev> Mo lo word a ati word b lana. </s>"


def test_parse_ok():
    p = R.parse(OK); assert p["dev_ok"] and p["eos_ok"] and p["devices"] == ["word a", "word b"] and p["final"].startswith("Mo lo")
    assert R.format_penalty(p, "eng_hm")[0] == 0.0


def test_format_failures():
    assert R.format_penalty(R.parse("<yor><eng><dev>a | b</dev> text"), "eng_hm")[0] == 1.0          # no eos
    assert R.format_penalty(R.parse("<yor><eng><dev>a | b text </s>"), "eng_hm")[0] == 1.0           # unclosed dev
    assert R.format_penalty(R.parse("<yor><eng><dev>a</dev> text </s>"), "eng_hm")[0] == 1.0         # one device only
    assert R.format_penalty(R.parse("<draft>abc <yor><eng><dev>a | b</dev> text </s>"), "eng_ood")[0] == 1.0   # unclosed draft


def test_draft_is_stripped_and_not_counted():
    c = "<draft>" + "long draft words " * 40 + "</draft><yor><dev>a b | c d</dev> short final text here. </s>"
    p = R.parse(c); assert p["draft"] and "long draft" not in p["final"]
    assert R.length_penalty(p["final"], "a short source text here", 0.5, 2.5)[0] == 0.0
    assert R.draft_penalty(p["draft"], "a short source text here")[0] > 0.0                             # draft size is capped


def test_length_loop_numbers():
    assert R.length_penalty("x" * 500, "y" * 100, 0.6, 1.8)[0] > 0
    assert R.loop_penalty(("la la la la " * 10).strip())[0] > 0 and R.loop_penalty("a b c d e f g h i j k l m n o p")[0] == 0
    assert R.number_penalty("o ra 4 oja", "he bought 4 items")[0] == 0
    assert R.number_penalty("o ra 5 oja", "he bought 4 items")[0] == 1.0                              # changed numeral
    assert R.number_penalty("o ra merin oja", "he bought 4 items", {"merin"})[0] < R.number_penalty("o ra oja", "he bought 4 items", {"merin"})[0]


def test_english_leak_and_switch():
    assert R.english_leak("the cat and the dog is in the house", False)[0] > 0 and R.english_leak("the cat and the dog is in the house", True)[0] == 0
    fin = "Mo lo ọja ati rí good morning friend lana oni"; pen, d = R.switch_penalty(fin, ["good morning friend"]); assert d["devices_in_text"] == 1
    assert R.switch_penalty("Mo lo ọja lana", ["good morning"])[0] == 1.0


def test_reward_combiner():
    from csw.rl.reward import Reward
    calls = []
    def fake(items): calls.append(len(items)); return [4.0 for _ in items]
    rw = Reward(fake)
    it = {"task": "eng_hm", "lang": "yor", "pin": "eng", "source": "I went to the market yesterday and bought 4 things", "switch_lang": "eng"}
    good = "<yor><eng><dev>went to the | bought 4</dev> Mo went to the oja lana mo bought 4 ohun. </s>"
    r, d = rw([it, it], [good, "<yor><eng><dev>a</dev> bad </s>"])
    assert r[1] == -1.0 and d[1]["gate"] and calls == [1]                       # the failed format never reaches the evaluator
    assert 0.0 < r[0] <= 0.75 and "evaluator" in d[0]
    r2, _ = rw([it], [good.replace("4", "5")]); assert r2[0] < r[0]       # changed numeral costs reward
