"""Per-language profiles for the code-switching corpus (eng / m_text / hm / ood).

Languages = SabiYarn's kept set intersected with the confirmed code-switching list. Switch languages for
som and orm are confirmed switch cases; the rest follow the ranked table of confirmed switch languages. efi/ibb/urh are not in that
table and are deferred (ask before adding).
"""
from __future__ import annotations

from csw.paths import DATA_ROOT
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Profile:
    code: str
    name: str
    switch: tuple[str, ...]            # code-switch languages, most common first
    dialects: tuple[str, ...]          # regional varieties a sample may be pinned to (first = default/standard)
    street: str                        # what "street / vernacular" means for this language
    orthography: str                   # diacritic / script rules the output must respect
    persona_regions: tuple[str, ...]   # where the speakers live (used for English-seed locale)
    tier: str                          # A = LLM reliable-ish, B = medium, C = weak (smaller quota, extra checks)
    lid_hint: str = ""
    extra: str = ""


P = {p.code: p for p in [
    Profile("yor", "Yoruba", ("English", "Nigerian Pidgin"), ("Standard/Oyo", "Lagos (Eko)", "Ibadan", "Ijebu", "Ekiti"),
            "Lagos street Yoruba: ìgboro slang, owambe/market banter, oríkì, ìjálá-style flourishes, proverbs (òwe)",
            "Full tone marks (à á, è é, ì í, ò ó, ù ú) and sub-dots (ẹ ọ ṣ) on every word. Never ASCII 's.' / 'o.' hacks.",
            ("Lagos", "Ibadan", "Abeokuta"), "A"),
    Profile("hau", "Hausa", ("English", "Arabic"), ("Standard/Kano (Kananci)", "Sokoto", "Zaria", "Katsina"),
            "Kano/Kaduna street Hausa: kasuwa and motor-park banter, Kannywood film slang, karin magana, roƙo/yabo formulas, Arabic-derived religious phrases",
            "Hooked letters ɓ ɗ ƙ ƴ mandatory; mark long vowels/tones only where the standard orthography does.",
            ("Kano", "Kaduna", "Maiduguri"), "A"),
    Profile("ibo", "Igbo", ("English", "Nigerian Pidgin"), ("Igbo Izugbe/standard", "Onitsha", "Owerri", "Enugu"),
            "Onitsha-market and Aba-trader Igbo, ịgba egwu/umu-nna banter, ilu (proverbs), nkọwa idioms, praise names",
            "Sub-dots ị ọ ụ ṅ mandatory; tone marks (à á ā) on every word where the standard marks them.",
            ("Onitsha", "Enugu", "Owerri"), "A"),
    Profile("pcm", "Nigerian Pidgin", ("Yoruba", "Igbo", "Hausa"), ("Lagos", "Warri/Delta", "Port Harcourt", "Calabar"),
            "deep basilect Pidgin: Warri/Lagos ghetto slang, proverbs, 'wetin/dey/abeg/wahala' idiom, ideophones",
            "No diacritics. Standard Naija Pidgin spelling (dey, wetin, sabi, abeg, wahala).",
            ("Lagos", "Warri", "Port Harcourt"), "A",
            extra="Pidgin is itself English-lexified: m_text = standard written Pidgin; hm = Pidgin mixed with standard Nigerian English register shifts; ood = deep basilect with ideophones and proverbs."),
    Profile("twi", "Twi (Akan)", ("English", "Ghanaian Pidgin"), ("Asante Twi", "Akuapem Twi", "Fante (flag if used)"),
            "Accra/Kumasi youth Twi: chale/herh banter, kasa-mu proverbs (mmebusɛm), ideophones, drum-language praise",
            "ɛ ɔ mandatory; tone marks not written.",
            ("Kumasi", "Accra", "Koforidua"), "A"),
    Profile("ewe", "Ewe", ("French", "English"), ("Anlo", "Tongu", "Lomé/Togo"),
            "Anlo/Lomé street Ewe: nyagbɔgbɔ (proverbs), halo praise poetry, market and trotro banter",
            "ɖ ɛ ƒ ɣ ɔ ŋ ʋ mandatory; tone marks (à á ǎ ã) where the standard writes them.",
            ("Accra", "Lomé", "Ho"), "B"),
    Profile("fon", "Fon", ("French", "Yoruba"), ("Cotonou", "Abomey"),
            "Cotonou street Fon: zémidjan/market banter, proverbs, ideophones, Vodun-linked sayings",
            "ɖ ɔ ɛ and tone marks (à á ǎ) per the Beninese standard.",
            ("Cotonou", "Abomey-Calavi"), "C"),
    Profile("kin", "Kinyarwanda", ("French", "English", "Swahili"), ("Kigali", "Standard"),
            "Kigali youth Kinyarwanda: imigani/ibisakuzo, 'agashya' slang, ikinyarwanda cy'umuhanda mixed with Swahili/French",
            "Standard orthography; tone not written; keep apostrophes (n'ubwo, cy'u).",
            ("Kigali", "Huye", "Musanze"), "B"),
    Profile("sna", "Shona", ("English", "Ndebele"), ("Zezuru (Harare)", "Karanga", "Manyika"),
            "Harare street Shona ('Shonglish'): madimikira/ngano-style flourishes, zvirevo/tsumo proverbs, ideophones, 'tsotsi' slang",
            "Standard Shona orthography (no tone marks; sv/zv/tsv/dzv, bh/mb).",
            ("Harare", "Bulawayo", "Masvingo"), "A"),
    Profile("zul", "Zulu (isiZulu)", ("English", "Afrikaans"), ("Durban", "Johannesburg/Soweto (iscamtho)", "Rural KZN"),
            "township isiZulu: iscamtho/tsotsitaal, izaga, izisho, ukuphawula banter, ideophones (ukuphawula)",
            "Standard orthography; no tone marks.",
            ("Durban", "Soweto", "Pietermaritzburg"), "A"),
    Profile("xho", "Xhosa (isiXhosa)", ("English", "Afrikaans"), ("Eastern Cape", "Cape Town townships (iscamtho)"),
            "township isiXhosa: iscamtho, izaci, amaqhalo, ideophones, Eastern Cape rural idiom",
            "Standard orthography with clicks (c q x gc ...); no tone marks.",
            ("East London", "Cape Town", "Mthatha"), "A"),
    Profile("swh", "Swahili", ("English", "Arabic", "Sheng"), ("Standard/Tanzania (Dar)", "Nairobi (Sheng)", "Mombasa/Coastal", "Uganda"),
            "Nairobi Sheng and Dar 'kihuni' street Swahili: methali, nahau, taarab/bongo-flava slang, Arabic-derived religious/coastal phrases",
            "Standard Swahili orthography (no tone marks).",
            ("Nairobi", "Dar es Salaam", "Mombasa"), "A"),
    Profile("orm", "Oromo (Afaan Oromoo)", ("Amharic", "English"), ("Central/Shewa", "Hararge (East)", "Wollega (West)", "Borana"),
            "Oromo street/campus register: mammaaksa, geerarsa, jechamaa proverbs, urban Finfinne speech mixed with Amharic",
            "Qubee Latin orthography with doubled vowels and digraphs (ch dh ny ph sh ts zh); apostrophe for hudhaa. Amharic switches may be written in Ge'ez script (keep it) or Qubee.",
            ("Addis Ababa/Finfinne", "Adama", "Jimma", "Harar"), "B"),
    Profile("som", "Somali", ("Arabic", "English", "Italian", "Amharic"), ("Standard (Maxaa-tiri/Northern)", "Benaadir (Mogadishu)", "Hargeisa", "Diaspora"),
            "Mogadishu/Hargeisa/diaspora youth Somali: maahmaah, murti, gabay-style imagery, 'wallaahi/xaaraam/inshallah' religious phrases, Italian-derived (e.g. in Benaadir) and Arabic-derived words",
            "Standard Latin orthography (apostrophe for ', digraphs kh dh sh); no tone marks.",
            ("Mogadishu", "Hargeisa", "Garowe", "Minneapolis"), "B",
            extra="switch language varies by speaker: Arabic (religion/formal), English (tech/diaspora), Italian (Benaadir/legacy words), Amharic (Somali-region Ethiopia). Pin ONE per sample."),
    Profile("wol", "Wolof", ("French", "Arabic"), ("Dakar urban", "Saint-Louis", "Gambian"),
            "urban Dakar Wolof: 'Dakar-Dakar' French-Wolof banter, waxtaan, kàddu gu xar proverbs, griot praise, tèngal",
            "Official CLAD orthography: ñ ŋ ë à é ó and doubled vowels mandatory.",
            ("Dakar", "Thiès", "Banjul"), "B"),
    Profile("tsn", "Setswana", ("English", "Afrikaans"), ("Botswana (Gaborone)", "North-West/Mahikeng", "Pretoria (Sepitori)"),
            "Gaborone/Pretoria street Setswana: maele (proverbs), mafoko a a tswang, Sepitori township slang, praise poetry",
            "Standard orthography with ê ô š where the standard writes them.",
            ("Gaborone", "Mahikeng", "Pretoria"), "B"),
    Profile("nya", "Chichewa (Nyanja)", ("English", "Tumbuka"), ("Malawi (Chichewa)", "Lusaka (Nyanja)", "Mozambique"),
            "Blantyre/Lilongwe/Lusaka street Chichewa-Nyanja: miyambi (proverbs), gule-wamkulu-flavoured imagery, urban English mixing",
            "Standard Chichewa orthography; no tone marks.",
            ("Lilongwe", "Blantyre", "Lusaka"), "B"),
    Profile("afr", "Afrikaans", ("English", "Zulu"), ("Standard", "Kaaps (Cape Flats)", "Gauteng urban", "Namibian"),
            "Kaaps/Gauteng street Afrikaans: sêgoed (sayings), 'lekker/ag/jong' interjections, Cape Flats slang, English and township words mixed in",
            "Full Afrikaans diacritics (ê é è ë ï ô ŉ).",
            ("Cape Town", "Pretoria", "Windhoek"), "A"),
    Profile("sot", "Sesotho", ("English", "Afrikaans"), ("Lesotho", "Free State/Gauteng township"),
            "Maseru/Soweto street Sesotho: maele le dithoko (proverbs, praise), township slang, Afrikaans-derived words",
            "Standard orthography with ê ô š where used.",
            ("Maseru", "Bloemfontein", "Soweto"), "B"),
    Profile("efi", "Efik", ("English", "Nigerian Pidgin"), ("Calabar/Standard Efik", "Uyo-Calabar urban"),
            "Calabar street Efik: Calabar-Pidgin-Efik banter, ndok/ukpọk proverbs (ọfọn ikpe), Ekpe/Nsibidi-linked sayings, owambe praise",
            "Standard Efik orthography with ọ ọ̀ ụ ị ń and tone marks as written in Efik Bible/primers; no ASCII hacks.",
            ("Calabar", "Uyo", "Lagos"), "C"),
    Profile("ibb", "Ibibio", ("English", "Nigerian Pidgin"), ("Uyo/Standard Ibibio", "Ikot Ekpene", "Eket-Oron borderlands"),
            "Uyo/Ikot Ekpene street Ibibio: Pidgin-Ibibio banter, ndiọk-ndiọk proverbs, Ekpo/Ekpe-linked sayings, market and okada speech",
            "Standard Ibibio orthography with ọ ụ ị ṅ ḿ ń and tone marks where the standard writes them.",
            ("Uyo", "Ikot Ekpene", "Port Harcourt"), "C"),
    Profile("lin", "Lingala", ("French", "Kituba"), ("Kinshasa", "Brazzaville", "Equateur"),
            "Kinshasa street Lingala: Indoubill / Lingala ya libanga (youth Lingala-French slang), rumba-song imagery, mpiko/lisapo proverbs, 'mboka' banter, sapeur flourishes",
            "Standard Lingala orthography: ɛ ɔ mandatory, tone marks á é í ó ú where the standard writes them.",
            ("Kinshasa", "Brazzaville", "Kisangani"), "B"),
    Profile("run", "Kirundi", ("French", "Swahili"), ("Bujumbura", "Standard"),
            "Bujumbura youth Kirundi: ivyivugo/imigani (proverbs), inka-and-drum imagery, urban Kirundi mixed with Swahili and French",
            "Standard Kirundi orthography; no tone marks; keep apostrophes.",
            ("Bujumbura", "Gitega", "Ngozi"), "B"),
    Profile("bam", "Bambara (Bamanankan)", ("French", "Wolof"), ("Bamako standard", "Segou", "Sikasso/Dioula-influenced"),
            "Bamako street Bambara: Bamanankan ni tubabukan (French mixing), jamu/praise-singing (jeli) imagery, proverbs (ntalenw, fɔlɔw), grin (tea-circle) banter",
            "Standard Bambara orthography: ɛ ɔ ɲ ŋ mandatory; tone marks not written.",
            ("Bamako", "Segou", "Sikasso"), "B"),
    Profile("tum", "Tumbuka", ("English", "Chichewa"), ("Northern Malawi (Mzuzu)", "Zambia (Eastern Province)"),
            "northern-Malawi street Tumbuka: miyambi (proverbs), vimbuza/dance imagery, urban Chichewa and English mixing",
            "Standard Tumbuka orthography; no tone marks.", ("Mzuzu", "Rumphi", "Chipata"), "B"),
    Profile("tso", "Xitsonga (Tsonga)", ("English", "Zulu"), ("Limpopo/Giyani", "Maputo/Mozambican Tsonga", "Johannesburg township"),
            "township Xitsonga: swivuriso (proverbs), xibelani-dance and migrant-labour imagery, Joburg slang mixed with Zulu and English",
            "Standard Tsonga orthography (hl, tl, ndl, ts digraphs; no tone marks).", ("Giyani", "Johannesburg", "Maputo"), "B"),
    Profile("nso", "Sepedi (Northern Sotho)", ("English", "Afrikaans"), ("Polokwane/Limpopo", "Pretoria (Sepitori)", "Mpumalanga"),
            "township Sepedi: dika (proverbs), Pretoria Sepitori slang, kwaito-flavoured speech, Afrikaans-derived words",
            "Standard Sepedi orthography with ê ô š where the standard writes them.", ("Polokwane", "Pretoria", "Tzaneen"), "B"),
    Profile("lug", "Luganda", ("English", "Swahili"), ("Kampala", "Buganda rural"),
            "Kampala street Luganda: byeya/ebigambo (proverbs), boda-boda and taxi-park banter, Kiganda courtesy formulae, English and Swahili mixing",
            "Standard Luganda orthography: ŋ and doubled consonants/vowels as written; no tone marks.", ("Kampala", "Masaka", "Entebbe"), "B"),
    Profile("kik", "Gikuyu (Kikuyu)", ("English", "Swahili"), ("Nyeri/Kiambu", "Nairobi urban", "Murang'a"),
            "Nairobi/Central Kenya Gikuyu: thimo (proverbs), matatu and market banter, gicandi riddles, Sheng-flavoured mixing",
            "Standard Gikuyu orthography: ĩ ũ mandatory; no tone marks.", ("Nyeri", "Nairobi", "Kiambu"), "B"),
    Profile("luo", "Dholuo", ("English", "Swahili"), ("Kisumu/Siaya", "Nairobi urban", "Tanzania (Tarime)"),
            "Kisumu/Nairobi Dholuo: ngero (proverbs), benga/ohangla music imagery, lakeside fishing and market banter, Sheng mixing",
            "Standard Dholuo orthography (ng' digraph, ny, dh, th); no tone marks.", ("Kisumu", "Siaya", "Nairobi"), "C"),
    Profile("mos", "Mooré", ("French", "Dioula"), ("Ouagadougou", "Central plateau"),
            "Ouagadougou street Mooré: yeelgo/sɛbgo (sayings), moped-and-market banter, French and Dioula mixing",
            "Standard Mooré orthography: ɛ ɩ ʋ ɔ and ã ẽ ĩ õ ũ where written; tone marks not written.", ("Ouagadougou", "Koudougou", "Bobo-Dioulasso"), "C"),
    Profile("ssw", "siSwati", ("English", "Zulu"), ("Mbabane/Manzini", "Rural Eswatini"),
            "Eswatini siSwati: tisho (sayings), incwala/umhlanga ceremonial imagery, royal-courtesy forms, English and Zulu mixing",
            "Standard siSwati orthography; no tone marks.", ("Mbabane", "Manzini", "Piggs Peak"), "C"),
    Profile("nde", "Ndebele (Zimbabwe)", ("English", "Shona"), ("Bulawayo", "Matabeleland rural"),
            "Bulawayo street Ndebele: izaga (proverbs), township slang, Shona and English mixing",
            "Standard Ndebele orthography; no tone marks.", ("Bulawayo", "Gwanda", "Hwange"), "C"),
    Profile("umb", "Umbundu", ("Portuguese", "Kimbundu"), ("Huambo/Bie", "Luanda urban"),
            "Huambo/Luanda Umbundu: ongeleka (proverbs), kuduro/semba imagery, Angolan Portuguese mixing",
            "Standard Umbundu orthography (Latin; nasal digraphs); no tone marks.", ("Huambo", "Benguela", "Luanda"), "C"),
    Profile("kmb", "Kimbundu", ("Portuguese", "Umbundu"), ("Luanda", "Malanje"),
            "Luanda Kimbundu: jisabu/misoso (proverbs and tales), semba and kuduro imagery, Angolan Portuguese street slang",
            "Standard Kimbundu orthography (Latin); no tone marks.", ("Luanda", "Malanje", "Ndalatando"), "C"),
    Profile("kon", "Kikongo (Kituba)", ("French", "Lingala"), ("Kituba / Kikongo ya leta", "Western DRC (Kinshasa-Kongo Central)"),
            "Brazzaville/Kinshasa Kituba: mpovi/ngana (sayings and tales), rumba imagery, French and Lingala mixing",
            "Standard Kikongo/Kituba orthography; ɛ ɔ where written; no tone marks.", ("Brazzaville", "Pointe-Noire", "Matadi"), "C"),
    Profile("sag", "Sango", ("French", "Lingala"), ("Bangui",),
            "Bangui street Sango: lingua-franca banter, market and river-port speech, French and Lingala mixing",
            "Standard Sango orthography: ë ö and circumflex-marked tones where written.", ("Bangui", "Bimbo", "Berbérati"), "C"),
    Profile("ful", "Fula (Fulfulde / Pulaar)", ("French", "English"), ("Fulfulde (Nigeria/Cameroon)", "Pulaar (Senegal/Guinea)", "Fuuta Jallon"),
            "Fula pastoral and urban speech: tindinol/baaŋgal sayings, cattle and pulaaku (honour-code) imagery, praise-singing, French/English mixing",
            "Standard Fula orthography: ɓ ɗ ŋ ɲ ƴ mandatory (Latin script).", ("Maroua", "Conakry", "Kano"), "C"),
    Profile("mlg", "Malagasy", ("French", "English"), ("Standard/Merina (Antananarivo)", "Coastal (Betsimisaraka/Sakalava)"),
            "Antananarivo street Malagasy: ohabolana (proverbs), hira-gasy imagery, market and taxi-be banter, French mixing",
            "Standard Malagasy orthography (no diacritics except in loans; ô, ò in dialect words).", ("Antananarivo", "Toamasina", "Mahajanga"), "C"),
    Profile("amh", "Amharic", ("English", "Arabic"), ("Addis Ababa standard", "Gondar/Gojjam", "Dire Dawa/Harar"),
            "Addis Ababa street Amharic: ተረት/ምሳሌያዊ አነጋገር (proverbs and wax-and-gold wordplay), minibus and cafe banter, English and Arabic mixing",
            "Ge'ez script (fidel) throughout m_text, hm and ood; code-switched words may be written in Ge'ez or Latin as native writers do.", ("Addis Ababa", "Bahir Dar", "Dire Dawa"), "B"),
    Profile("tir", "Tigrinya", ("Amharic", "English"), ("Asmara (Eritrea)", "Mekelle (Tigray)"),
            "Asmara/Mekelle street Tigrinya: ምስላ/ምስላ (proverbs), coffee-ceremony and market banter, Amharic and English mixing",
            "Ge'ez script (fidel) throughout m_text, hm and ood; code-switched words may be written in Ge'ez or Latin as native writers do.", ("Asmara", "Mekelle", "Adi Keyh"), "C"),
    Profile("bin", "Edo (Ghagbo)", ("English", "Nigerian Pidgin"), ("Benin City urban", "Edo rural"),
            "Benin City street Edo: ebe/ọtaọ (sayings), Benin-bronze and palace imagery, Pidgin and English mixing",
            "Standard Edo orthography with ẹ ọ and tone marks where the standard writes them.", ("Benin City", "Ekpoma", "Auchi"), "C"),
    Profile("gpe", "Ghanaian Pidgin", ("Twi", "Ewe", "Ga"), ("Accra", "Kumasi", "Takoradi/Tema"),
            "Accra/Kumasi street Pidgin: 'chale', 'herh', trotro and market banter, campus and church-youth speech, Twi/Ewe/Ga exclamations and proverbs mixed in",
            "Ghanaian Pidgin spelling (wey/dey/sef/chale/den/dem); Twi, Ewe and Ga words keep their full letters and diacritics (ɛ ɔ ŋ ɖ ƒ ɣ ʋ, tone marks where written).",
            ("Accra", "Kumasi", "Takoradi"), "C",
            extra="Ghanaian Pidgin is its own variety: not Nigerian Pidgin. English is NOT a switch language; the partners are Ghana's own languages."),
    Profile("wes", "Cameroonian Pidgin (Kamtok)", ("French", "Ewondo", "Duala"), ("Douala/Littoral", "Bamenda/North-West", "Yaounde/Centre"),
            "Douala/Bamenda street Kamtok: njangi (savings circle) and bayam-sellam banter, mimbo (palm-wine) and bush-taxi imagery, French and local-language exclamations mixed in",
            "Kamtok spelling (wuna, dem, na, fo, di, chop, sey); Ewondo and Duala words keep their full letters and diacritics (ɛ ɔ ŋ ɓ, tone marks where written); French words with French accents.",
            ("Douala", "Bamenda", "Yaounde"), "C",
            extra="Cameroonian Pidgin (Kamtok) is its own variety: not Nigerian Pidgin. English is NOT a switch language; the partners are French and Cameroon's own languages."),
    Profile("urh", "Urhobo", ("English", "Nigerian Pidgin"), ("Agbarha-Ame/Standard Urhobo", "Okpe", "Warri urban"),
            "Warri/Effurun street Urhobo: Warri-Pidgin-Urhobo banter, ẹtoto/proverbs, ọmọ-ovwa praise, odjẹ-style flourishes, ogbẹrẹ idiom",
            "Standard Urhobo orthography with ẹ ọ ṣ vw ph and tone marks where the standard writes them.",
            ("Warri", "Effurun", "Sapele"), "C"),
]}

LANGS = tuple(P)
DEFERRED = {}   # urh (Urhobo) was removed and then reinstated on 2026-10-18 when 13 corrected examples were added (Gemma alone cannot write it: chrF ~ a random sentence even with real verse pairs)

GENRES = {   # genre -> (weight, one-line brief used in the scenario card)
    "chat": (0.15, "a WhatsApp / voice-note style exchange between friends, relatives or colleagues"),
    "market_bargain": (0.08, "bargaining, trading, haggling, transport or errands"),
    "advice_family": (0.1, "advice, scolding, family or community mediation, relationships"),
    "explain_teach": (0.12, "someone explaining a concept (science, health, money, law, tech, farming) to a younger or less-informed person, using local analogies"),
    "story": (0.12, "a short folktale, personal anecdote, or a child's story told aloud"),
    "opinion": (0.08, "an opinion, debate or argument on a social, political or generational topic"),
    "instruct": (0.08, "step-by-step practical instructions (cooking, farming, repairs, ceremonies)"),
    "tradition_law": (0.07, "customary law, kinship, inheritance, philosophy, proverb discussion, rites of passage"),
    "emotion_ceremony": (0.08, "greetings, praise, condolence, blessing, congratulation, wedding/naming/funeral speech"),
    "faith": (0.04, "religious reflection, prayer, preaching, or religious banter (Christian, Muslim, traditional)"),
    "humor": (0.08, "jokes, teasing, banter, gist/gossip"),
}

LENGTHS = {   # band -> (weight, sentence range text, max_tokens)   (v2 length spec: requested length is enforced, see SENT_RANGE + gates.check)
    "short": (0.45, "2-5 sentences", 6000),
    "medium": (0.35, "6-12 sentences", 10000),
    "long": (0.20, "13-20 sentences, coherent as one continuous piece or dialogue", 18000),
    "brief": (0.0, "1-3 short sentences in plain everyday vocabulary", 4000),       # low / very-low / weakest-mid languages: guaranteed quality
    "single": (0.0, "ONE simple sentence of at most 15 words in plain everyday vocabulary", 1500),   # legacy (v3.9 and earlier plans)
}
SENT_RANGE = {"brief": (1, 3), "short": (2, 5), "medium": (6, 12), "long": (13, 20), "single": (1, 1)}   # requested number of sentences in eng, per band

TRACKS = {"A": 0.55, "B": 0.30, "C": 0.15}   # A anchor-seeded English, B native-seeded, C English story/seed localisation


# which seed genres may inspire which planned genres (thin genres borrow voice/topics from conversational seeds)
COMPAT = {"chat": {"chat"}, "advice_family": {"chat"}, "market_bargain": {"chat"}, "humor": {"chat"}, "emotion_ceremony": {"chat", "faith"},
          "story": {"story"}, "explain_teach": {"explain_teach"}, "instruct": {"explain_teach"}, "opinion": {"opinion"},
          "tradition_law": {"tradition_law", "faith"}, "faith": {"faith"}}


# ---- how well the generator (Gemma-4-31B) actually handles each language -------------------------------------------------
# Measured 2026-10-18 by lang_support.py: share of words in generated m_text/ood that never occur in real text of that language,
# relative to held-out real text (ratio ~1 = natural, >>1 = invented/garbled words), plus earlier judge scores for tier-C languages.
# ok          : ratio_m <= 1.3 and ratio_ood <= 1.7  -> full weight
# borderline  : in between, or weak judge scores     -> half weight
# dropped     : ratio_m > 2.2 (or both > 3)          -> no GPU time until a better generator is used
SUPPORT = {
    "ok": ["pcm", "afr", "hau", "yor", "ibo", "orm", "mlg", "bam", "swh", "zul", "tsn", "ful", "wol", "sot", "som", "xho", "nya", "lug", "sna", "nso", "amh", "kin"],
    "borderline": ["ewe", "fon", "kon", "nde", "ssw", "run", "luo", "tum", "twi", "lin", "tir", "tso", "kik", "gpe", "wes"],   # gpe/wes: not yet measured
    "dropped": ["kmb", "urh", "umb", "ibb", "mos", "efi", "sag", "bin"],
}
HOLD = list(SUPPORT["dropped"])    # held: generated LAST, only if time and compute remain - gen_rounds.sh skips them unless RUN_HELD=1
WEIGHTS = {**{l: 1.0 for l in SUPPORT["ok"]}, **{l: 0.5 for l in SUPPORT["borderline"]}, **{l: 0.0 for l in SUPPORT["dropped"]}}


# ---- per-language targets (rows) - user plan 2026-10-18 -------------------------------------------------------------------
# high (top 20): 6,000 | mid: 2-4k | low: <=1k | very low: 150-300 as ONE simple sentence | pidgin: 12k, generated LAST
HIGH = ["swh", "hau", "yor", "ibo", "zul", "amh", "som", "orm", "xho", "afr", "sna", "kin", "nya", "tsn", "sot", "wol", "bam", "mlg", "ful", "lug"]
MID3K = ["twi", "ewe", "lin", "run", "nso"]
MID2K = ["tso", "tum", "kik", "luo", "fon", "tir", "nde", "ssw"]
LOW = ["kon"]
PIDGIN_VARIANTS = ["gpe", "wes"]          # Ghanaian Pidgin, Cameroonian Pidgin (Kamtok): scaled like Nigerian Pidgin through given English (track D)
VERY_LOW = ["efi", "ibb", "urh", "bin", "kmb", "umb", "mos", "sag"]
BRIEF = set(LOW) | set(VERY_LOW) | set(MID2K)   # these languages always get 1-3 short sentences per row (quality over length); MID3K, high-resource and Pidgin use the 2-5 / 6-12 / 13-20 bands


def _gold_counts() -> dict:
    """Number of native-corrected rows per language in exemplars/augmented_gold.jsonl (+ the universal gold.jsonl): more examples -> a bigger target."""
    import json as _j
    from pathlib import Path as _P
    out: dict = {}
    for name in ("augmented_gold.jsonl", "gold.jsonl"):
        f = DATA_ROOT / "exemplars" / name
        if not f.exists():
            continue
        dec, txt, i = _j.JSONDecoder(), f.read_text(encoding="utf-8"), 0
        while True:
            while i < len(txt) and txt[i].isspace():
                i += 1
            if i >= len(txt):
                break
            try:
                o, i = dec.raw_decode(txt, i)
            except Exception:
                nxt = txt.find("\n{", i + 1)
                i = nxt + 1 if nxt > 0 else len(txt)
                continue
            if "status" not in o and o.get("lang"):
                out[o["lang"]] = out.get(o["lang"], 0) + 1
    return out


GOLD_N = _gold_counts()


def _round100(x: float) -> int:
    return int(round(x / 100.0)) * 100


def _target(lang: str) -> int:
    n = GOLD_N.get(lang, 0)
    if lang == "pcm":
        return 12000
    if lang in HIGH:
        return 6000
    if lang in MID3K or lang in MID2K:                      # mid: 2-3k base, rises with the number of corrected examples, capped at 4k
        base = 4000 if lang in MID3K else 2000
        return min(4000, max(base, _round100(800 * n)))
    if lang in LOW:                                          # low: 1k base, 150 per corrected example
        return max(1000, _round100(150 * n))
    return min(2000, max(300, _round100(150 * n)))           # very low: 150 per corrected example, capped at 2k


TARGETS = {l: _target(l) for l in HIGH + MID3K + MID2K + LOW + VERY_LOW + ["pcm"]}
TARGETS.update({l: 5000 for l in PIDGIN_VARIANTS})
LAST = ["pcm", "gpe", "wes"]                            # generated after the languages whose English it reuses (track D)
# Track D: the generated `eng` of these languages becomes the given English for the share of rows below
REUSE_FROM = {"pcm": ["yor", "hau", "ibo", "twi", "ewe", "zul"], "gpe": ["twi", "ewe", "swh", "hau", "yor", "ibo", "zul"], "wes": ["swh", "hau", "yor", "ibo", "zul", "xho", "som", "amh"]}
REUSE_SHARE = {"pcm": 0.70, "gpe": 0.70, "wes": 0.70, **{l: 0.30 for l in MID3K + MID2K}, **{l: 0.50 for l in LOW}, **{l: 0.80 for l in VERY_LOW}}
REUSE_DEFAULT_SOURCES = ["swh", "hau", "yor", "ibo", "zul", "xho", "som", "amh"]
assert set(TARGETS) <= set(P), set(TARGETS) - set(P)


# Native seeds are skipped for languages whose seed pool is short machine-translated sentence pairs (median seed < 40 words, 70-100% MT-ish): they use anchors and English seeds instead.
SKIP_NATIVE_SEEDS = {"mlg", "kon", "kmb", "umb"}
