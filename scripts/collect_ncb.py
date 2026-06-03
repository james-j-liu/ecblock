"""Collect NCB governor speeches/interviews discovered via web search.

URLs are gathered from targeted web searches (per governor, on the NCB domain) and
extracted with the ncb_scrape engine (direct fetch, Wayback fallback for blocked
sites). Saves to data/processed/ncb_scraped.jsonl for later merge into the corpus.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from ecblock.corpus.ncb_scrape import collect_urls
from ecblock.config import PROCESSED
from ecblock.schema import save_corpus

B = "https://www.bank.lv"; LT = "https://www.lb.lt"; HR = "https://www.hnb.hr"
MT = "https://www.centralbankmalta.org"; SI = "https://www.bsi.si"; SK = "https://nbs.sk"
EE = "https://www.eestipank.ee"
BE = "https://www.nbb.be"; GR = "https://www.bankofgreece.gr"
_grn = GR + "/en/news-and-media/press-office/news-list/news?announcement="
PT = "https://www.bportugal.pt"; AT = "https://www.oenb.at"; CY = "https://www.centralbank.cy"
BG = "https://www.bnb.bg"; MTb = "https://www.centralbankmalta.org"; ES = "https://www.bde.es"
NL = "https://www.dnb.nl"
_pti = PT + "/en/intervencoes/"
_cys = CY + "/en/the-governor/governor-s-speeches/"
_cyp = CY + "/en/the-governor/governor-speeches/"

# {speaker, institution, lang, [urls]}
GOV = [
  ("Mārtiņš Kazāks", "Latvijas Banka", "en", [
    B+"/en/news-and-events/news-and-articles/news/17052-opening-speech-by-martins-kazaks-at-the-economic-conference-2024",
    B+"/en/news-and-events/news-and-articles/news/17333-opening-speech-by-martins-kazaks-at-the-latvia-fintech-forum-2025",
    B+"/en/news-and-events/news-and-articles/news/16517-our-economy-weathered-crises-quite-well-but-risk-of-inequality-may-have-increased",
    B+"/en/news-and-events/news-and-articles/news/16694-opening-remarks-by-martins-kazaks",
    B+"/en/news-and-events/news-and-articles/news/17421-presentation-by-martins-kazaks-governor-of-latvijas-banka-at-the-mni-connect-investor-teleconference",
    B+"/en/news-and-events/news-and-articles/news/16590-ecb-s-kazaks-keeps-door-open-to-rate-hikes-if-needed",
  ]),
  ("Gediminas Šimkus", "Lietuvos bankas", "en", [
    LT+"/en/speeches-interviews-presentations/speech-by-gediminas-simkus-at-the-annual-economics-conference-pillars-of-resilience-amid-global-geopolitical-shifts",
    LT+"/en/news/speech-by-gediminas-simkus-at-the-international-conference-future-of-central-banking",
    LT+"/en/speeches-interviews-presentations/speech-by-gediminas-simkus-at-baltic-aml-forum-2021",
  ]),
  ("Boris Vujčić", "Hrvatska narodna banka", "en", [
    HR+"/en/-/governor-vujcic-delivers-speech-at-the-conference-one-year-with-the-euro",
    HR+"/en/-/governor-vujcic-for-reuters",
    HR+"/en/-/risk-ecb-undershoots-inflation-not-high-vujcic",
    HR+"/en/-/ecb-s-vujcic-cutting-faster-would-need-a-more-significant-departure-from-our-projections",
    HR+"/en/-/ecb-needs-patience-before-declaring-inflation-win-vujcic-says",
  ]),
  ("Boštjan Vasle", "Banka Slovenije", "en", [
    SI+"/en/media/1510/monetary-policy-meeting-of-the-governing-council-of-the-ecb-statement-by-governor-bostjan-vasle",
    SI+"/en/media/posts/opening-remarks-by-bostjan-vasle-governor-of-banka-slovenije-at-the-conference-towards-the-green-transition-investment-and-prices",
    SI+"/en/speeches-and-interviews/1736/address-by-bostjan-vasle-governor-of-banka-slovenije-at-the-celebration-of-banka-slovenijes-30th-anniversary",
    SI+"/en/media/posts/article-by-governor-bostjan-vasle-in-eurofi-magazine",
  ]),
  ("Primož Dolenc", "Banka Slovenije", "en", [
    SI+"/en/media/posts/statement-by-acting-governor-primoz-dolenc-following-the-ecb-s-monetary-policy-meeting",
    SI+"/en/media/posts/opening-address-by-the-acting-governor-primoz-dolenc-at-the-lecture-ever-changing-payment-landscape-what-will-a-digital-euro-bring",
  ]),
  ("Peter Kažimír", "Národná banka Slovenska", "en", [
    SK+"/en/news/governor-peter-kazimir-a-watched-pot-never-boils/",
    SK+"/en/news/governor-peter-kazimir-slow-and-steady-wins-the-race/",
    SK+"/en/news/strategic-vigilance-in-volatile-times/",
    SK+"/en/news/governor-peter-kazimir-the-stage-is-set/",
    SK+"/en/news/governor-peter-kazimir-the-long-expected-pause-button-pushed/",
    SK+"/en/news/keep-calm-summers-almost-here/",
  ]),
  ("Madis Müller", "Eesti Pank", "en", [
    EE+"/en/press/madis-muller-why-inflation-so-high-and-so-different-different-euro-area-countries-03032023",
    EE+"/en/press/estonias-strength-during-crisis-has-come-integration-and-cooperation-says-madis-muller-19042021",
    EE+"/en/press/high-inflation-remains-concern-estonia-says-madis-muller-18092025",
  ]),
  ("Pierre Wunsch", "Nationale Bank van België / Banque Nationale de Belgique", "en", [
    BE+"/en/news-events/interview-pierre-wunsch-financial-times-ecb-may-have-cut-interest-rates-below-2-former",
    BE+"/en/news-events/interview-cnbc-pierre-wunsch-trumps-tariffs-are-making-ecbs-interest-rate-path-more",
  ]),
  ("Yannis Stournaras", "Bank of Greece", "en", [
    _grn+"4510cfe8-69dd-46b0-938f-622657711dd7",
    _grn+"05be290a-c8d9-4dc1-b331-8f45a060786a",
    _grn+"da80126b-1327-4ef3-8e69-2920d2a15f00",
    _grn+"ab69d668-ae31-4bb1-8c8d-7d23159f09be",
    _grn+"ba0a2914-8bbe-4782-9f09-459a720e2f72",
    _grn+"9ee1f7c6-4bb5-4f14-b9f0-4ce07d0d324d",
  ]),
  ("Mário Centeno", "Banco de Portugal", "en", [
    _pti+"opening-remarks-governor-mario-centeno-conference-financial-stability",
    _pti+"interview-governor-mario-centeno-econostream-4",
    _pti+"interview-governor-mario-centeno-politico-1",
    _pti+"interview-reuters-mario-centeno-governor-banco-de-portugal",
    _pti+"interview-governor-mario-centeno-reuters-3",
    _pti+"interview-governor-mario-centeno-bloomberg-9",
  ]),
  ("Robert Holzmann", "Oesterreichische Nationalbank", "en", [
    AT+"/en/Media/Press-Archives/2025/20250325.html",
    AT+"/en/Media/Press-Archives/2024/20240322.html",
    AT+"/en/Media/Press-Archives/2023/20230323.html",
    AT+"/en/Media/Press-Archives/2023/20230517.html",
    AT+"/en/Media/Press-Archives/2022/20220330.html",
  ]),
  ("Constantinos Herodotou", "Central Bank of Cyprus", "en", [
    _cys+"18-05-2023", _cys+"06-10-2023", _cys+"04-07-2023",
    _cys+"speech-by-constantinos-herodotou-governor-of-the-central-bank-of-cyprus-during-ms-lagarde-visit-to-cyprus-30-03-2022",
  ]),
  ("Christodoulos Patsalides", "Central Bank of Cyprus", "en", [
    _cyp+"16-01-2025",
    _cyp+"gov-speech-the-economy-of-cyprus-developments-and-outlook-07-04-2025",
  ]),
  ("Dimitar Radev", "Bulgarian National Bank", "en", [
    BG+"/AboutUs/PressOffice/POStatements/POATheme/02_RADEV_20230724_EN",
  ]),
  ("Edward Scicluna", "Central Bank of Malta", "en", [
    MTb+"/finance-malta-conference-2021", MTb+"/family-affairs-committee",
    MTb+"/gov-interview-businessnow", MTb+"/gov-scicluna-business-now-sep22",
    MTb+"/distinguished-speakers-luncheon-seminar-london",
    MTb+"/mcast-visit-2023", MTb+"/gov-interview-cnbc-2023",
  ]),
  ("José Luis Escrivá", "Banco de España", "en", [
    ES+"/wbe/en/inicio/intervenciones-home/gobernador-the-ecb-and-its-watchers-xxv-.html",
  ]),
  # --- batch 2 ---
  ("Mārtiņš Kazāks", "Latvijas Banka", "en", [
    B+"/en/news-and-events/news-and-articles/news/16702-ecb-martins-kazaks-happy-with-rates-at-current-levels",
    B+"/en/news-and-events/news-and-articles/news/17496-monetary-policy-remains-in-a-good-place",
  ]),
  ("Gediminas Šimkus", "Lietuvos bankas", "en", [
    LT+"/en/news/lithuania-s-economic-situation-is-one-of-the-best-in-europe-but-labour-shortage-and-inflation-challenges-arise",
  ]),
  ("Olaf Sleijpen", "De Nederlandsche Bank", "en", [
    NL+"/en/general-news/speech-2023/speech-olaf-sleijpen-investing-in-the-future/",
    NL+"/en/general-news/speech-2023/introductory-line-for-the-standing-parliamentary-committee-for-finance/",
  ]),
  ("Klaas Knot", "De Nederlandsche Bank", "en", [
    NL+"/en/general-news/speeches-2022/speech-klaas-knot-a-time-to-act-the-outlook-for-monetary-policy-in-the-euro-area",
  ]),
]

OUT = PROCESSED / "ncb_scraped.jsonl"


def main():
    all_sp = []
    for speaker, inst, lang, urls in GOV:
        sp = collect_urls(urls, speaker, inst, lang=lang)
        print(f"{speaker:22} {len(sp):2}/{len(urls)} extracted  ({inst})")
        all_sp.extend(sp)
    save_corpus(all_sp, OUT)
    print(f"\nTotal: {len(all_sp)} records -> {OUT}")


if __name__ == "__main__":
    main()
