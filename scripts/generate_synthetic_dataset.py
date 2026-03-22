"""Generate a balanced synthetic adverse media dataset for KYC training."""

from __future__ import annotations

import csv
import random
from pathlib import Path

OUTPUT_PATH = Path("data/synthetic_adverse_media.csv")
SEED = 42
TOTAL_SAMPLES = 2000
SAMPLES_PER_LABEL = TOTAL_SAMPLES // 2

ENTITY_PREFIXES = [
    "Aster", "BlueRiver", "Crown", "Delta", "Everstone", "Falcon", "Granite", "Harbor",
    "Ironwood", "Juniper", "Keystone", "Lighthouse", "Monarch", "Northstar", "Oakbridge",
    "Pioneer", "Quartz", "Redwood", "Summit", "Trident", "Umber", "Vale", "Westbridge", "Zenith",
]
ENTITY_SUFFIXES = [
    "Capital", "Holdings", "Global", "Partners", "Infrastructure", "Logistics", "Energy",
    "Commodities", "Payments", "Advisory", "Resources", "Ventures", "Telecom", "Shipping",
    "Trading", "Finance", "Bank", "Securities", "Digital", "Investments",
]
COUNTRIES = [
    "Singapore", "Malaysia", "Indonesia", "the UAE", "the UK", "France", "Germany",
    "Nigeria", "South Africa", "Brazil", "Mexico", "India", "Australia", "Canada",
]
REGULATORS = [
    "the financial intelligence unit", "the anti-corruption bureau", "the central bank",
    "the securities regulator", "the public prosecutor", "the competition authority",
    "the serious fraud office", "the ministry of justice",
]
SOURCES = [
    "regional newspapers", "investigative reporters", "wire services", "industry blogs",
    "court filings", "regulatory statements", "parliamentary testimony", "forensic audit summaries",
]
PENALTIES = [
    "$18 million", "$42 million", "$95 million", "$210 million", "$480 million", "$2.9 billion",
]
RUMOR_CHANNELS = [
    "social media posts", "anonymous Telegram channels", "microblog threads", "online rumor accounts",
]
WEAK_SIGNALS = [
    "compliance officers said the allegations were still being assessed",
    "the company denied wrongdoing and called the claims incomplete",
    "lawyers for the group argued that some details were taken out of context",
    "management said it had opened an internal review after the first reports emerged",
]
STRONG_SIGNALS = [
    "prosecutors later filed charges after tracing payments through shell companies",
    "regulators concluded that controls had failed and imposed a financial penalty",
    "the court record described fabricated invoices, concealed beneficiaries, and false accounting entries",
    "the final settlement included admissions around anti-money-laundering control failures",
]
NEGATIVE_OUTCOMES = [
    "The case pushed the client into enhanced due diligence and a high-risk review queue.",
    "KYC teams would treat the pattern as material adverse media because the narrative escalated from suspicion to enforcement.",
    "For onboarding teams, the progression from rumor to formal action would justify a material-risk flag.",
]
POSITIVE_OUTCOMES = [
    "KYC teams would still document the media trail, but the file would usually remain non-material while monitoring continues.",
    "The overall assessment would likely stay non-material because the strongest claims were not substantiated by enforcement action.",
    "Compliance teams would note the weak signal but keep the relationship in routine monitoring rather than escalate it to high risk.",
]

NEGATIVE_PATTERNS = [
    "tender_irregularities",
    "social_media_to_enforcement",
    "confirmed_aml_penalty",
    "executive_fraud_action",
    "embezzlement_progression",
]
POSITIVE_PATTERNS = [
    "rumor_denied",
    "investigation_without_findings",
    "historic_issue_remediated",
    "executive_cleared",
    "vendor_dispute_not_criminal",
]


def entity_name(index: int) -> str:
    prefix = ENTITY_PREFIXES[index % len(ENTITY_PREFIXES)]
    suffix = ENTITY_SUFFIXES[(index // len(ENTITY_PREFIXES)) % len(ENTITY_SUFFIXES)]
    return f"{prefix} {suffix}"


def sentence_join(parts: list[str]) -> str:
    midpoint = len(parts) // 2
    return " ".join(parts[:midpoint]) + "\n\n" + " ".join(parts[midpoint:])


def negative_story(index: int, rng: random.Random) -> str:
    entity = entity_name(index)
    country = rng.choice(COUNTRIES)
    regulator = rng.choice(REGULATORS)
    source = rng.choice(SOURCES)
    penalty = rng.choice(PENALTIES)
    weak_signal = rng.choice(WEAK_SIGNALS)
    strong_signal = rng.choice(STRONG_SIGNALS)
    outcome = rng.choice(NEGATIVE_OUTCOMES)
    pattern = NEGATIVE_PATTERNS[index % len(NEGATIVE_PATTERNS)]

    if pattern == "tender_irregularities":
        parts = [
            f"{entity}, a contractor active in {country}, appeared in {source} after reporters questioned a cluster of public tender awards.",
            "Initial articles described unusually narrow bidding windows and repeated subcontracting to politically connected intermediaries.",
            f"At first, the matter looked like a procurement governance issue and {weak_signal}.",
            f"Follow-up coverage said investigators were examining whether facilitation payments were routed through consultants tied to {entity}.",
            f"{strong_signal}.",
            f"{regulator} eventually announced sanctions and a {penalty} resolution linked to bribery controls.",
            outcome,
        ]
    elif pattern == "social_media_to_enforcement":
        rumor_channel = rng.choice(RUMOR_CHANNELS)
        parts = [
            f"Discussion about {entity} began with {rumor_channel} claiming that accounts linked to the group were moving funds for politically exposed persons in {country}.",
            "Early posts mixed speculation with screenshots of internal spreadsheets, so the first signal was noisy and partly unverified.",
            f"Traditional media later picked up the story, and {weak_signal}.",
            f"As the narrative developed, journalists connected the entity to nominee shareholders, cash-intensive distributors, and unexplained payment corridors.",
            f"{strong_signal}.",
            f"The eventual enforcement package cited sanctions screening gaps and anti-money-laundering failures, leading to a {penalty} penalty.",
            outcome,
        ]
    elif pattern == "confirmed_aml_penalty":
        parts = [
            f"{entity} was mentioned in several compliance briefings after correspondent banks in {country} noticed repeated gaps in suspicious transaction reporting.",
            "The earliest coverage said the institution was under pressure but had not yet been accused of deliberate misconduct.",
            f"That ambiguity mattered because {weak_signal}.",
            "Subsequent reporting outlined how high-risk customers were onboarded through weak beneficial-ownership checks and poorly monitored nested accounts.",
            f"{strong_signal}.",
            f"Authorities finalized the case with a deferred prosecution style settlement and a {penalty} monetary sanction.",
            outcome,
        ]
    elif pattern == "executive_fraud_action":
        parts = [
            f"Coverage of {entity} intensified after a senior executive in {country} was accused of disguising side agreements with distributors and lenders.",
            "The first wave of stories described accounting anomalies and abrupt resignations rather than a completed fraud case.",
            f"Internal sources insisted the board was reviewing the matter, and {weak_signal}.",
            "Later articles documented whistleblower emails, forged approvals, and off-book liabilities that had been hidden from investors.",
            f"{strong_signal}.",
            f"Regulators and prosecutors coordinated their response, culminating in director bans and a {penalty} settlement.",
            outcome,
        ]
    else:
        parts = [
            f"{entity} surfaced in long-form coverage comparing the case to large-scale embezzlement scandals after money raised for projects in {country} could not be fully reconciled.",
            "Early reporting only hinted at governance failures, related-party transactions, and unusually expensive advisory mandates.",
            f"For several weeks the case sat in a gray zone because {weak_signal}.",
            "More complete timelines later showed round-tripped transfers, luxury asset purchases, and opaque offshore entities attached to the same funding stream.",
            f"{strong_signal}.",
            f"The final wave of coverage described cross-border settlements, asset seizures, and a {penalty} global resolution.",
            outcome,
        ]

    while len(parts) < 5:
        parts.append(outcome)
    return sentence_join(parts)


def positive_story(index: int, rng: random.Random) -> str:
    entity = entity_name(index + 1000)
    country = rng.choice(COUNTRIES)
    regulator = rng.choice(REGULATORS)
    source = rng.choice(SOURCES)
    weak_signal = rng.choice(WEAK_SIGNALS)
    outcome = rng.choice(POSITIVE_OUTCOMES)
    pattern = POSITIVE_PATTERNS[index % len(POSITIVE_PATTERNS)]

    if pattern == "rumor_denied":
        rumor_channel = rng.choice(RUMOR_CHANNELS)
        parts = [
            f"{entity} appeared in {rumor_channel} after anonymous accounts alleged bribery around infrastructure permits in {country}.",
            "Those posts circulated quickly and created an adverse-media hit because they referenced unnamed insiders and blurred copies of invoices.",
            f"Mainstream coverage repeated the rumor but also noted that no agency had opened a public enforcement action, and {weak_signal}.",
            "Independent follow-up reporting found that the invoices matched ordinary customs fees rather than illicit transfers.",
            f"{regulator} later said it had not identified evidence supporting criminal allegations against the company.",
            outcome,
        ]
    elif pattern == "investigation_without_findings":
        parts = [
            f"{entity} was mentioned by {source} when a routine review in {country} examined procurement files involving several vendors.",
            "Initial stories created concern because investigators requested records from multiple counterparties and asked questions about political links.",
            f"Even so, the reports stressed that the review was preliminary, and {weak_signal}.",
            "After several months, the public summary said documentation gaps had been corrected but intentional bribery or fraud was not established.",
            f"{regulator} recommended process improvements, training, and stronger vendor screening rather than penalties.",
            outcome,
        ]
    elif pattern == "historic_issue_remediated":
        parts = [
            f"Articles about {entity} revisited an older anti-money-laundering concern in {country} after journalists compared the firm to peers facing sanctions issues.",
            "The first references looked adverse because they repeated legacy shortcomings in transaction monitoring and onboarding controls.",
            f"At the same time, the coverage acknowledged that {weak_signal}.",
            "More detailed reporting showed the customer population had been remediated, independent testing had improved, and supervisors had closed the main findings.",
            f"No new monetary enforcement action followed, and {regulator} described the residual risk as manageable with continued monitoring.",
            outcome,
        ]
    elif pattern == "executive_cleared":
        parts = [
            f"{entity} entered the media cycle when a former executive in {country} was accused online of embezzlement and self-dealing.",
            "The allegations sounded serious because they referenced offshore vehicles, procurement contracts, and abrupt internal departures.",
            f"Early briefings emphasized uncertainty, and {weak_signal}.",
            "Later court and audit updates showed that the disputed transfers were approved retention payments and documented loan settlements.",
            f"The individual still faced reputational scrutiny, but authorities closed the criminal angle without charges against {entity}.",
            outcome,
        ]
    else:
        parts = [
            f"{entity} was included in {source} discussing a dispute with a state-linked vendor in {country} over inflated contract pricing.",
            "Because the story referenced commissions, subcontractors, and politically connected brokers, it initially looked like a corruption narrative.",
            f"The reporting also said the issue was tied to a civil billing disagreement, and {weak_signal}.",
            "Subsequent filings framed the matter as a commercial dispute over milestones and quality claims rather than bribery or money laundering.",
            f"{regulator} did not announce penalties, and later coverage focused on governance lessons instead of proven criminal conduct.",
            outcome,
        ]

    while len(parts) < 5:
        parts.append(outcome)
    return sentence_join(parts)


def main() -> None:
    rng = random.Random(SEED)
    rows: list[dict[str, object]] = []

    for index in range(SAMPLES_PER_LABEL):
        rows.append({"text": negative_story(index, rng), "label": 1})
        rows.append({"text": positive_story(index, rng), "label": 0})

    rng.shuffle(rows)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["text", "label"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Generated {len(rows)} rows at {OUTPUT_PATH}")


if __name__ == "__main__":
    main()