"""
Fraud-Domain Question Sets for Laya

Per SKILL.md:
  - Questions ask what the text SAYS, not what to do about it
  - Every option has a description
  - Option lists are short (< 20)
  - State should be short and front-loaded
"""


def fraud_triage_questions() -> dict:
    """
    Standard triage question set for a transaction alert or customer message
    in the fraud-detection domain.

    Returns a dict of three questions that run in a single forward pass:
      - fraud_category (choice): What type of fraud pattern does the text describe?
      - urgency (score): How urgent is the situation described?
      - churn_risk (noul): Does the customer threaten to leave or cancel?
    """
    return {
        "fraud_category": {
            "type": "choice",
            "instructions": "What type of financial issue or fraud pattern is described in this text?",
            "criteria": {
                "duplicate_charge": "duplicate transactions, charged twice, double billing",
                "unauthorized_transaction": "transactions the account holder did not make, stolen card, unknown charges",
                "account_takeover": "someone else accessed the account, password changed, suspicious login",
                "phishing_scam": "fake emails, links asking for credentials, social engineering",
                "money_laundering": "structuring deposits, rapid transfers between accounts, layering funds",
                "legitimate": "normal transaction activity, no fraud indicators present",
            },
        },
        "urgency": {
            "type": "score",
            "instructions": "How urgent is the financial situation described in this text?",
            "criteria": [
                "routine inquiry, no time pressure",
                "needs attention soon but not immediately",
                "critical issue requiring immediate action, active fraud or significant financial loss",
            ],
        },
        "churn_risk": {
            "type": "noul",
            "instructions": "Does the text contain a threat to cancel, close the account, or leave the bank?",
            "criteria": {
                "true": "the person explicitly or implicitly threatens to leave or cancel",
                "false": "no indication of wanting to leave or cancel",
            },
        },
    }


def transaction_classification_questions() -> dict:
    """
    Classify a transaction description by type and risk level.
    Intended for raw transaction metadata converted to text.
    """
    return {
        "transaction_type": {
            "type": "choice",
            "instructions": "What type of financial transaction is described?",
            "criteria": {
                "transfer": "money sent from one account to another",
                "cash_out": "cash withdrawal or ATM transaction",
                "cash_in": "cash deposit into an account",
                "payment": "payment for goods or services",
                "debit": "direct debit or recurring charge",
            },
        },
        "risk_level": {
            "type": "score",
            "instructions": "How suspicious or risky does this transaction appear based on the description?",
            "criteria": [
                "normal activity, nothing unusual",
                "slightly unusual but possibly legitimate",
                "highly suspicious, strong indicators of fraud",
            ],
        },
    }


def alert_routing_questions() -> dict:
    """
    Route a fraud alert to the appropriate team.
    """
    return {
        "department": {
            "type": "choice",
            "instructions": "Which department should handle this alert?",
            "criteria": {
                "fraud_investigation": "confirmed or suspected fraud cases needing deep investigation",
                "customer_support": "customer complaints, billing disputes, general inquiries",
                "compliance": "regulatory concerns, AML flags, reporting requirements",
                "technical": "system errors, failed transactions, technical glitches",
            },
        },
        "needs_escalation": {
            "type": "noul",
            "instructions": "Does this alert describe a situation that requires immediate escalation to a senior analyst?",
            "criteria": {
                "true": "large monetary loss, multiple accounts affected, or active ongoing attack",
                "false": "routine alert that can be handled by standard process",
            },
        },
    }


def build_transaction_state(
    transaction_type: str = "",
    amount: float = 0.0,
    sender: str = "",
    receiver: str = "",
    description: str = "",
    balance_before: float = 0.0,
    balance_after: float = 0.0,
) -> dict:
    """
    Build a Laya-friendly state dict from structured transaction fields.

    Per SKILL.md: put what matters first, keep it short, put numbers
    into words where the decision depends on a comparison.
    """
    parts = {}

    if description:
        parts["description"] = description

    if transaction_type:
        parts["transaction_type"] = transaction_type

    if amount > 0:
        parts["amount"] = f"${amount:,.2f}"

    if sender:
        parts["from"] = sender
    if receiver:
        parts["to"] = receiver

    # Per SKILL.md: compute comparisons in code, hand Laya the conclusion
    if balance_before > 0 and balance_after >= 0:
        delta = balance_after - balance_before
        if abs(delta) < 0.01:
            parts["balance_change"] = "balance unchanged"
        elif delta < 0:
            parts["balance_change"] = f"balance decreased by ${abs(delta):,.2f}"
        else:
            parts["balance_change"] = f"balance increased by ${delta:,.2f}"

        ratio = balance_after / balance_before if balance_before > 0 else 0
        if ratio < 0.1:
            parts["balance_observation"] = "nearly all funds were removed from the account"
        elif ratio < 0.5:
            parts["balance_observation"] = "more than half the balance was removed"

    return parts
