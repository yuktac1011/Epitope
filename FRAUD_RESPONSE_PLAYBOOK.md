# FedShield Fraud Response Playbook

## 1. Purpose

FedShield helps participating banks detect suspicious transaction behavior without sharing raw transaction rows or customer account identifiers. It combines:

- A federated fraud model for transaction-level risk scoring.
- Laya for text-based fraud classification, urgency scoring, churn-risk detection, and routing.
- An audit surface that verifies what crosses the bank boundary.

This is a synthetic-data research prototype. It supports investigation and prioritization; it does not replace a bank's fraud, compliance, or legal decision process.

## 2. Operating Model

1. Each bank keeps transaction data, account IDs, and local train/test rows inside its own boundary.
2. Each bank trains or evaluates its local model.
3. Banks send bounded model updates and weighting metadata to the coordinator.
4. The coordinator validates the payload and aggregates updates with FedAvg.
5. The shared model is evaluated separately for each bank.
6. Laya triages text reports and routes low-confidence cases to human review.

The coordinator must never receive raw transaction rows, `nameOrig`, `nameDest`, or account identifiers.

## 3. Fraud Patterns Covered

### Transaction and model-risk layer

The federated model is trained on synthetic PaySim-style transaction data. It identifies suspicious versus legitimate behavior, including unusual transfers, cash-outs, payments, amount patterns, and activity that differs from a bank's local distribution.

### Laya triage layer

Laya can classify these report-level patterns:

- Duplicate charge
- Unauthorized transaction
- Account takeover
- Phishing or social engineering
- Money laundering and layering
- Card skimming or counterfeit-card activity
- Identity theft
- Synthetic identity fraud
- Business email compromise
- Mule-account activity
- Investment scam
- Legitimate activity

Laya also returns urgency, churn risk, confidence, and a review flag.

## 4. Analyst Workflow

### Step A: Confirm the alert

Capture the customer report, transaction type, amount, sender, receiver, and relevant balance change. Do not paste secrets, passwords, full card numbers, or unnecessary identity documents into the triage narrative.

### Step B: Run Laya triage

1. Open **Laya Triage**.
2. Select the closest scenario preset or write the report.
3. Enter transaction type, amount, sender, and receiver context.
4. Set the confidence threshold. A higher threshold sends more cases to review.
5. Run the decision model.

### Step C: Interpret the result

- High-confidence result above the threshold: follow the institution's approved automated action policy.
- Result below the threshold: route to a human analyst.
- Critical urgency: prioritize immediate containment and customer contact.
- High churn risk: coordinate fraud handling with customer support.
- Money laundering, mule, or identity patterns: route to the appropriate compliance or fraud-investigation team.

### Step D: Validate with the federated model

Use the Bank Results and Participation views to compare local-only and federated performance. A positive macro gain does not prove that every bank improved, so inspect each participant separately.

### Step E: Record evidence

Record the alert text, model category, confidence, urgency, review decision, analyst action, and final disposition. Preserve the audit result showing that raw rows and account IDs were not sent to the coordinator.

## 5. Routing Guide

| Pattern                                      | First response                                                                                            | Escalate to                              |
| -------------------------------------------- | --------------------------------------------------------------------------------------------------------- | ---------------------------------------- |
| Unauthorized transaction or account takeover | Freeze or protect the affected account according to policy; verify the customer through a trusted channel | Fraud investigation                      |
| Phishing or business email compromise        | Stop the requested transfer; preserve message and domain evidence                                         | Fraud investigation and security         |
| Duplicate charge                             | Verify authorization, merchant, and settlement state                                                      | Customer support or payments operations  |
| Card skimming                                | Block or replace the card; review nearby transactions                                                     | Card fraud team                          |
| Identity or synthetic identity               | Pause onboarding or suspicious activity; verify identity using approved controls                          | Fraud investigation and compliance       |
| Mule account or money laundering             | Preserve transaction chain and avoid tipping off the subject                                              | AML/compliance                           |
| Investment scam                              | Stop outbound payment where possible; provide customer protection guidance                                | Fraud investigation and customer support |
| Low-confidence or conflicting result         | Do not automate a final action                                                                            | Human analyst                            |

## 6. Simulation Demonstration

Use **Live Simulation** to explain the federated process:

1. Banks process local data one at a time.
2. Each bank produces a parameter update.
3. Only weights and metadata cross the boundary.
4. The coordinator validates the payload.
5. FedAvg creates a shared model.
6. The shared model is evaluated at each bank.
7. The event log and bank detail modal show the values behind each step.

The replay is an explanatory visualization. It does not start a new training job from the browser unless a future backend run endpoint is added.

## 7. Evidence and Metrics

Use these metrics in reports:

- **PR-AUC / Average Precision:** preferred for highly imbalanced fraud data.
- **Recall @ 1% FPR:** fraud captured while holding false positives to a fixed rate.
- **Local versus federated delta:** per-bank improvement or regression.
- **Communication cost:** parameters and bandwidth sent per round.
- **Boundary audit:** raw rows and account IDs shared should remain zero.

Do not present reused or unavailable metrics as independently measured results. The backend must export a metric before it is claimed in a report.

## 8. Limitations

- The benchmark data is synthetic and does not establish real-bank performance.
- Laya classifies the submitted narrative; it does not prove that fraud occurred.
- Model updates can still carry privacy risk through inference attacks.
- Confusion-matrix counts and some dataset metadata are unavailable unless the backend export includes them.
- Human review remains necessary for low-confidence, high-impact, or regulated cases.

## 9. Run Checklist

Before a demonstration:

- Start the FastAPI backend.
- Start the frontend.
- Confirm Laya health is ready.
- Confirm experiment CSV artifacts exist.
- Open Overview and verify backend status.
- Open Live Simulation and replay the four phases.
- Click a bank tile to show participant-level results.
- Open Laya Triage and run one high-risk and one legitimate scenario.
- Open Boundary Audit and confirm raw rows and account IDs are not shared.

## 10. Presentation Summary

> FedShield lets banks collaborate on fraud-model training without exchanging raw customer transactions. The federated model identifies suspicious transaction behavior, while Laya classifies reported fraud patterns and routes uncertain cases to human analysts. Every result is shown with per-bank impact and a data-boundary audit so the system's benefits and limitations remain visible.
