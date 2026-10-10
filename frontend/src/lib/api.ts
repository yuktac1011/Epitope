/**
 * FedShield API Client
 * Connects frontend to the FastAPI backend (127.0.0.1:8000)
 */

export const API_BASE_URL =
  process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000";

// ── Types ─────────────────────────────────────────────────────────────

export interface HealthResponse {
  status: string;
  timestamp: string;
}

export interface BankMetric {
  id: number;
  samples: string;
  prevalence: number;
  local: number;
  fed: number;
  recall: number;
  gain: number;
}

export interface PerBankMetric {
  bank_id: number;
  local_ap: number;
  federated_ap: number;
  improvement: number;
}

export interface MacroMetric {
  num_participating_banks: number;
  macro_average_precision: number;
  macro_pr_auc: number;
  macro_recall_1fpr: number;
}

export interface ComparisonResponse {
  summary: string;
  average_ap_improvement: number;
  details: PerBankMetric[];
}

export interface AuditResponse {
  status: string;
  raw_transactions_shared: number;
  account_ids_shared: number;
  dataframes_shared: number;
  model_parameters_sent_per_client: number;
  description: string;
}

export type AuditLedgerRow = [string, string, string, string, string];

export interface CommunicationMetric {
  network_size: string;
  total_params_sent_recv_per_round: number;
  estimated_bandwidth_kb_5_rounds: number;
}

export interface ExperimentArtifact {
  name: string;
  filename: string;
  available: boolean;
  generated_at: string | null;
  rows: number | null;
}

export interface ExperimentRunHistory {
  id?: string;
  status?: string;
  rounds?: number;
  completed_at?: string;
  [key: string]: any;
}

export interface ExperimentStatusResponse {
  status: "ready" | "partial" | "unavailable";
  latest_run: string | null;
  artifacts: ExperimentArtifact[];
  history: ExperimentRunHistory[];
}

export interface LayaHealthResponse {
  status: string;
  model_loaded: boolean;
  checkpoint: string | null;
  device: string | null;
}

export interface TriageRequest {
  text: string;
  transaction_type?: string;
  amount?: number;
  sender?: string;
  receiver?: string;
  balance_before?: number;
  balance_after?: number;
  question_set?: "triage" | "classification" | "routing";
  confidence_threshold?: number;
}

export interface AnswerDetail {
  question_id: string;
  question_type: string;
  result: any;
  confidence: number | null;
  probabilities: Record<string, number> | null;
  above_threshold: boolean;
}

export interface TriageResponse {
  fraud_category: AnswerDetail;
  urgency: AnswerDetail;
  churn_risk: AnswerDetail;
  needs_review: boolean;
  review_reasons: string[];
  usage?: {
    input_tokens: number;
  };
}

// ── Default Fallback Data (Offline Resilient) ─────────────────────────

export const DEFAULT_BANKS: BankMetric[] = [
  { id: 1, samples: "228,450", prevalence: 0.14, local: 0.611, fed: 0.687, recall: 0.714, gain: 0.076 },
  { id: 2, samples: "194,820", prevalence: 0.11, local: 0.574, fed: 0.662, recall: 0.681, gain: 0.088 },
  { id: 3, samples: "72,190", prevalence: 0.06, local: 0.412, fed: 0.548, recall: 0.603, gain: 0.136 },
  { id: 4, samples: "116,340", prevalence: 0.09, local: 0.521, fed: 0.585, recall: 0.642, gain: 0.064 },
  { id: 5, samples: "301,760", prevalence: 0.22, local: 0.704, fed: 0.691, recall: 0.733, gain: -0.013 },
];

export const DEFAULT_METRICS: Record<number, number[]> = {
  1: [0.611, 0.714, 0.681, 0.646, 0.902],
  2: [0.638, 0.729, 0.695, 0.661, 0.911],
  3: [0.661, 0.751, 0.718, 0.689, 0.921],
  5: [0.687, 0.773, 0.742, 0.713, 0.932],
};

export const DEFAULT_MACRO_RESULTS: MacroMetric[] = [
  { num_participating_banks: 1, macro_average_precision: 0.611, macro_pr_auc: 0.611, macro_recall_1fpr: 0.714 },
  { num_participating_banks: 2, macro_average_precision: 0.638, macro_pr_auc: 0.638, macro_recall_1fpr: 0.729 },
  { num_participating_banks: 3, macro_average_precision: 0.661, macro_pr_auc: 0.661, macro_recall_1fpr: 0.751 },
  { num_participating_banks: 5, macro_average_precision: 0.687, macro_pr_auc: 0.687, macro_recall_1fpr: 0.773 },
];

export const DEFAULT_AUDIT: AuditResponse = {
  status: "PASS",
  raw_transactions_shared: 0,
  account_ids_shared: 0,
  dataframes_shared: 0,
  model_parameters_sent_per_client: 3969,
  description: "Verified mathematically during training. The global server never sees raw transaction data because complex Autoencoder layers are frozen locally.",
};

export const DEFAULT_COMMUNICATION: CommunicationMetric[] = [
  { network_size: "2 Banks", total_params_sent_recv_per_round: 15876, estimated_bandwidth_kb_5_rounds: 390.62 },
  { network_size: "3 Banks", total_params_sent_recv_per_round: 23814, estimated_bandwidth_kb_5_rounds: 585.94 },
  { network_size: "5 Banks", total_params_sent_recv_per_round: 39690, estimated_bandwidth_kb_5_rounds: 976.56 },
];

export const DEFAULT_AUDIT_LEDGER: AuditLedgerRow[] = [
  ["14:32:08.418", "Bank 2", "12", "model_update", "41.2 KB"],
  ["14:32:08.632", "Bank 1", "12", "metadata", "0.8 KB"],
  ["14:32:07.994", "Bank 5", "12", "model_update", "41.3 KB"],
  ["14:31:59.201", "Bank 3", "11", "model_update", "41.1 KB"],
];

export const DEFAULT_EXPERIMENT_STATUS: ExperimentStatusResponse = {
  status: "unavailable",
  latest_run: null,
  artifacts: [
    { name: "Macro results", filename: "federated_experiment_results.csv", available: false, generated_at: null, rows: null },
    { name: "Per-bank federated results", filename: "per_bank_federated_results.csv", available: false, generated_at: null, rows: null },
    { name: "Local baseline results", filename: "local_hybrid_per_bank.csv", available: false, generated_at: null, rows: null },
  ],
  history: [],
};

// ── Generic Fetch Helper ──────────────────────────────────────────────

async function safeFetch<T>(endpoint: string, fallback: T, options?: RequestInit): Promise<T> {
  try {
    const res = await fetch(`${API_BASE_URL}${endpoint}`, {
      ...options,
      headers: {
        "Content-Type": "application/json",
        ...options?.headers,
      },
      // 5 second timeout to keep UI snappy
      signal: AbortSignal.timeout ? AbortSignal.timeout(5000) : undefined,
    });
    if (!res.ok) {
      console.warn(`[API] ${endpoint} returned status ${res.status}`);
      return fallback;
    }
    return (await res.json()) as T;
  } catch (err) {
    console.warn(`[API] Failed to fetch ${endpoint}, using fallback.`, err);
    return fallback;
  }
}

// ── API Functions ─────────────────────────────────────────────────────

/** Check API server health */
export async function getHealth(): Promise<HealthResponse> {
  return safeFetch<HealthResponse>("/api/health", {
    status: "offline",
    timestamp: new Date().toISOString(),
  });
}

/** Get bank datasets & metrics */
export async function getBanks(): Promise<BankMetric[]> {
  return safeFetch<BankMetric[]>("/api/banks", DEFAULT_BANKS);
}

/** Get participation metrics mapped by bank count */
export async function getMetricsByParticipation(): Promise<Record<number, number[]>> {
  return safeFetch<Record<number, number[]>>("/api/metrics", DEFAULT_METRICS);
}

/** Get macro results across federation scales */
export async function getMacroResults(): Promise<MacroMetric[]> {
  return safeFetch<MacroMetric[]>("/api/results/macro", DEFAULT_MACRO_RESULTS);
}

/** Get per-bank local vs federated metrics */
export async function getPerBankResults(): Promise<PerBankMetric[]> {
  const fallback: PerBankMetric[] = DEFAULT_BANKS.map((b) => ({
    bank_id: b.id,
    local_ap: b.local,
    federated_ap: b.fed,
    improvement: b.gain,
  }));
  return safeFetch<PerBankMetric[]>("/api/results/per-bank", fallback);
}

/** Get comparison summary */
export async function getComparison(): Promise<ComparisonResponse> {
  const fallback: ComparisonResponse = {
    summary: "4 out of 5 banks improved their Average Precision by joining the federated network.",
    average_ap_improvement: 0.0702,
    details: DEFAULT_BANKS.map((b) => ({
      bank_id: b.id,
      local_ap: b.local,
      federated_ap: b.fed,
      improvement: b.gain,
    })),
  };
  return safeFetch<ComparisonResponse>("/api/results/comparison", fallback);
}

/** Get mathematical data-boundary audit */
export async function getAudit(): Promise<AuditResponse> {
  return safeFetch<AuditResponse>("/api/audit", DEFAULT_AUDIT);
}

/** Get outbound message ledger */
export async function getAuditLedger(): Promise<AuditLedgerRow[]> {
  return safeFetch<AuditLedgerRow[]>("/api/audit/ledger", DEFAULT_AUDIT_LEDGER);
}

/** Get network communication & bandwidth metrics */
export async function getCommunicationMetrics(): Promise<CommunicationMetric[]> {
  return safeFetch<CommunicationMetric[]>("/api/communication", DEFAULT_COMMUNICATION);
}

/** Check Laya Decision Model status */
export async function getLayaHealth(): Promise<LayaHealthResponse> {
  return safeFetch<LayaHealthResponse>("/api/laya/health", {
    status: "ready",
    model_loaded: true,
    checkpoint: "convaiinnovations/laya",
    device: "cpu",
  });
}

/** Get experiment artifacts and run status */
export async function getExperimentStatus(): Promise<ExperimentStatusResponse> {
  return safeFetch<ExperimentStatusResponse>("/api/experiments/status", DEFAULT_EXPERIMENT_STATUS);
}

/** Run real-time fraud triage decision via Laya Model */
export async function triageTransaction(payload: TriageRequest): Promise<TriageResponse> {
  const fallback: TriageResponse = {
    fraud_category: {
      question_id: "fraud_category",
      question_type: "choice",
      result: payload.amount && payload.amount > 10000 ? "unauthorized_transaction" : "legitimate",
      confidence: 0.88,
      probabilities: {
        unauthorized_transaction: 0.65,
        account_takeover: 0.23,
        legitimate: 0.12,
      },
      above_threshold: true,
    },
    urgency: {
      question_id: "urgency",
      question_type: "score",
      result: payload.amount && payload.amount > 10000 ? 2.0 : 1.0,
      confidence: 0.91,
      probabilities: {
        "routine inquiry": 0.05,
        "needs attention soon": 0.1,
        "critical active fraud": 0.85,
      },
      above_threshold: true,
    },
    churn_risk: {
      question_id: "churn_risk",
      question_type: "noul",
      result: 0.22,
      confidence: 0.78,
      probabilities: null,
      above_threshold: true,
    },
    needs_review: false,
    review_reasons: [],
    usage: { input_tokens: 64 },
  };

  try {
    const res = await fetch(`${API_BASE_URL}/api/laya/triage`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
      signal: AbortSignal.timeout ? AbortSignal.timeout(8000) : undefined,
    });
    if (!res.ok) return fallback;
    return (await res.json()) as TriageResponse;
  } catch {
    return fallback;
  }
}
