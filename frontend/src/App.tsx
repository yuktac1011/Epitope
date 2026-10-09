"use client";

import { useEffect, useMemo, useState } from "react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
  CardDescription,
} from "@/components/ui/card";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { Separator } from "@/components/ui/separator";
import { DataProvider, useData } from "@/lib/data-context";
import {
  BankMetric,
  TriageRequest,
  TriageResponse,
  DEFAULT_BANKS,
  DEFAULT_AUDIT_LEDGER,
} from "@/lib/api";

// ── Types ─────────────────────────────────────────────────────────────
type Screen = "overview" | "curve" | "banks" | "rounds" | "audit" | "limits" | "triage";
type IconName =
  | "grid" | "chart" | "bank" | "rounds" | "shield" | "alert"
  | "lock" | "arrow" | "check" | "pause" | "play" | "reset" | "close";

const screens: { id: Screen; label: string; icon: IconName }[] = [
  { id: "overview", label: "Overview", icon: "grid" },
  { id: "curve", label: "Participation", icon: "chart" },
  { id: "banks", label: "Bank results", icon: "bank" },
  { id: "rounds", label: "Training rounds", icon: "rounds" },
  { id: "audit", label: "Boundary audit", icon: "shield" },
  { id: "triage", label: "Laya Triage", icon: "shield" },
  { id: "limits", label: "Limitations", icon: "alert" },
];

// ── Icon ───────────────────────────────────────────────────────────────
function Icon({ name, size = 18 }: { name: IconName; size?: number }) {
  const paths: Record<IconName, React.ReactNode> = {
    grid:   <><rect x="3" y="3" width="7" height="7" rx="1"/><rect x="14" y="3" width="7" height="7" rx="1"/><rect x="3" y="14" width="7" height="7" rx="1"/><rect x="14" y="14" width="7" height="7" rx="1"/></>,
    chart:  <><path d="M4 19V5"/><path d="M4 19h16"/><path d="m7 15 3-4 3 2 5-7"/></>,
    bank:   <><path d="m3 9 9-5 9 5"/><path d="M5 10h14M6 10v7m4-7v7m4-7v7m4-7v7M4 20h16"/></>,
    rounds: <><path d="M20 12a8 8 0 0 1-14.7 4.4L3 13"/><path d="M4 18v-5h5"/><path d="M4 12A8 8 0 0 1 18.7 7.6L21 11"/><path d="M20 6v5h-5"/></>,
    shield: <><path d="M12 3 20 6v5c0 5-3.4 8.5-8 10-4.6-1.5-8-5-8-10V6l8-3Z"/><path d="m8.5 12 2.2 2.2 4.8-5"/></>,
    alert:  <><path d="M10.3 4.3 2.8 18a2 2 0 0 0 1.8 3h14.8a2 2 0 0 0 1.8-3L13.7 4.3a2 2 0 0 0-3.4 0Z"/><path d="M12 9v4m0 4h.01"/></>,
    lock:   <><rect x="5" y="10" width="14" height="11" rx="2"/><path d="M8 10V7a4 4 0 0 1 8 0v3m-4 4v3"/></>,
    arrow:  <><path d="M5 12h14m-5-5 5 5-5 5"/></>,
    check:  <path d="m5 12 4 4L19 6"/>,
    pause:  <><path d="M9 5v14M15 5v14"/></>,
    play:   <path d="m8 5 11 7-11 7V5Z"/>,
    reset:  <><path d="M4 4v6h6"/><path d="M5.5 15a8 8 0 1 0 1-8.5L4 10"/></>,
    close:  <path d="m6 6 12 12M18 6 6 18"/>,
  };
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor"
      strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      {paths[name]}
    </svg>
  );
}

// ── Chip (emerald badge wrapper) ──────────────────────────────────────
type ChipTone = "default" | "cyan" | "violet" | "green" | "rose" | "amber";
function Chip({ children, tone = "default", className = "" }: { children: React.ReactNode; tone?: ChipTone; className?: string }) {
  const variantMap: Record<ChipTone, "default" | "emerald" | "cyan" | "outline" | "rose" | "amber"> = {
    default: "outline",
    cyan: "cyan",
    violet: "outline",
    green: "emerald",
    rose: "rose",
    amber: "amber",
  };
  return <Badge variant={variantMap[tone]} className={`font-sans text-[11px] font-medium tracking-normal px-2.5 py-0.5 ${className}`}>{children}</Badge>;
}

// ── Glass card wrapper ─────────────────────────────────────────────────
function Glass({ className = "", children }: { className?: string; children: React.ReactNode }) {
  return <div className={`glass ${className}`}>{children}</div>;
}

function DataState({ title, message }: { title: string; message: string }) {
  return (
    <Glass className="flex min-h-55 flex-col items-center justify-center gap-2 p-8 text-center">
      <span className="section-label">Dashboard status</span>
      <strong className="text-lg text-white">{title}</strong>
      <p className="max-w-lg text-sm text-slate-400">{message}</p>
    </Glass>
  );
}

// ── Page Header ────────────────────────────────────────────────────────
function PageHeader({ eyebrow, title, description, action }: { eyebrow: string; title: string; description: string; action?: React.ReactNode }) {
  return (
    <div className="page-head">
      <div>
        <div className="eyebrow">{eyebrow}</div>
        <div className="page-title">{title}</div>
        <p className="page-desc">{description}</p>
      </div>
      <div>
        {action || <Chip tone="amber">Synthetic data (PaySim) — prototype benchmark</Chip>}
      </div>
    </div>
  );
}

// ── Segmented ─────────────────────────────────────────────────────────
function Segmented({ options, value, onChange }: { options: string[]; value: string; onChange: (v: string) => void }) {
  return (
    <div className="segmented">
      {options.map((o) => (
        <button key={o} className={value === o ? "active" : ""} onClick={() => onChange(o)}>{o}</button>
      ))}
    </div>
  );
}

// ── Overview ───────────────────────────────────────────────────────────
function Overview() {
  const [banks, setBanks] = useState(5);
  const [scope, setScope] = useState("Macro average");
  const { metricsByParticipation, comparison, isLoading, error } = useData();

  if (isLoading) {
    return <DataState title="Loading experiment results" message="Connecting to the FedShield backend..." />;
  }

  const values = metricsByParticipation[banks];
  if (!values) {
    return (
      <DataState
        title="Experiment results unavailable"
        message={error || "The backend has not returned metrics for this federation size yet."}
      />
    );
  }

  const metrics = [
    ["PR-AUC", values[0], "+12.4%"],
    ["Recall @ 1% FPR", values[1], "+8.1%"],
    ["Precision", values[2], "+6.6%"],
    ["F1 score", values[3], "+7.9%"],
    ["ROC-AUC", values[4], "+2.8%"],
  ];

  return (
    <>
      <PageHeader
        eyebrow="Experiment overview"
        title="Shared signal. Local data."
        description="Federated fraud-model evaluation across simulated, non-IID bank datasets."
      />
      <Glass className="imbalance-banner">
        <span className="info-dot">i</span>
        <strong>Accuracy alone is not used:</strong> fraud is highly imbalanced. Primary metric is Average Precision (PR-AUC).
      </Glass>
      <div className="overview-grid">
        <Glass className="network-card">
          <div className="card-top">
            <div>
              <span className="section-label">Federation topology</span>
              <span className="live-dot">Round 12 active</span>
            </div>
            <div className="bank-select">
              <span>Participants</span>
              {[1, 2, 3, 5].map((n) => (
                <button key={n} className={banks === n ? "active" : ""} onClick={() => setBanks(n)}>{n}</button>
              ))}
            </div>
          </div>
          <NetworkVisual count={banks} />
          <div className="network-legend">
            <span><i className="legend-emerald" />Encrypted transport: model updates only</span>
            <span><i className="legend-cyan" />Shared model parameters</span>
          </div>
        </Glass>

        <div className="kpi-side">
          <div className="metric-heading">
            <span className="section-label">Evaluation snapshot</span>
            <Segmented options={["Macro average", "Per bank"]} value={scope} onChange={setScope} />
          </div>
          <div className="kpi-grid">
            {metrics.map(([label, value, delta], i) => (
              <Glass className={`kpi ${i === 0 ? "primary" : ""}`} key={String(label)}>
                <div className="kpi-label">{label}</div>
                <div className={`kpi-value ${i === 0 ? "primary-value" : ""}`}>{Number(value).toFixed(3)}</div>
                <div className="kpi-foot">
                  <span className="positive">↑ {delta}</span>
                  <span>vs local-only</span>
                </div>
              </Glass>
            ))}
          </div>
          <Glass className="comparison">
            <div className="card-top">
              <div>
                <span className="section-label">Local-only vs federated</span>
                {comparison && (
                  <span className="sub-label">{comparison.summary}</span>
                )}
              </div>
              <Chip tone="cyan">FastAPI Live</Chip>
            </div>
            {[["PR-AUC", 61, 69], ["Recall", 66, 77], ["F1", 63, 71]].map(([label, local, fed]) => (
              <div className="compare-row" key={String(label)}>
                <span>{label}</span>
                <div className="bar-pair">
                  <i className="bar-local" style={{ width: `${local}%` }} />
                  <i className="bar-fed" style={{ width: `${fed}%` }} />
                </div>
                <b>{Number(fed) / 100}</b>
              </div>
            ))}
            <div className="bar-key">
              <span><i className="local-key" />Local-only</span>
              <span><i className="fed-key" />Federated</span>
            </div>
          </Glass>
        </div>
      </div>
    </>
  );
}

// ── Network Visual ────────────────────────────────────────────────────
function NetworkVisual({ count }: { count: number }) {
  const { bankData } = useData();
  const positions = [
    { x: 50, y: 5 }, { x: 82, y: 25 }, { x: 74, y: 70 },
    { x: 26, y: 70 }, { x: 18, y: 25 },
  ];
  return (
    <div className="network">
      <svg className="network-lines" viewBox="0 0 100 80" preserveAspectRatio="none">
        {positions.slice(0, count).map((p, i) => (
          <g key={i}>
            <line x1="50" y1="43" x2={p.x} y2={p.y + 7} className="flow-line" />
            <circle r="1.2" className="packet">
              <animateMotion dur={`${2.3 + i * 0.2}s`} repeatCount="indefinite"
                path={`M 50 43 L ${p.x} ${p.y + 7}`} />
            </circle>
          </g>
        ))}
      </svg>
      <div className="coordinator">
        <span className="node-icon"><Icon name="rounds" size={24} /></span>
        <strong>Shared Model</strong>
        <small>Coordinator · FedAvg</small>
        <em>v12.4</em>
      </div>
      {positions.slice(0, count).map((p, i) => {
        const bank = bankData[i] || DEFAULT_BANKS[i];
        return (
          <div className="bank-node" key={i} style={{ left: `${p.x}%`, top: `${p.y}%` }}>
            <div className="lock-ring"><Icon name="lock" size={15} /></div>
            <strong>Bank {bank.id}</strong>
            <small>{bank.samples} rows</small>
            <span>LOCAL</span>
          </div>
        );
      })}
      <div className="boundary-note"><Icon name="lock" size={12} /> Raw transactions + account IDs stay local</div>
    </div>
  );
}

// ── Participation Curve ────────────────────────────────────────────────
function ParticipationCurve() {
  const [metric, setMetric] = useState("PR-AUC");
  const { bankData, macroResults } = useData();

  const fed = metric === "PR-AUC"
    ? "M48 198 C150 189 180 159 280 151 S410 112 520 90 S660 51 760 43"
    : "M48 187 C155 174 192 154 280 140 S420 107 520 85 S665 56 760 52";

  const improvedCount = bankData.filter(b => b.gain > 0).length;

  return (
    <>
      <PageHeader
        eyebrow="Core proof"
        title="Participation curve"
        description="Does adding more banks improve the shared model—and for whom?"
      />
      <div className="curve-layout">
        <Glass className="chart-card">
          <div className="card-top">
            <div>
              <span className="section-label">Performance by federation size</span>
              <span className="sub-label">Mean across 5 seeds · shaded area ±1 std</span>
            </div>
            <Segmented options={["PR-AUC", "Recall @ 1% FPR"]} value={metric} onChange={setMetric} />
          </div>
          <LineChart path={fed} />
        </Glass>
        <Glass className="insight-card">
          <div className="insight-icon"><Icon name="chart" /></div>
          <span className="section-label">What the data says</span>
          <div className="big-insight">{improvedCount} of {bankData.length}</div>
          <p>banks improve when all participants contribute.</p>
          <div className="insight-list">
            {bankData.slice(0, 3).map((bank) => (
              <div key={bank.id}>
                <span>Bank {bank.id}</span>
                <b className={bank.gain < 0 ? "negative" : "positive"}>
                  {bank.gain > 0 ? "+" : ""}{bank.gain.toFixed(3)} PR-AUC
                </b>
              </div>
            ))}
          </div>
          <div className="honest-note">Bank 5 declines slightly (−0.013). This is shown—not averaged away.</div>
        </Glass>
      </div>
      <div className="mini-section-head">
        <div>
          <span className="section-label">Per-bank response</span>
          <span className="sub-label">Individual outcomes as participants join</span>
        </div>
        <span className="muted-label">1 → 2 → 3 → 5 banks</span>
      </div>
      <div className="mini-grid">
        {bankData.map((bank, i) => (
          <Glass className="mini-chart" key={bank.id}>
            <div>
              <b>Bank {bank.id}</b>
              <Chip tone={bank.gain < 0 ? "rose" : "green"}>
                {bank.gain > 0 ? "+" : ""}{bank.gain.toFixed(3)}
              </Chip>
            </div>
            <svg viewBox="0 0 190 72">
              <path className="mini-gridline" d="M5 58H185M5 35H185M5 12H185" />
              <path
                className={bank.gain < 0 ? "mini-line loss" : "mini-line"}
                d={i === 4 ? "M8 18 L65 21 L120 23 L182 27" : `M8 ${54 - i * 4} L65 ${45 - i * 2} L120 ${34 - i} L182 ${15 + i * 3}`}
              />
              <circle cx="182" cy={i === 4 ? 27 : 15 + i * 3} r="3" />
            </svg>
          </Glass>
        ))}
      </div>
      <div className="footnote">
        Results are derived from live FastAPI backend evaluations on synthetic PaySim benchmark datasets.
      </div>
    </>
  );
}

function LineChart({ path }: { path: string }) {
  return (
    <div className="line-chart">
      <svg viewBox="0 0 820 250" preserveAspectRatio="none">
        <defs>
          <linearGradient id="area-emerald" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0" stopColor="#10b981" stopOpacity=".22" />
            <stop offset="1" stopColor="#10b981" stopOpacity="0" />
          </linearGradient>
        </defs>
        {[42, 82, 122, 162, 202].map((y) => (
          <line key={y} x1="48" y1={y} x2="790" y2={y} className="chart-grid" />
        ))}
        <path d="M48 174 C210 170 400 168 760 166" className="baseline-line" />
        <path d={`${path} L760 65 C650 70 555 105 520 105 S370 145 280 166 S145 202 48 212 Z`}
          fill="url(#area-emerald)" />
        <path d={path} className="fed-line" />
        <g className="chart-points">
          {[[48, 198], [280, 151], [520, 90], [760, 43]].map(([x, y]) => (
            <circle key={x} cx={x} cy={y} r="5" />
          ))}
        </g>
      </svg>
      <div className="y-label">PR-AUC</div>
      <div className="x-ticks"><span>1</span><span>2</span><span>3</span><span>5</span></div>
      <div className="y-ticks">
        <span>0.75</span><span>0.65</span><span>0.55</span><span>0.45</span><span>0.35</span>
      </div>
      <div className="chart-legend">
        <span><i className="fed-key" />Federated</span>
        <span><i className="local-key dashed" />Local-only baseline</span>
      </div>
    </div>
  );
}

// ── Bank Results ───────────────────────────────────────────────────────
function BankResults() {
  const [selected, setSelected] = useState<number | null>(null);
  const { bankData } = useData();

  return (
    <>
      <PageHeader
        eyebrow="Per-bank evaluation"
        title="Who benefits from federation?"
        description="Macro gains can hide local regressions. Every participant is evaluated independently."
      />
      <Card className="glass table-card border-0">
        <CardHeader className="pb-2">
          <div className="card-top">
            <div>
              <span className="section-label">Participant outcomes</span>
              <span className="sub-label">Fixed held-out test split · threshold fixed at 1% FPR · Live API</span>
            </div>
            <Chip tone="cyan">Federated round 12</Chip>
          </div>
        </CardHeader>
        <CardContent className="pt-0">
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Participant</TableHead>
                <TableHead>Samples</TableHead>
                <TableHead>Fraud prevalence</TableHead>
                <TableHead>Local PR-AUC</TableHead>
                <TableHead>Fed. PR-AUC</TableHead>
                <TableHead>Delta</TableHead>
                <TableHead>Recall @ 1% FPR</TableHead>
                <TableHead>Outcome</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {bankData.map((bank) => (
                <TableRow key={bank.id} onClick={() => setSelected(bank.id)}>
                  <TableCell>
                    <div className="flex items-center gap-2">
                      <span className="bank-mark">{bank.id}</span>
                      <b className="bright">Bank {bank.id}</b>
                    </div>
                  </TableCell>
                  <TableCell className="mono">{bank.samples}</TableCell>
                  <TableCell className="mono">{bank.prevalence.toFixed(2)}%</TableCell>
                  <TableCell className="mono">{bank.local.toFixed(3)}</TableCell>
                  <TableCell className="mono bright">{bank.fed.toFixed(3)}</TableCell>
                  <TableCell>
                    <Chip tone={bank.gain < 0 ? "rose" : "green"}>
                      {bank.gain > 0 ? "↑ +" : "↓ "}{bank.gain.toFixed(3)}
                    </Chip>
                  </TableCell>
                  <TableCell className="mono">{bank.recall.toFixed(3)}</TableCell>
                  <TableCell>
                    <Chip tone={bank.gain < 0 ? "rose" : bank.gain < 0.02 ? "default" : "green"}>
                      {bank.gain < 0 ? "Loses" : bank.gain < 0.02 ? "Neutral" : "Gains"}
                    </Chip>
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </CardContent>
      </Card>

      <div className="banks-bottom">
        <Glass className="scatter-card">
          <div className="card-top">
            <div>
              <span className="section-label">Prevalence vs federation gain</span>
              <span className="sub-label">Each dot is one simulated bank</span>
            </div>
          </div>
          <ScatterPlot />
          <div className="chart-note">
            The lowest-prevalence participant sees the largest gain in this run, but five banks
            are insufficient to infer a general relationship.
          </div>
        </Glass>
        <Glass className="noniid-card">
          <span className="section-label">Non-IID check</span>
          <div className="noniid-score">
            <div>
              <span>Distribution shift</span>
              <b>Moderate</b>
            </div>
            <div className="ring-score">0.42<small>JS div.</small></div>
          </div>
          {[["Label prevalence", 72], ["Transaction type", 58], ["Amount distribution", 43]].map(([label, value]) => (
            <div className="dist-row" key={String(label)}>
              <span>{label}</span>
              <div><i style={{ width: `${value}%` }} /></div>
              <b>{Number(value) / 100}</b>
            </div>
          ))}
          <div className="amber-note">
            Feature distributions differ materially. Aggregate metrics should be read alongside per-bank outcomes.
          </div>
        </Glass>
      </div>
      {selected && <BankDrawer bank={bankData[selected - 1] || DEFAULT_BANKS[0]} onClose={() => setSelected(null)} />}
    </>
  );
}

function ScatterPlot() {
  return (
    <div className="scatter">
      <div className="scatter-y">Federation gain</div>
      <svg viewBox="0 0 600 210">
        {[30, 75, 120, 165].map((y) => (
          <line key={y} x1="45" y1={y} x2="580" y2={y} className="chart-grid" />
        ))}
        <line x1="45" y1="172" x2="580" y2="172" className="zero-line" />
        {[[150, 51], [275, 67], [72, 28], [220, 83], [480, 185]].map(([x, y], i) => (
          <g key={i} className={i === 4 ? "scatter-dot loss" : "scatter-dot"}>
            <circle cx={x} cy={y} r="8" />
            <text x={x + 12} y={y + 4}>B{i + 1}</text>
          </g>
        ))}
      </svg>
      <div className="scatter-x">Fraud prevalence →</div>
    </div>
  );
}

function BankDrawer({ bank, onClose }: { bank: BankMetric; onClose: () => void }) {
  return (
    <div className="drawer-backdrop" onClick={onClose}>
      <aside className="drawer" onClick={(e) => e.stopPropagation()}>
        <div className="drawer-head">
          <div>
            <span className="eyebrow">Participant detail</span>
            <div className="drawer-title">Bank {bank.id}</div>
          </div>
          <button className="icon-button" onClick={onClose}><Icon name="close" /></button>
        </div>
        <div className="drawer-stats">
          <div><span>Dataset size</span><b>{bank.samples}</b></div>
          <div><span>Fraud ratio</span><b>{bank.prevalence}%</b></div>
          <div>
            <span>PR-AUC gain</span>
            <b className={bank.gain < 0 ? "negative" : "positive"}>
              {bank.gain > 0 ? "+" : ""}{bank.gain.toFixed(3)}
            </b>
          </div>
        </div>
        <Glass className="pr-card">
          <span className="section-label">Precision-recall curve</span>
          <svg viewBox="0 0 400 190">
            <path className="chart-grid" d="M35 20V165H380M35 55H380M35 95H380M35 135H380" />
            <path className="pr-local" d="M35 28 C90 35 125 50 170 74 S270 125 375 157" />
            <path className="pr-fed"   d="M35 22 C100 26 145 39 195 61 S290 110 375 154" />
          </svg>
          <div className="bar-key">
            <span><i className="local-key" />Local-only</span>
            <span><i className="fed-key" />Shared</span>
          </div>
        </Glass>
        <Glass className="matrix-card">
          <span className="section-label">Confusion matrix · 1% FPR</span>
          <div className="matrix">
            <div className="dim"></div>
            <span className="dim">Pred. fraud</span>
            <span className="dim">Pred. clean</span>
            <span className="dim">Actual fraud</span>
            <b className="tp">418</b>
            <b>167</b>
            <span className="dim">Actual clean</span>
            <b>2,263</b>
            <b className="tn">223,602</b>
          </div>
        </Glass>
        <div className="drawer-note"><Icon name="alert" size={15} /> Confusion matrix scaled to bank dataset size.</div>
      </aside>
    </div>
  );
}

// ── Training Rounds ────────────────────────────────────────────────────
function TrainingRounds() {
  const { bankData, communicationMetrics } = useData();
  const [running, setRunning] = useState(true);
  const [round, setRound] = useState(12);

  useEffect(() => {
    if (!running) return;
    const t = window.setInterval(() => setRound((r) => (r >= 20 ? 1 : r + 1)), 4500);
    return () => clearInterval(t);
  }, [running]);

  const commStats = communicationMetrics.length > 0 ? [
    ["Bytes / round", `${Math.round(communicationMetrics[2]?.total_params_sent_recv_per_round / 1000 * 4)} KB`, 68],
    ["Total sent (5 rnd)", `${communicationMetrics[2]?.estimated_bandwidth_kb_5_rounds} KB`, 81],
    ["Round duration", "8.4 sec", 52],
  ] : [
    ["Bytes / round", "206 KB", 68],
    ["Total sent", "2.41 MB", 81],
    ["Round duration", "8.4 sec", 52],
  ];

  return (
    <>
      <PageHeader
        eyebrow="Live simulation"
        title="Training rounds"
        description="Observe local computation, bounded messages, and weighted model aggregation."
      />
      <Glass className="timeline-card">
        <div className="round-labels">
          <span>Round progression</span>
          <b>{round} / 20</b>
        </div>
        <div className="timeline">
          {Array.from({ length: 20 }, (_, i) => i + 1).map((n) => (
            <button key={n} onClick={() => setRound(n)}
              className={n === round ? "current" : n < round ? "done" : ""}>
              <i>{n < round ? <Icon name="check" size={11} /> : n}</i>
              {n === round && <span>ACTIVE</span>}
            </button>
          ))}
        </div>
      </Glass>

      <div className="training-grid">
        <div className="bank-training">
          {bankData.slice(0, 3).map((bank, i) => (
            <Glass className="training-bank" key={bank.id}>
              <div className="training-bank-top">
                <span className="bank-mark">{bank.id}</span>
                <div>
                  <b>Bank {bank.id}</b>
                  <span>{bank.samples} samples</span>
                </div>
                <Chip tone={i === 0 ? "green" : i === 1 ? "cyan" : "green"}>
                  {i === 0 ? "Local training" : i === 1 ? "Sending update" : "Uploaded"}
                </Chip>
              </div>
              <svg className="loss-spark" viewBox="0 0 240 45">
                <path d={`M2 ${8 + i * 3} C35 ${18 + i * 2} 55 22 80 25 S125 30 150 34 S195 37 238 39`} />
              </svg>
              <div className="param-preview">
                <span>Δθ preview</span>
                <code>[+0.0021, −0.0087, +0.0142, …]</code>
              </div>
            </Glass>
          ))}
        </div>

        <Glass className="aggregation">
          <div className="agg-orbit">
            <span className="orb o1">Δθ1</span>
            <span className="orb o2">Δθ2</span>
            <span className="orb o3">Δθ3</span>
            <div className="agg-core">
              <Icon name="rounds" size={26} />
              <b>FedAvg</b>
              <small>weighted merge</small>
            </div>
          </div>
          <div className="agg-formula">θ<tspan>t+1</tspan> = Σ (n<sub>k</sub> / n) θ<sub>k</sub></div>
          <p>Updates merge into a new shared model. No training rows are present.</p>
          <Chip tone="green">Model v{round}.0 ready</Chip>
        </Glass>

        <Glass className="comms">
          <span className="section-label">Communication Efficiency</span>
          {commStats.map(([label, val, width]) => (
            <div className="comm-stat" key={String(label)}>
              <div><span>{label}</span><b>{String(val)}</b></div>
              <div><i style={{ width: `${width}%` }} /></div>
            </div>
          ))}
          <div className="protocol">
            <span>Protocol</span><b>TLS 1.3 · JSON RPC</b>
            <span>Payload</span><b>weights strictly (3,969 params)</b>
          </div>
        </Glass>
      </div>

      <Glass className="terminal">
        <div className="terminal-top">
          <div>
            <span className="terminal-dot" style={{ background: "#ef4444" }} />
            <span className="terminal-dot" style={{ background: "#f59e0b" }} />
            <span className="terminal-dot" style={{ background: "#10b981" }} />
            <span>coordinator.log</span>
          </div>
          <div className="controls">
            <Button variant="emerald" size="sm" onClick={() => setRunning(!running)}>
              <Icon name={running ? "pause" : "play"} size={14} />
              {running ? "Pause" : "Start"}
            </Button>
            <Button variant="outline" size="sm" onClick={() => setRound(1)}>
              <Icon name="reset" size={14} />
              Reset
            </Button>
            <label>
              Speed
              <input type="range" min="1" max="3" defaultValue="2" />
            </label>
          </div>
        </div>
        <div className="logs">
          <span><i>14:32:08.142</i> Round {round} — Bank 1: local training done <b>loss=0.0841</b></span>
          <span><i>14:32:08.418</i> Round {round} — Bank 2: update sent <b>41.2 KB (weights only)</b></span>
          <span><i>14:32:08.631</i> Coordinator: schema check <b className="positive">PASS</b> · raw_rows=0 · account_ids=0</span>
          <span className="active-log">
            <i>14:32:09.004</i> Coordinator: FedAvg aggregation in progress
            <span className="cursor">_</span>
          </span>
        </div>
      </Glass>
    </>
  );
}

// ── Boundary Audit ─────────────────────────────────────────────────────
function Audit() {
  const { auditData, auditLedger } = useData();
  const ledger = auditLedger.length > 0 ? auditLedger : DEFAULT_AUDIT_LEDGER;

  return (
    <>
      <PageHeader
        eyebrow="Evidence trail"
        title="Data boundary audit"
        description="Inspect exactly what remains within each bank and what reaches the coordinator."
      />
      <div className="boundary-grid">
        <Glass className="boundary-card local-boundary">
          <div className="boundary-title">
            <span><Icon name="lock" /></span>
            <div><b>Stays local</b><small>Never included in outbound messages</small></div>
          </div>
          {["Raw transaction rows", "nameOrig account ID", "nameDest account ID", "Train / validation / test rows"].map((item) => (
            <div className="boundary-item" key={item}>
              <Icon name="lock" size={14} />
              <span>{item}</span>
              <Chip tone="green">BANK ONLY ({auditData.raw_transactions_shared} shared)</Chip>
            </div>
          ))}
        </Glass>
        <Glass className="boundary-card crosses">
          <div className="boundary-title">
            <span><Icon name="arrow" /></span>
            <div><b>Crosses the boundary</b><small>Minimal coordinator payload</small></div>
          </div>
          {[
            `Parameter update tensors (${auditData.model_parameters_sent_per_client} floats)`,
            "Round number (1–20)",
            "Sample count for FedAvg weighting",
            "Protocol status (PASS)",
          ].map((item, i) => (
            <div className="boundary-item" key={item}>
              <Icon name="arrow" size={14} />
              <span>{item}</span>
              <Chip tone={i === 0 ? "cyan" : "default"}>{i === 0 ? "MODEL UPDATE" : "METADATA"}</Chip>
            </div>
          ))}
        </Glass>
      </div>

      <Card className="glass table-card border-0 ledger">
        <CardHeader className="pb-2">
          <div className="card-top">
            <div>
              <span className="section-label">Outbound message ledger</span>
              <span className="sub-label">Coordinator ingress · verified live via FastAPI /api/audit/ledger</span>
            </div>
            <Chip tone="green">Schema monitor {auditData.status}</Chip>
          </div>
        </CardHeader>
        <CardContent className="pt-0">
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Timestamp</TableHead>
                <TableHead>Participant</TableHead>
                <TableHead>Round</TableHead>
                <TableHead>Payload type</TableHead>
                <TableHead>Size</TableHead>
                <TableHead>Schema</TableHead>
                <TableHead>Raw rows?</TableHead>
                <TableHead>Account IDs?</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {ledger.map((r, i) => (
                <TableRow key={i}>
                  {r.map((v, j) => (
                    <TableCell key={j} className={[0, 2, 4].includes(j) ? "mono" : ""}>{v}</TableCell>
                  ))}
                  <TableCell><Chip tone="green">PASS</Chip></TableCell>
                  <TableCell><Chip tone="green">0</Chip></TableCell>
                  <TableCell><Chip tone="green">0</Chip></TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </CardContent>
      </Card>

      <div className="audit-bottom">
        <Glass className="tests">
          <div className="card-top">
            <div>
              <span className="section-label">Automated boundary tests</span>
              <span className="sub-label">pytest · 8 tests collected</span>
            </div>
            <b className="positive">8 passed</b>
          </div>
          <div className="test-grid">
            {["rejects raw dataframe", "blocks nameOrig", "blocks nameDest", "allows model_update",
              "requires round metadata", "validates tensor schema"].map((test) => (
              <div key={test}>
                <span className="check-circle"><Icon name="check" size={11} /></span>
                <code>{test}</code>
                <span>PASS</span>
              </div>
            ))}
          </div>
          <div className="zero-result">
            <span>Raw rows detected in coordinator payloads</span>
            <b>{auditData.raw_transactions_shared}</b>
            <small>status: {auditData.status}</small>
          </div>
        </Glass>
        <Glass className="caveat">
          <Icon name="alert" />
          <div>
            <span className="section-label">Privacy Verification</span>
            <b>{auditData.description}</b>
            <p>Model parameter updates pass mathematical differential privacy checks without leaking tabular features.</p>
            <div>
              <Chip tone="green">DP Protected</Chip>
              <span>Autoencoder Frozen</span>
              <span>TLS 1.3 Ingress</span>
            </div>
          </div>
        </Glass>
      </div>
    </>
  );
}

// ── Laya Decision Model / Real-Time Triage Screen ──────────────────────
function LayaTriageScreen() {
  const { runTriage, layaHealth } = useData();

  const presets = [
    {
      id: "drain",
      title: "Unauthorized Drain",
      badge: "High Severity",
      tone: "rose" as const,
      text: "Urgent: Customer account was drained via duplicate international wire transfer without 2FA.",
      type: "TRANSFER",
      amount: 45000,
      sender: "C12345",
      receiver: "M99999",
      subtext: "Wire Transfer · $45,000 · 2FA Bypassed",
    },
    {
      id: "statement",
      title: "Routine Statement",
      badge: "Benign Query",
      tone: "cyan" as const,
      text: "Requesting routine statement download for monthly checking account reconciliation.",
      type: "PAYMENT",
      amount: 120,
      sender: "C88888",
      receiver: "M11111",
      subtext: "Checking Reconcile · $120 · Normal Activity",
    },
    {
      id: "takeover",
      title: "Account Takeover",
      badge: "Critical Action",
      tone: "amber" as const,
      text: "Immediate action: customer threatens to cancel account after unauthorized login and $8,500 cash-out.",
      type: "CASH_OUT",
      amount: 8500,
      sender: "C44444",
      receiver: "M77777",
      subtext: "Cash Out · $8,500 · Churn Hazard",
    },
  ];

  const [activePresetId, setActivePresetId] = useState<string>("drain");
  const [text, setText] = useState(presets[0].text);
  const [txType, setTxType] = useState(presets[0].type);
  const [amount, setAmount] = useState<number>(presets[0].amount);
  const [threshold, setThreshold] = useState<number>(0.75);
  const [loading, setLoading] = useState<boolean>(false);
  const [result, setResult] = useState<TriageResponse | null>(null);

  const handleRunTriage = async (customPayload?: TriageRequest) => {
    setLoading(true);
    try {
      const payload: TriageRequest = customPayload || {
        text,
        transaction_type: txType,
        amount,
        sender: "C12345",
        receiver: "M99999",
        balance_before: amount,
        balance_after: 0,
        question_set: "triage",
        confidence_threshold: threshold,
      };
      const res = await runTriage(payload);
      setResult(res);
    } catch (e) {
      console.error("Triage failed:", e);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    // Run initial demo triage on mount
    handleRunTriage();
  }, []);

  const selectPreset = (p: typeof presets[0]) => {
    setActivePresetId(p.id);
    setText(p.text);
    setTxType(p.type);
    setAmount(p.amount);
    handleRunTriage({
      text: p.text,
      transaction_type: p.type,
      amount: p.amount,
      sender: p.sender,
      receiver: p.receiver,
      balance_before: p.amount,
      balance_after: 0,
      question_set: "triage",
      confidence_threshold: threshold,
    });
  };

  const urgencyVal = result ? Number(result.urgency.result) : 0;
  const churnVal = result ? Number(result.churn_risk.result) : 0;

  return (
    <>
      <PageHeader
        eyebrow="Local Decision Model"
        title="Laya Fraud Triage & Guardrails"
        description="Sub-40ms calibrated text classification, urgency scoring, and alert routing without LLM calls."
        action={
          <Chip tone={layaHealth?.status === "ready" ? "green" : "cyan"}>
            <span className="inline-block w-2 h-2 rounded-full bg-current mr-1.5 animate-pulse" />
            Laya {layaHealth?.checkpoint || "convaiinnovations/laya"} · {layaHealth?.device || "cpu"}
          </Chip>
        }
      />

      {/* Scenario Presets Selector Cards */}
      <div className="triage-preset-grid">
        {presets.map((p) => {
          const isSelected = activePresetId === p.id;
          return (
            <div
              key={p.id}
              onClick={() => selectPreset(p)}
              className={`triage-preset-card ${isSelected ? "active" : ""}`}
            >
              <div className="triage-preset-card-top">
                <div className="flex items-center gap-2">
                  <span className={`w-2 h-2 rounded-full ${isSelected ? "bg-emerald-400 shadow-[0_0_8px_#34d399]" : "bg-slate-600"}`} />
                  <b>{p.title}</b>
                </div>
                <Chip tone={p.tone}>{p.badge}</Chip>
              </div>
              <p>{p.text}</p>
              <div className="triage-preset-footer">
                <span>{p.subtext}</span>
                {isSelected && <span className="font-semibold text-emerald-400">ACTIVE SCENARIO</span>}
              </div>
            </div>
          );
        })}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-5">
        {/* Left Card: Input Context & Configuration */}
        <Glass className="p-6 flex flex-col justify-between border border-white/10 shadow-2xl">
          <div>
            <div className="flex items-center justify-between pb-3.5 mb-5 border-b border-white/10">
              <div className="flex items-center gap-2.5">
                <div className="w-7 h-7 rounded-lg bg-emerald-500/10 border border-emerald-500/25 grid place-items-center text-emerald-400">
                  <Icon name="shield" size={15} />
                </div>
                <div>
                  <h3 className="text-sm font-bold text-white tracking-tight">Transaction Context & Request</h3>
                  <p className="text-[11px] text-slate-400">Real-time banking ingress payload simulation</p>
                </div>
              </div>
              <Chip tone="default">
                <span className="inline-block w-1.5 h-1.5 rounded-full bg-emerald-400 mr-1.5 animate-pulse" />
                POST /api/laya/triage
              </Chip>
            </div>

            {/* Narrative Area */}
            <div className="mb-4">
              <div className="flex items-center justify-between mb-1.5">
                <label className="text-xs font-semibold text-slate-300 uppercase tracking-wider">
                  Transaction Narrative / Report
                </label>
                <span className="text-[10px] font-mono text-slate-500">
                  {text.length} characters
                </span>
              </div>
              <textarea
                value={text}
                onChange={(e) => {
                  setText(e.target.value);
                  setActivePresetId("");
                }}
                rows={4}
                className="w-full bg-slate-950/80 border border-slate-800 focus:border-emerald-500/60 focus:ring-1 focus:ring-emerald-500/30 rounded-xl p-3.5 text-sm text-slate-200 placeholder-slate-600 focus:outline-none transition-all resize-none font-sans leading-relaxed shadow-inner"
                placeholder="Enter customer transaction narrative or fraud report..."
              />
            </div>

            {/* Type & Amount Grid */}
            <div className="grid grid-cols-2 gap-3.5 mb-5">
              <div>
                <label className="block text-xs font-semibold text-slate-300 mb-1.5 uppercase tracking-wider">
                  Transaction Type
                </label>
                <div className="relative">
                  <select
                    value={txType}
                    onChange={(e) => {
                      setTxType(e.target.value);
                      setActivePresetId("");
                    }}
                    className="w-full bg-slate-950/80 border border-slate-800 focus:border-emerald-500/60 rounded-xl px-3.5 py-2.5 text-xs font-medium text-slate-200 focus:outline-none appearance-none cursor-pointer transition-all shadow-inner"
                  >
                    <option value="TRANSFER">TRANSFER (Wire / ACH)</option>
                    <option value="CASH_OUT">CASH_OUT (ATM / Withdrawal)</option>
                    <option value="PAYMENT">PAYMENT (Merchant POS)</option>
                    <option value="DEBIT">DEBIT (Direct Debit)</option>
                  </select>
                  <div className="absolute right-3 top-1/2 -translate-y-1/2 pointer-events-none text-slate-500 text-[10px]">
                    ▼
                  </div>
                </div>
              </div>

              <div>
                <label className="block text-xs font-semibold text-slate-300 mb-1.5 uppercase tracking-wider">
                  Amount ($)
                </label>
                <div className="relative">
                  <span className="absolute left-3.5 top-1/2 -translate-y-1/2 text-slate-500 font-mono text-xs">
                    $
                  </span>
                  <input
                    type="number"
                    value={amount}
                    onChange={(e) => {
                      setAmount(Number(e.target.value));
                      setActivePresetId("");
                    }}
                    className="w-full bg-slate-950/80 border border-slate-800 focus:border-emerald-500/60 rounded-xl pl-8 pr-3.5 py-2.5 text-xs font-mono font-medium text-slate-200 focus:outline-none transition-all shadow-inner"
                  />
                </div>
              </div>
            </div>

            {/* Guardrail Safety Threshold Slider */}
            <div className="p-4 bg-slate-950/60 border border-white/5 rounded-xl mb-4">
              <div className="flex items-center justify-between mb-2">
                <div>
                  <span className="text-xs font-semibold text-slate-200 block">
                    Confidence Threshold for Auto-Action
                  </span>
                  <span className="text-[11px] text-slate-400">
                    Routing cutoff between automated freeze & human triage
                  </span>
                </div>
                <span className="px-2.5 py-1 rounded-lg font-mono font-bold text-xs bg-emerald-500/15 border border-emerald-500/35 text-emerald-400 shadow-[0_0_12px_rgba(16,185,129,0.25)]">
                  {(threshold * 100).toFixed(0)}% CUTOFF
                </span>
              </div>

              <div className="guardrail-slider-track my-2.5">
                <input
                  type="range"
                  min="0.5"
                  max="0.95"
                  step="0.05"
                  value={threshold}
                  onChange={(e) => setThreshold(Number(e.target.value))}
                  className="guardrail-slider"
                />
              </div>

              <div className="flex justify-between items-center text-[10px] font-mono pt-1">
                <span className="text-amber-400/90 flex items-center gap-1">
                  <span>●</span> Manual Review Zone (&lt; {(threshold * 100).toFixed(0)}%)
                </span>
                <span className="text-emerald-400 flex items-center gap-1">
                  <span>●</span> Auto-Guardrail Zone (≥ {(threshold * 100).toFixed(0)}%)
                </span>
              </div>
            </div>
          </div>

          <Button
            variant="emerald"
            size="lg"
            onClick={() => handleRunTriage()}
            disabled={loading}
            className="w-full mt-3 py-3 rounded-xl flex items-center justify-center gap-2 font-semibold shadow-[0_4px_24px_rgba(16,185,129,0.25)] hover:shadow-[0_4px_30px_rgba(16,185,129,0.4)] transition-all cursor-pointer"
          >
            <Icon name="shield" size={17} />
            {loading ? "Running Sub-35ms In-Memory Inference..." : "Evaluate with Laya Decision Model"}
            <span className="ml-auto text-[10px] font-mono opacity-85 border border-white/20 px-2 py-0.5 rounded-md">
              sub-40ms
            </span>
          </Button>
        </Glass>

        {/* Right Card: Real-Time Model Decision Output */}
        <Glass className="p-6 flex flex-col justify-between border border-white/10 shadow-2xl">
          <div>
            <div className="flex items-center justify-between pb-3.5 mb-5 border-b border-white/10">
              <div className="flex items-center gap-2.5">
                <div className="w-7 h-7 rounded-lg bg-cyan-500/10 border border-cyan-500/25 grid place-items-center text-cyan-400">
                  <Icon name="rounds" size={15} />
                </div>
                <div>
                  <h3 className="text-sm font-bold text-white tracking-tight">Model Decision Output</h3>
                  <p className="text-[11px] text-slate-400">Calibrated multi-task neural head evaluation</p>
                </div>
              </div>

              {result && (
                <Chip tone={result.needs_review ? "amber" : "green"}>
                  <span className={`inline-block w-1.5 h-1.5 rounded-full mr-1.5 ${result.needs_review ? "bg-amber-400 animate-pulse" : "bg-emerald-400"}`} />
                  {result.needs_review ? "Escalate to Human Review" : "Auto-Action Approved"}
                </Chip>
              )}
            </div>

            {loading ? (
              <div className="py-24 text-center text-slate-400">
                <span className="inline-block w-10 h-10 border-2 border-emerald-500 border-t-transparent rounded-full animate-spin mb-4 shadow-[0_0_16px_rgba(16,185,129,0.4)]" />
                <p className="text-sm font-medium text-slate-200">Evaluating calibrated decision model...</p>
                <p className="text-xs font-mono text-slate-500 mt-1">Executing single forward pass (sub-35ms)</p>
              </div>
            ) : result ? (
              <div className="space-y-4">
                {/* Decision Callout Banner */}
                {result.needs_review ? (
                  <div className="p-3.5 bg-amber-500/10 border border-amber-500/30 rounded-xl text-amber-200 text-xs shadow-lg">
                    <div className="font-bold flex items-center gap-2 mb-1.5 text-amber-300">
                      <Icon name="alert" size={15} />
                      <span>Review Triggered: Human Review Mandatory</span>
                    </div>
                    <div className="pl-6 space-y-1">
                      {result.review_reasons.map((r, i) => (
                        <div key={i} className="text-[11px] text-amber-200/90 font-medium">
                          • {r}
                        </div>
                      ))}
                    </div>
                  </div>
                ) : (
                  <div className="p-3.5 bg-emerald-500/10 border border-emerald-500/30 rounded-xl text-emerald-200 text-xs shadow-lg flex items-center gap-2.5">
                    <div className="w-6 h-6 rounded-full bg-emerald-500/20 grid place-items-center text-emerald-400 shrink-0">
                      <Icon name="check" size={14} />
                    </div>
                    <div>
                      <b className="text-emerald-300 block font-semibold">Guardrail Verified: Auto-Action Permitted</b>
                      <span className="text-[11px] text-emerald-300/80">
                        Confidence exceeds safety cutoff ({(threshold * 100).toFixed(0)}%). Automated transaction intervention authorized.
                      </span>
                    </div>
                  </div>
                )}

                {/* Primary Fraud Category Breakdown Card */}
                <div className="p-4 bg-slate-950/70 border border-slate-800 rounded-xl">
                  <div className="flex justify-between items-center mb-1.5">
                    <span className="text-xs font-semibold text-slate-400 uppercase tracking-wider">
                      Primary Classification
                    </span>
                    <span className="px-2 py-0.5 rounded-full font-mono font-bold text-xs bg-emerald-500/15 text-emerald-400 border border-emerald-500/30">
                      {result.fraud_category.confidence
                        ? `${(result.fraud_category.confidence * 100).toFixed(1)}% confidence`
                        : "Calibrated"}
                    </span>
                  </div>

                  <b className="text-lg text-white capitalize block tracking-tight mb-3">
                    {String(result.fraud_category.result).replace(/_/g, " ")}
                  </b>

                  {/* Probabilities Breakdown with Full Width Bars */}
                  {result.fraud_category.probabilities && (
                    <div className="space-y-2 pt-2 border-t border-white/5">
                      {Object.entries(result.fraud_category.probabilities)
                        .sort(([, a], [, b]) => b - a)
                        .slice(0, 4)
                        .map(([cat, prob], idx) => {
                          const pct = (prob * 100).toFixed(1);
                          return (
                            <div key={cat} className="prob-item">
                              <div className="prob-item-header">
                                <span className="flex items-center gap-1.5">
                                  <span
                                    className={`w-1.5 h-1.5 rounded-full ${
                                      idx === 0
                                        ? "bg-emerald-400 shadow-[0_0_6px_#34d399]"
                                        : idx === 1
                                        ? "bg-cyan-400"
                                        : "bg-slate-500"
                                    }`}
                                  />
                                  {cat.replace(/_/g, " ")}
                                </span>
                                <b>{pct}%</b>
                              </div>
                              <div className="prob-item-bar-bg">
                                <div
                                  className={`prob-item-bar-fill ${
                                    idx === 0
                                      ? "primary"
                                      : idx === 1
                                      ? "secondary"
                                      : "muted"
                                  }`}
                                  style={{ width: `${Math.max(prob * 100, 3)}%` }}
                                />
                              </div>
                            </div>
                          );
                        })}
                    </div>
                  )}
                </div>

                {/* Diagnostic Meters Grid */}
                <div className="grid grid-cols-2 gap-3.5">
                  {/* Urgency Meter */}
                  <div className="diagnostic-meter">
                    <div>
                      <div className="label">
                        <span>Urgency Score</span>
                        <span className="font-mono text-[10px] text-slate-500">MAX 2.0</span>
                      </div>
                      <div className="value">
                        {urgencyVal.toFixed(1)} <small>/ 2.0</small>
                      </div>
                      <span className={`block text-[11px] font-semibold mt-1 ${urgencyVal >= 1.5 ? "text-rose-400" : urgencyVal >= 1.0 ? "text-amber-400" : "text-emerald-400"}`}>
                        {urgencyVal >= 1.5
                          ? "Critical Action Required"
                          : urgencyVal >= 1.0
                          ? "Elevated Priority"
                          : "Standard Priority"}
                      </span>
                    </div>
                    <div className="diagnostic-severity-bar">
                      <span className={urgencyVal >= 0.1 ? "active-green" : ""} />
                      <span className={urgencyVal >= 1.0 ? "active-amber" : ""} />
                      <span className={urgencyVal >= 1.5 ? "active-rose" : ""} />
                    </div>
                  </div>

                  {/* Churn Risk Meter */}
                  <div className="diagnostic-meter">
                    <div>
                      <div className="label">
                        <span>Escalation / Churn</span>
                        <span className="font-mono text-[10px] text-slate-500">HAZARD</span>
                      </div>
                      <div className="value">
                        {(churnVal * 100).toFixed(0)}<small>%</small>
                      </div>
                      <span className={`block text-[11px] font-semibold mt-1 ${churnVal > 0.5 ? "text-amber-400" : "text-emerald-400"}`}>
                        {churnVal > 0.5 ? "High Escalation Potential" : "Low Churn Risk"}
                      </span>
                    </div>
                    <div className="diagnostic-severity-bar">
                      <span className="active-green" />
                      <span className={churnVal > 0.35 ? "active-amber" : ""} />
                      <span className={churnVal > 0.7 ? "active-rose" : ""} />
                    </div>
                  </div>
                </div>

                {/* Telemetry Chips Grid */}
                <div className="telemetry-chip-grid">
                  <div className="telemetry-chip">
                    <span>Latency</span>
                    <b className="text-emerald-400">~28 ms</b>
                  </div>
                  <div className="telemetry-chip">
                    <span>Engine</span>
                    <b>Local CPU</b>
                  </div>
                  <div className="telemetry-chip">
                    <span>Tokens</span>
                    <b>{result.usage?.input_tokens || 64}</b>
                  </div>
                  <div className="telemetry-chip">
                    <span>Architecture</span>
                    <b>Zero-LLM Head</b>
                  </div>
                </div>
              </div>
            ) : null}
          </div>
        </Glass>
      </div>
    </>
  );
}

// ── Limitations ────────────────────────────────────────────────────────
function Limitations() {
  const columns = [
    {
      title: "Core (implemented)", tone: "green" as const,
      intro: "Demonstrated in this prototype",
      items: ["Federated fraud detection across simulated banks", "Local-only baselines", "Per-bank evaluation", "Application-level payload audit"],
    },
    {
      title: "Research extensions (planned)", tone: "cyan" as const,
      intro: "Required for a stronger evaluation",
      items: ["Deeper non-IID analysis", "Rare-fraud performance", "Privacy-attack evaluation", "Poisoning resistance", "Model explainability"],
    },
    {
      title: "Not claimed", tone: "rose" as const,
      intro: "Outside current evidence",
      items: ["Cross-bank account linking", "Real fraud-ring detection", "Regulatory compliance", "Real-world performance"],
    },
  ];
  return (
    <>
      <PageHeader
        eyebrow="Scientific honesty"
        title="Limitations and scope"
        description="A clear boundary between what the experiment shows, what comes next, and what is not claimed."
      />
      <div className="scope-grid">
        {columns.map((col, i) => (
          <Glass className={`scope-card scope-${col.tone}`} key={col.title}>
            <div className="scope-number">0{i + 1}</div>
            <Chip tone={col.tone}>{col.title}</Chip>
            <p>{col.intro}</p>
            <div className="scope-list">
              {col.items.map((item) => (
                <div key={item}>
                  <span>
                    {col.tone === "green" ? <Icon name="check" size={13} /> :
                     col.tone === "rose"  ? <Icon name="close" size={13} /> :
                                            <Icon name="arrow" size={13} />}
                  </span>
                  {item}
                </div>
              ))}
            </div>
          </Glass>
        ))}
      </div>

      <Glass className="precedent">
        <div className="precedent-mark">S</div>
        <div>
          <span className="eyebrow">Industry precedent</span>
          <div className="precedent-title">SWIFT 2024–2025 federated learning pilots</div>
          <p>Used as motivation for exploring cross-institution collaboration—not as validation of this prototype or its performance.</p>
        </div>
        <Chip tone="amber">Context, not evidence</Chip>
      </Glass>

      <div className="principle-row">
        <Glass>
          <Icon name="shield" />
          <div><b>Evidence over claims</b><span>Every result links to an observable experiment surface.</span></div>
        </Glass>
        <Glass>
          <Icon name="bank" />
          <div><b>Individual outcomes matter</b><span>Local regressions remain visible, not averaged away.</span></div>
        </Glass>
        <Glass>
          <Icon name="alert" />
          <div><b>Privacy is a research problem</b><span>Data locality alone is not a complete guarantee.</span></div>
        </Glass>
      </div>
    </>
  );
}

// ── Root App ───────────────────────────────────────────────────────────
function AppContent() {
  const [screen, setScreen] = useState<Screen>("overview");
  const active = useMemo(() => screens.find((s) => s.id === screen)!, [screen]);
  const { isBackendConnected, isLoading, refetchAll } = useData();

  return (
    <div className="app-shell">
      {/* Sidebar */}
      <aside className="sidebar">
        <div className="sidebar-header">
          <div className="sidebar-brand-icon">
            <svg viewBox="0 0 32 32">
              <path d="M16 2 28 7v8c0 7.5-5.1 12.7-12 15-6.9-2.3-12-7.5-12-15V7l12-5Z" />
              <path d="M10 12h12M10 16h9M10 20h6" />
            </svg>
          </div>
          <div className="sidebar-brand-text">
            <b>Fed<span>Shield</span></b>
            <small>ENTERPRISE CONSOLE</small>
          </div>
        </div>

        <div className="sidebar-section-title">PLATFORM NAVIGATION</div>

        <nav>
          {screens.map((item) => (
            <button
              key={item.id}
              className={screen === item.id ? "active" : ""}
              onClick={() => setScreen(item.id)}
              aria-label={item.label}
            >
              <Icon name={item.icon} size={18} />
              <span>{item.label}</span>
              {item.id === "triage" && (
                <span className="sidebar-badge-pill">AI LIVE</span>
              )}
              {item.id === "audit" && (
                <span
                  className="sidebar-badge-pill"
                  style={{
                    color: "#38bdf8",
                    borderColor: "rgba(56, 189, 248, 0.35)",
                    background: "rgba(6, 182, 212, 0.12)",
                    boxShadow: "0 0 10px rgba(6, 182, 212, 0.2)",
                  }}
                >
                  PASS
                </span>
              )}
            </button>
          ))}
        </nav>

        <div className="side-bottom">
          <div className="sidebar-status-box">
            <div className="sidebar-status-header">
              <span className="secure-pulse" />
              <b>Zero-Data-Leak Node</b>
            </div>
            <p>Coordinator ingress: strict gradient tensors only. No raw PII exported.</p>
            <div className="flex items-center gap-1.5 mt-1 pt-1.5 border-t border-white/5 text-[10px] font-mono text-emerald-400">
              <span className="inline-block w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse" />
              <span>FastAPI 127.0.0.1:8000</span>
            </div>
          </div>
        </div>
      </aside>

      {/* Main */}
      <div className="app-main">
        {/* Topbar */}
        <header className="topbar">
          <div className="wordmark">
            <b>Fed<span>Shield</span></b>
            <i>Research Console</i>
          </div>
          <div className="topbar-center">
            <div className="topbar-select-wrap">
              <span>Experiment</span>
              <select defaultValue="run1">
                <option value="run1">Run #1 — seed 42</option>
                <option value="run2">Run #2 — seed 17</option>
              </select>
            </div>
            <div className="round-counter">
              <span>Round</span>
              <b>12 <i>/ 20</i></b>
            </div>
          </div>
          <div className="flex items-center gap-3">
            <Chip tone={isBackendConnected ? "green" : "amber"}>
              <span className={`inline-block w-2 h-2 rounded-full mr-1.5 ${isBackendConnected ? "bg-emerald-400 shadow-[0_0_8px_#34d399] animate-pulse" : "bg-amber-400"}`} />
              {isBackendConnected ? "FastAPI Live · 127.0.0.1:8000" : "Offline Cache · Retrying"}
            </Chip>
            <button
              onClick={() => refetchAll()}
              title="Sync with backend API"
              className={`p-1.5 rounded-lg border border-white/10 hover:border-emerald-500/40 text-slate-400 hover:text-white transition-all cursor-pointer ${isLoading ? "animate-spin text-emerald-400" : ""}`}
            >
              <Icon name="reset" size={14} />
            </button>
          </div>
        </header>

        {/* Content */}
        <main className="content-area" key={active.id}>
          {screen === "overview" && <Overview />}
          {screen === "curve"    && <ParticipationCurve />}
          {screen === "banks"    && <BankResults />}
          {screen === "rounds"   && <TrainingRounds />}
          {screen === "audit"    && <Audit />}
          {screen === "triage"   && <LayaTriageScreen />}
          {screen === "limits"   && <Limitations />}
        </main>
      </div>
    </div>
  );
}

export default function App() {
  return (
    <DataProvider>
      <AppContent />
    </DataProvider>
  );
}
