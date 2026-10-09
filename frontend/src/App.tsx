"use client";

import { useEffect, useMemo, useState } from "react";

type Screen = "overview" | "curve" | "banks" | "rounds" | "audit" | "limits";
type IconName =
  | "grid"
  | "chart"
  | "bank"
  | "rounds"
  | "shield"
  | "alert"
  | "lock"
  | "arrow"
  | "check"
  | "pause"
  | "play"
  | "reset"
  | "close";

const screens: { id: Screen; label: string; icon: IconName }[] = [
  { id: "overview", label: "Overview", icon: "grid" },
  { id: "curve", label: "Participation", icon: "chart" },
  { id: "banks", label: "Bank results", icon: "bank" },
  { id: "rounds", label: "Training rounds", icon: "rounds" },
  { id: "audit", label: "Boundary audit", icon: "shield" },
  { id: "limits", label: "Limitations", icon: "alert" },
];

const bankData = [
  { id: 1, samples: "228,450", prevalence: 0.14, local: 0.611, fed: 0.687, recall: 0.714, gain: 0.076 },
  { id: 2, samples: "194,820", prevalence: 0.11, local: 0.574, fed: 0.662, recall: 0.681, gain: 0.088 },
  { id: 3, samples: "72,190", prevalence: 0.06, local: 0.412, fed: 0.548, recall: 0.603, gain: 0.136 },
  { id: 4, samples: "116,340", prevalence: 0.09, local: 0.521, fed: 0.585, recall: 0.642, gain: 0.064 },
  { id: 5, samples: "301,760", prevalence: 0.22, local: 0.704, fed: 0.691, recall: 0.733, gain: -0.013 },
];

const metricsByParticipation: Record<number, number[]> = {
  1: [0.611, 0.714, 0.681, 0.646, 0.902],
  2: [0.638, 0.729, 0.695, 0.661, 0.911],
  3: [0.661, 0.751, 0.718, 0.689, 0.921],
  5: [0.687, 0.773, 0.742, 0.713, 0.932],
};

function Icon({ name, size = 18 }: { name: IconName; size?: number }) {
  const paths: Record<IconName, React.ReactNode> = {
    grid: <><rect x="3" y="3" width="7" height="7" rx="1" /><rect x="14" y="3" width="7" height="7" rx="1" /><rect x="3" y="14" width="7" height="7" rx="1" /><rect x="14" y="14" width="7" height="7" rx="1" /></>,
    chart: <><path d="M4 19V5" /><path d="M4 19h16" /><path d="m7 15 3-4 3 2 5-7" /></>,
    bank: <><path d="m3 9 9-5 9 5" /><path d="M5 10h14M6 10v7m4-7v7m4-7v7m4-7v7M4 20h16" /></>,
    rounds: <><path d="M20 12a8 8 0 0 1-14.7 4.4L3 13" /><path d="M4 18v-5h5" /><path d="M4 12A8 8 0 0 1 18.7 7.6L21 11" /><path d="M20 6v5h-5" /></>,
    shield: <><path d="M12 3 20 6v5c0 5-3.4 8.5-8 10-4.6-1.5-8-5-8-10V6l8-3Z" /><path d="m8.5 12 2.2 2.2 4.8-5" /></>,
    alert: <><path d="M10.3 4.3 2.8 18a2 2 0 0 0 1.8 3h14.8a2 2 0 0 0 1.8-3L13.7 4.3a2 2 0 0 0-3.4 0Z" /><path d="M12 9v4m0 4h.01" /></>,
    lock: <><rect x="5" y="10" width="14" height="11" rx="2" /><path d="M8 10V7a4 4 0 0 1 8 0v3m-4 4v3" /></>,
    arrow: <><path d="M5 12h14m-5-5 5 5-5 5" /></>,
    check: <path d="m5 12 4 4L19 6" />,
    pause: <><path d="M9 5v14M15 5v14" /></>,
    play: <path d="m8 5 11 7-11 7V5Z" />,
    reset: <><path d="M4 4v6h6" /><path d="M5.5 15a8 8 0 1 0 1-8.5L4 10" /></>,
    close: <path d="m6 6 12 12M18 6 6 18" />,
  };
  return <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">{paths[name]}</svg>;
}

function Chip({ children, tone = "default" }: { children: React.ReactNode; tone?: "default" | "cyan" | "violet" | "green" | "rose" | "amber" }) {
  return <span className={`chip chip-${tone}`}>{children}</span>;
}

function Glass({ className = "", children }: { className?: string; children: React.ReactNode }) {
  return <div className={`glass ${className}`}>{children}</div>;
}

function PageHeader({ eyebrow, title, description }: { eyebrow: string; title: string; description: string }) {
  return (
    <div className="page-head">
      <div>
        <div className="eyebrow">{eyebrow}</div>
        <div className="page-title">{title}</div>
        <p>{description}</p>
      </div>
      <Chip tone="amber">Synthetic data (PaySim) — prototype benchmark</Chip>
    </div>
  );
}

function Segmented({ options, value, onChange }: { options: string[]; value: string; onChange: (next: string) => void }) {
  return <div className="segmented">{options.map((option) => <button key={option} className={value === option ? "active" : ""} onClick={() => onChange(option)}>{option}</button>)}</div>;
}

function Overview() {
  const [banks, setBanks] = useState(5);
  const [scope, setScope] = useState("Macro average");
  const values = metricsByParticipation[banks];
  const metrics = [
    ["PR-AUC", values[0], "+12.4%"],
    ["Recall @ 1% FPR", values[1], "+8.1%"],
    ["Precision", values[2], "+6.6%"],
    ["F1 score", values[3], "+7.9%"],
    ["ROC-AUC", values[4], "+2.8%"],
  ];
  return (
    <>
      <PageHeader eyebrow="Experiment overview" title="Shared signal. Local data." description="Federated fraud-model evaluation across simulated, non-IID bank datasets." />
      <Glass className="imbalance-banner"><span className="info-dot">i</span><strong>Accuracy alone is not used:</strong> fraud is highly imbalanced. Primary metric is Average Precision (PR-AUC).</Glass>
      <div className="overview-grid">
        <Glass className="network-card">
          <div className="card-top">
            <div><span className="section-label">Federation topology</span><span className="live-dot">Round 12 active</span></div>
            <div className="bank-select"><span>Participants</span>{[1, 2, 3, 5].map((n) => <button key={n} className={banks === n ? "active" : ""} onClick={() => setBanks(n)}>{n}</button>)}</div>
          </div>
          <NetworkVisual count={banks} />
          <div className="network-legend"><span><i className="legend-cyan" />Encrypted transport: model updates only</span><span><i className="legend-violet" />Shared model parameters</span></div>
        </Glass>
        <div className="kpi-side">
          <div className="metric-heading"><span className="section-label">Evaluation snapshot</span><Segmented options={["Macro average", "Per bank"]} value={scope} onChange={setScope} /></div>
          <div className="kpi-grid">
            {metrics.map(([label, value, delta], i) => <Glass className={`kpi ${i === 0 ? "primary" : ""}`} key={String(label)}><div className="kpi-label">{label}</div><div className="kpi-value">{Number(value).toFixed(3)}</div><div className="kpi-foot"><span className="positive">↑ {delta}</span><span>vs local-only</span></div></Glass>)}
          </div>
          <Glass className="comparison">
            <div className="card-top"><span className="section-label">Local-only vs federated</span><span className="muted">Placeholder results</span></div>
            {[["PR-AUC", 61, 69], ["Recall", 66, 77], ["F1", 63, 71]].map(([label, local, fed]) => <div className="compare-row" key={String(label)}><span>{label}</span><div className="bar-pair"><i className="bar-local" style={{ width: `${local}%` }} /><i className="bar-fed" style={{ width: `${fed}%` }} /></div><b>{Number(fed) / 100}</b></div>)}
            <div className="bar-key"><span><i className="local-key" />Local-only</span><span><i className="fed-key" />Federated</span></div>
          </Glass>
        </div>
      </div>
    </>
  );
}

function NetworkVisual({ count }: { count: number }) {
  const positions = [{ x: 50, y: 5 }, { x: 82, y: 25 }, { x: 74, y: 70 }, { x: 26, y: 70 }, { x: 18, y: 25 }];
  return (
    <div className="network">
      <svg className="network-lines" viewBox="0 0 100 80" preserveAspectRatio="none">
        {positions.slice(0, count).map((p, i) => <g key={i}><line x1="50" y1="43" x2={p.x} y2={p.y + 7} className="flow-line" /><circle r="1.2" className="packet"><animateMotion dur={`${2.3 + i * .2}s`} repeatCount="indefinite" path={`M 50 43 L ${p.x} ${p.y + 7}`} /></circle></g>)}
      </svg>
      <div className="coordinator"><span className="node-icon"><Icon name="rounds" size={22} /></span><strong>Shared Model</strong><small>Coordinator · FedAvg</small><em>v12.4</em></div>
      {positions.slice(0, count).map((p, i) => <div className="bank-node" key={i} style={{ left: `${p.x}%`, top: `${p.y}%` }}><div className="lock-ring"><Icon name="lock" size={16} /></div><strong>Bank {i + 1}</strong><small>{["228k", "195k", "72k", "116k", "302k"][i]} rows</small><span>LOCAL</span></div>)}
      <div className="boundary-note"><Icon name="lock" size={13} /> Raw transactions + account IDs stay local</div>
    </div>
  );
}

function ParticipationCurve() {
  const [metric, setMetric] = useState("PR-AUC");
  const fed = metric === "PR-AUC" ? "M48 198 C150 189 180 159 280 151 S410 112 520 90 S660 51 760 43" : "M48 187 C155 174 192 154 280 140 S420 107 520 85 S665 56 760 52";
  return (
    <>
      <PageHeader eyebrow="Core proof" title="Participation curve" description="Does adding more banks improve the shared model—and for whom?" />
      <div className="curve-layout">
        <Glass className="chart-card">
          <div className="card-top"><div><span className="section-label">Performance by federation size</span><span className="sub-label">Mean across 5 seeds · shaded area ±1 std</span></div><Segmented options={["PR-AUC", "Recall @ 1% FPR"]} value={metric} onChange={setMetric} /></div>
          <LineChart path={fed} />
        </Glass>
        <Glass className="insight-card">
          <div className="insight-icon"><Icon name="chart" /></div>
          <span className="section-label">What the data says</span>
          <div className="big-insight">4 of 5</div><p>banks improve when all participants contribute.</p>
          <div className="insight-list"><div><span>Bank 3</span><b className="positive">+0.136 PR-AUC</b></div><div><span>Bank 2</span><b className="positive">+0.088 PR-AUC</b></div><div><span>Bank 5</span><b className="negative">−0.013 PR-AUC</b></div></div>
          <div className="honest-note">Bank 5 declines slightly. This is shown—not averaged away.</div>
        </Glass>
      </div>
      <div className="mini-section-head"><div><span className="section-label">Per-bank response</span><span className="sub-label">Individual outcomes as participants join</span></div><span className="muted">1 → 2 → 3 → 5 banks</span></div>
      <div className="mini-grid">{bankData.map((bank, i) => <Glass className="mini-chart" key={bank.id}><div><b>Bank {bank.id}</b><Chip tone={bank.gain < 0 ? "rose" : "green"}>{bank.gain > 0 ? "+" : ""}{bank.gain.toFixed(3)}</Chip></div><svg viewBox="0 0 190 72"><path className="mini-gridline" d="M5 58H185M5 35H185M5 12H185" /><path className={bank.gain < 0 ? "mini-line loss" : "mini-line"} d={i === 4 ? "M8 18 L65 21 L120 23 L182 27" : `M8 ${54-i*4} L65 ${45-i*2} L120 ${34-i} L182 ${15+i*3}`} /><circle cx="182" cy={i === 4 ? 27 : 15+i*3} r="3" /></svg></Glass>)}</div>
      <div className="footnote">Results are from our own evaluation on synthetic data; they are not a prediction for real banks. Values shown are realistic placeholders for this prototype.</div>
    </>
  );
}

function LineChart({ path }: { path: string }) {
  return <div className="line-chart"><svg viewBox="0 0 820 250" preserveAspectRatio="none"><defs><linearGradient id="area" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stopColor="#8B5CF6" stopOpacity=".28" /><stop offset="1" stopColor="#8B5CF6" stopOpacity="0" /></linearGradient></defs>{[42, 82, 122, 162, 202].map((y) => <line key={y} x1="48" y1={y} x2="790" y2={y} className="chart-grid" />)}<path d="M48 174 C210 170 400 168 760 166" className="baseline-line" /><path d={`${path} L760 65 C650 70 555 105 520 105 S370 145 280 166 S145 202 48 212 Z`} fill="url(#area)" /><path d={path} className="fed-line" /><g className="chart-points">{[[48,198],[280,151],[520,90],[760,43]].map(([x,y]) => <circle key={x} cx={x} cy={y} r="5" />)}</g></svg><div className="y-label">PR-AUC</div><div className="x-ticks"><span>1</span><span>2</span><span>3</span><span>5</span></div><div className="y-ticks"><span>0.75</span><span>0.65</span><span>0.55</span><span>0.45</span><span>0.35</span></div><div className="chart-legend"><span><i className="fed-key" />Federated</span><span><i className="local-key dashed" />Local-only baseline</span></div></div>;
}

function BankResults() {
  const [selected, setSelected] = useState<number | null>(null);
  return (
    <>
      <PageHeader eyebrow="Per-bank evaluation" title="Who benefits from federation?" description="Macro gains can hide local regressions. Every participant is evaluated independently." />
      <Glass className="table-card">
        <div className="card-top"><div><span className="section-label">Participant outcomes</span><span className="sub-label">Fixed held-out test split · threshold fixed at 1% FPR</span></div><Chip tone="violet">Federated round 12</Chip></div>
        <div className="table-wrap"><table><thead><tr><th>Participant</th><th>Samples</th><th>Fraud prevalence</th><th>Local PR-AUC</th><th>Fed. PR-AUC</th><th>Delta</th><th>Recall @ 1% FPR</th><th>Outcome</th></tr></thead><tbody>{bankData.map((bank) => <tr key={bank.id} onClick={() => setSelected(bank.id)}><td><span className="bank-mark">{bank.id}</span><b>Bank {bank.id}</b></td><td className="mono">{bank.samples}</td><td className="mono">{bank.prevalence.toFixed(2)}%</td><td className="mono">{bank.local.toFixed(3)}</td><td className="mono bright">{bank.fed.toFixed(3)}</td><td><Chip tone={bank.gain < 0 ? "rose" : "green"}>{bank.gain > 0 ? "↑ +" : "↓ "}{bank.gain.toFixed(3)}</Chip></td><td className="mono">{bank.recall.toFixed(3)}</td><td><Chip tone={bank.gain < 0 ? "rose" : bank.gain < .02 ? "default" : "green"}>{bank.gain < 0 ? "Loses" : bank.gain < .02 ? "Neutral" : "Gains"}</Chip></td></tr>)}</tbody></table></div>
      </Glass>
      <div className="banks-bottom">
        <Glass className="scatter-card"><div className="card-top"><div><span className="section-label">Prevalence vs federation gain</span><span className="sub-label">Each dot is one simulated bank</span></div></div><ScatterPlot /><div className="chart-note">The lowest-prevalence participant sees the largest gain in this run, but five banks are insufficient to infer a general relationship.</div></Glass>
        <Glass className="noniid-card"><span className="section-label">Non-IID check</span><div className="noniid-score"><div><span>Distribution shift</span><b>Moderate</b></div><div className="ring-score">0.42<small>JS div.</small></div></div>{[["Label prevalence",72],["Transaction type",58],["Amount distribution",43]].map(([label,value]) => <div className="dist-row" key={String(label)}><span>{label}</span><div><i style={{width:`${value}%`}} /></div><b>{Number(value)/100}</b></div>)}<div className="amber-note">Feature distributions differ materially. Aggregate metrics should be read alongside per-bank outcomes.</div></Glass>
      </div>
      {selected && <BankDrawer bank={bankData[selected - 1]} onClose={() => setSelected(null)} />}
    </>
  );
}

function ScatterPlot() {
  return <div className="scatter"><div className="scatter-y">Federation gain</div><svg viewBox="0 0 600 210">{[30,75,120,165].map(y=><line key={y} x1="45" y1={y} x2="580" y2={y} className="chart-grid" />)}<line x1="45" y1="172" x2="580" y2="172" className="zero-line" />{[[150,51],[275,67],[72,28],[220,83],[480,185]].map(([x,y],i)=><g key={i} className={i===4?"scatter-dot loss":"scatter-dot"}><circle cx={x} cy={y} r="8" /><text x={x+12} y={y+4}>B{i+1}</text></g>)}</svg><div className="scatter-x">Fraud prevalence →</div></div>;
}

function BankDrawer({ bank, onClose }: { bank: typeof bankData[number]; onClose: () => void }) {
  return <div className="drawer-backdrop" onClick={onClose}><aside className="drawer" onClick={(e)=>e.stopPropagation()}><div className="drawer-head"><div><span className="eyebrow">Participant detail</span><div className="drawer-title">Bank {bank.id}</div></div><button className="icon-button" onClick={onClose}><Icon name="close" /></button></div><div className="drawer-stats"><div><span>Dataset size</span><b>{bank.samples}</b></div><div><span>Fraud ratio</span><b>{bank.prevalence}%</b></div><div><span>PR-AUC gain</span><b className={bank.gain < 0 ? "negative":"positive"}>{bank.gain>0?"+":""}{bank.gain.toFixed(3)}</b></div></div><Glass className="pr-card"><span className="section-label">Precision-recall curve</span><svg viewBox="0 0 400 190"><path className="chart-grid" d="M35 20V165H380M35 55H380M35 95H380M35 135H380" /><path className="pr-local" d="M35 28 C90 35 125 50 170 74 S270 125 375 157" /><path className="pr-fed" d="M35 22 C100 26 145 39 195 61 S290 110 375 154" /></svg><div className="bar-key"><span><i className="local-key"/>Local-only</span><span><i className="fed-key"/>Shared</span></div></Glass><Glass className="matrix-card"><span className="section-label">Confusion matrix · 1% FPR</span><div className="matrix"><div className="dim"></div><span className="dim">Pred. fraud</span><span className="dim">Pred. clean</span><span className="dim">Actual fraud</span><b className="tp">418</b><b>167</b><span className="dim">Actual clean</span><b>2,263</b><b className="tn">223,602</b></div></Glass><div className="drawer-note"><Icon name="alert" size={16}/> Placeholder confusion matrix scaled to dataset size.</div></aside></div>;
}

function TrainingRounds() {
  const [running, setRunning] = useState(true);
  const [round, setRound] = useState(12);
  useEffect(() => { if (!running) return; const timer = window.setInterval(() => setRound((r) => r >= 20 ? 1 : r + 1), 4500); return () => clearInterval(timer); }, [running]);
  return (
    <>
      <PageHeader eyebrow="Live simulation" title="Training rounds" description="Observe local computation, bounded messages, and weighted model aggregation." />
      <Glass className="timeline-card"><div className="round-labels"><span>Round progression</span><b>{round} / 20</b></div><div className="timeline">{Array.from({length:20},(_,i)=>i+1).map(n=><button key={n} onClick={()=>setRound(n)} className={n===round?"current":n<round?"done":""}><i>{n<round?<Icon name="check" size={11}/>:n}</i>{n===round&&<span>ACTIVE</span>}</button>)}</div></Glass>
      <div className="training-grid">
        <div className="bank-training">{bankData.slice(0,3).map((bank,i)=><Glass className="training-bank" key={bank.id}><div className="training-bank-top"><span className="bank-mark">{bank.id}</span><div><b>Bank {bank.id}</b><span>{bank.samples} samples</span></div><Chip tone={i===0?"cyan":i===1?"violet":"green"}>{i===0?"Local training":i===1?"Sending update":"Uploaded"}</Chip></div><svg className="loss-spark" viewBox="0 0 240 45"><path d={`M2 ${8+i*3} C35 ${18+i*2} 55 22 80 25 S125 30 150 34 S195 37 238 39`} /></svg><div className="param-preview"><span>Δθ preview</span><code>[+0.0021, −0.0087, +0.0142, …]</code></div></Glass>)}</div>
        <Glass className="aggregation"><div className="agg-orbit"><span className="orb o1">Δθ1</span><span className="orb o2">Δθ2</span><span className="orb o3">Δθ3</span><div className="agg-core"><Icon name="rounds" size={28}/><b>FedAvg</b><small>weighted merge</small></div></div><div className="agg-formula">θ<tspan>t+1</tspan> = Σ (n<sub>k</sub> / n) θ<sub>k</sub></div><p>Updates merge into a new shared model. No training rows are present.</p><Chip tone="violet">Model v{round}.0 ready</Chip></Glass>
        <Glass className="comms"><span className="section-label">Communication</span>{[["Bytes / round","206 KB",68],["Total sent","2.41 MB",81],["Round duration","8.4 sec",52]].map(([label,val,width])=><div className="comm-stat" key={String(label)}><div><span>{label}</span><b>{val}</b></div><div><i style={{width:`${width}%`}}/></div></div>)}<div className="protocol"><span>Protocol</span><b>TLS 1.3 · JSON mock</b><span>Payload</span><b>weights + metadata</b></div></Glass>
      </div>
      <Glass className="terminal"><div className="terminal-top"><div><i/><i/><i/><span>coordinator.log</span></div><div className="controls"><button onClick={()=>setRunning(!running)}><Icon name={running?"pause":"play"} size={15}/>{running?"Pause":"Start"}</button><button onClick={()=>setRound(1)}><Icon name="reset" size={15}/>Reset</button><label>Speed<input type="range" min="1" max="3" defaultValue="2"/></label></div></div><div className="logs"><span><i>14:32:08.142</i> Round {round} — Bank 1: local training done <b>loss=0.0841</b></span><span><i>14:32:08.418</i> Round {round} — Bank 2: update sent <b>41.2 KB</b></span><span><i>14:32:08.631</i> Coordinator: schema check <b className="positive">PASS</b> · raw_rows=0 · account_ids=0</span><span className="active-log"><i>14:32:09.004</i> Coordinator: FedAvg aggregation in progress<span className="cursor">_</span></span></div></Glass>
    </>
  );
}

function Audit() {
  const ledger = [
    ["14:32:08.418","Bank 2","12","model_update","41.2 KB"],
    ["14:32:08.632","Bank 1","12","metadata","0.8 KB"],
    ["14:32:07.994","Bank 5","12","model_update","41.3 KB"],
    ["14:31:59.201","Bank 3","11","model_update","41.1 KB"],
  ];
  return (
    <>
      <PageHeader eyebrow="Evidence trail" title="Data boundary audit" description="Inspect exactly what remains within each bank and what reaches the coordinator." />
      <div className="boundary-grid">
        <Glass className="boundary-card local-boundary"><div className="boundary-title"><span><Icon name="lock"/></span><div><b>Stays local</b><small>Never included in outbound messages</small></div></div>{["Raw transaction rows","nameOrig account ID","nameDest account ID","Train / validation / test rows"].map(item=><div className="boundary-item" key={item}><Icon name="lock" size={15}/><span>{item}</span><Chip tone="cyan">BANK ONLY</Chip></div>)}</Glass>
        <Glass className="boundary-card crosses"><div className="boundary-title"><span><Icon name="arrow"/></span><div><b>Crosses the boundary</b><small>Minimal coordinator payload</small></div></div>{["Parameter update tensors","Round number","Sample count for weighting","Protocol status"].map((item,i)=><div className="boundary-item" key={item}><Icon name="arrow" size={15}/><span>{item}</span><Chip tone={i===0?"violet":"default"}>{i===0?"MODEL UPDATE":"METADATA"}</Chip></div>)}</Glass>
      </div>
      <Glass className="table-card ledger"><div className="card-top"><div><span className="section-label">Outbound message ledger</span><span className="sub-label">Coordinator ingress · latest events</span></div><Chip tone="green">Schema monitor active</Chip></div><div className="table-wrap"><table><thead><tr><th>Timestamp</th><th>Participant</th><th>Round</th><th>Payload type</th><th>Size</th><th>Schema</th><th>Raw rows?</th><th>Account IDs?</th></tr></thead><tbody>{ledger.map((r,i)=><tr key={i}>{r.map((v,j)=><td key={j} className={j===0||j===2||j===4?"mono":""}>{v}</td>)}<td><Chip tone="green">PASS</Chip></td><td><Chip tone="green">No</Chip></td><td><Chip tone="green">No</Chip></td></tr>)}</tbody></table></div></Glass>
      <div className="audit-bottom">
        <Glass className="tests"><div className="card-top"><div><span className="section-label">Automated boundary tests</span><span className="sub-label">pytest · 8 tests collected</span></div><b className="positive">8 passed</b></div><div className="test-grid">{["rejects raw dataframe","blocks nameOrig","blocks nameDest","allows model_update","requires round metadata","validates tensor schema"].map(test=><div key={test}><span className="check-circle"><Icon name="check" size={12}/></span><code>{test}</code><span>PASS</span></div>)}</div><div className="zero-result"><span>Raw rows detected in coordinator payloads</span><b>0</b><small>application-level audit</small></div></Glass>
        <Glass className="caveat"><Icon name="alert"/><div><span className="section-label">Important caveat</span><b>This is not a formal privacy guarantee.</b><p>This audit checks application-level payloads. Model updates can still leak information through inference attacks.</p><div><Chip tone="amber">Future work</Chip><span>Differential privacy</span><span>Secure aggregation</span></div></div></Glass>
      </div>
    </>
  );
}

function Limitations() {
  const columns = [
    {title:"Core (implemented)",tone:"green" as const,intro:"Demonstrated in this prototype",items:["Federated fraud detection across simulated banks","Local-only baselines","Per-bank evaluation","Application-level payload audit"]},
    {title:"Research extensions (planned)",tone:"violet" as const,intro:"Required for a stronger evaluation",items:["Deeper non-IID analysis","Rare-fraud performance","Privacy-attack evaluation","Poisoning resistance","Model explainability"]},
    {title:"Not claimed",tone:"rose" as const,intro:"Outside current evidence",items:["Cross-bank account linking","Real fraud-ring detection","Regulatory compliance","Real-world performance"]},
  ];
  return (
    <>
      <PageHeader eyebrow="Scientific honesty" title="Limitations and scope" description="A clear boundary between what the experiment shows, what comes next, and what is not claimed." />
      <div className="scope-grid">{columns.map((col,i)=><Glass className={`scope-card scope-${col.tone}`} key={col.title}><div className="scope-number">0{i+1}</div><Chip tone={col.tone}>{col.title}</Chip><p>{col.intro}</p><div className="scope-list">{col.items.map(item=><div key={item}><span>{col.tone==="green"?<Icon name="check" size={14}/>:col.tone==="rose"?<Icon name="close" size={14}/>:<Icon name="arrow" size={14}/>}</span>{item}</div>)}</div></Glass>)}</div>
      <Glass className="precedent"><div className="precedent-mark">S</div><div><span className="eyebrow">Industry precedent</span><div className="precedent-title">SWIFT 2024–2025 federated learning pilots</div><p>Used as motivation for exploring cross-institution collaboration—not as validation of this prototype or its performance.</p></div><Chip tone="amber">Context, not evidence</Chip></Glass>
      <div className="principle-row"><Glass><Icon name="shield"/><div><b>Evidence over claims</b><span>Every result links to an observable experiment surface.</span></div></Glass><Glass><Icon name="bank"/><div><b>Individual outcomes matter</b><span>Local regressions remain visible, not averaged away.</span></div></Glass><Glass><Icon name="alert"/><div><b>Privacy is a research problem</b><span>Data locality alone is not a complete guarantee.</span></div></Glass></div>
    </>
  );
}

export default function App() {
  const [screen, setScreen] = useState<Screen>("overview");
  const active = useMemo(() => screens.find((item) => item.id === screen)!, [screen]);
  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand-mark"><svg viewBox="0 0 32 32"><path d="M16 2 28 7v8c0 7.5-5.1 12.7-12 15-6.9-2.3-12-7.5-12-15V7l12-5Z"/><path d="M10 12h12M10 16h9M10 20h6"/></svg></div>
        <nav>{screens.map((item)=><button key={item.id} className={screen===item.id?"active":""} onClick={()=>setScreen(item.id)} aria-label={item.label}><Icon name={item.icon}/><span>{item.label}</span></button>)}</nav>
        <div className="side-bottom"><span className="secure-pulse"/><small>LOCAL<br/>SECURE</small></div>
      </aside>
      <div className="app-main">
        <header className="topbar">
          <div className="wordmark"><b>Fed<span>Shield</span></b><i>Research Console</i></div>
          <div className="topbar-center"><div className="experiment-select"><span>Experiment</span><select defaultValue="run1"><option value="run1">Run #1 — seed 42</option><option value="run2">Run #2 — seed 17</option></select></div><div className="round-counter"><span>Round</span><b>12 <i>/ 20</i></b></div></div>
          <Chip tone="cyan"><span className="status-dot"/>Coordinator: neutral · receives updates only</Chip>
        </header>
        <main key={active.id}>
          {screen === "overview" && <Overview />}
          {screen === "curve" && <ParticipationCurve />}
          {screen === "banks" && <BankResults />}
          {screen === "rounds" && <TrainingRounds />}
          {screen === "audit" && <Audit />}
          {screen === "limits" && <Limitations />}
        </main>
      </div>
    </div>
  );
}
