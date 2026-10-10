"use client";

import React, { createContext, useContext, useEffect, useState, useCallback } from "react";
import {
  BankMetric,
  MacroMetric,
  PerBankMetric,
  ComparisonResponse,
  AuditResponse,
  AuditLedgerRow,
  CommunicationMetric,
  LayaHealthResponse,
  TriageRequest,
  TriageResponse,
  DEFAULT_BANKS,
  DEFAULT_METRICS,
  DEFAULT_MACRO_RESULTS,
  DEFAULT_AUDIT,
  DEFAULT_COMMUNICATION,
  DEFAULT_AUDIT_LEDGER,
  DEFAULT_EXPERIMENT_STATUS,
  ExperimentStatusResponse,
  getHealth,
  getBanks,
  getMetricsByParticipation,
  getMacroResults,
  getPerBankResults,
  getComparison,
  getAudit,
  getAuditLedger,
  getCommunicationMetrics,
  getLayaHealth,
  getExperimentStatus,
  triageTransaction,
} from "./api";

interface DataContextType {
  bankData: BankMetric[];
  metricsByParticipation: Record<number, number[]>;
  macroResults: MacroMetric[];
  perBankResults: PerBankMetric[];
  comparison: ComparisonResponse | null;
  auditData: AuditResponse;
  auditLedger: AuditLedgerRow[];
  communicationMetrics: CommunicationMetric[];
  layaHealth: LayaHealthResponse | null;
  experimentStatus: ExperimentStatusResponse | null;
  isBackendConnected: boolean;
  isLoading: boolean;
  error: string | null;
  lastSynced: Date | null;
  refetchAll: () => Promise<void>;
  runTriage: (req: TriageRequest) => Promise<TriageResponse>;
}

const DataContext = createContext<DataContextType | null>(null);

export function DataProvider({ children }: { children: React.ReactNode }) {
  const [bankData, setBankData] = useState<BankMetric[]>(DEFAULT_BANKS);
  const [metricsByParticipation, setMetricsByParticipation] = useState<Record<number, number[]>>(DEFAULT_METRICS);
  const [macroResults, setMacroResults] = useState<MacroMetric[]>(DEFAULT_MACRO_RESULTS);
  const [perBankResults, setPerBankResults] = useState<PerBankMetric[]>([]);
  const [comparison, setComparison] = useState<ComparisonResponse | null>(null);
  const [auditData, setAuditData] = useState<AuditResponse>(DEFAULT_AUDIT);
  const [auditLedger, setAuditLedger] = useState<AuditLedgerRow[]>(DEFAULT_AUDIT_LEDGER);
  const [communicationMetrics, setCommunicationMetrics] = useState<CommunicationMetric[]>(DEFAULT_COMMUNICATION);
  const [layaHealth, setLayaHealth] = useState<LayaHealthResponse | null>(null);
  const [experimentStatus, setExperimentStatus] = useState<ExperimentStatusResponse | null>(DEFAULT_EXPERIMENT_STATUS);

  const [isBackendConnected, setIsBackendConnected] = useState<boolean>(false);
  const [isLoading, setIsLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const [lastSynced, setLastSynced] = useState<Date | null>(null);

  const refetchAll = useCallback(async () => {
    setIsLoading(true);
    setError(null);
    try {
      // 1. Health check
      const health = await getHealth();
      const connected = health.status === "ok";
      setIsBackendConnected(connected);

      // 2. Parallel fetch all endpoints
      const [
        banks,
        metrics,
        macro,
        perBank,
        comp,
        audit,
        ledger,
        comms,
        laya,
        expStatus,
      ] = await Promise.all([
        getBanks(),
        getMetricsByParticipation(),
        getMacroResults(),
        getPerBankResults(),
        getComparison(),
        getAudit(),
        getAuditLedger(),
        getCommunicationMetrics(),
        getLayaHealth(),
        getExperimentStatus(),
      ]);

      if (banks && banks.length > 0) setBankData(banks);
      if (metrics) setMetricsByParticipation(metrics);
      if (macro && macro.length > 0) setMacroResults(macro);
      if (perBank && perBank.length > 0) setPerBankResults(perBank);
      if (comp) setComparison(comp);
      if (audit) setAuditData(audit);
      if (ledger && ledger.length > 0) setAuditLedger(ledger);
      if (comms && comms.length > 0) setCommunicationMetrics(comms);
      if (laya) setLayaHealth(laya);
      if (expStatus) setExperimentStatus(expStatus);

      setLastSynced(new Date());
    } catch (err) {
      console.warn("Failed to sync with backend API:", err);
      setIsBackendConnected(false);
      setError("The backend could not be reached. Showing the latest cached results.");
    } finally {
      setIsLoading(false);
    }
  }, []);

  useEffect(() => {
    refetchAll();
    // Auto-sync every 30 seconds
    const interval = setInterval(refetchAll, 30000);
    return () => clearInterval(interval);
  }, [refetchAll]);

  const runTriage = useCallback(async (req: TriageRequest): Promise<TriageResponse> => {
    return await triageTransaction(req);
  }, []);

  return (
    <DataContext.Provider
      value={{
        bankData,
        metricsByParticipation,
        macroResults,
        perBankResults,
        comparison,
        auditData,
        auditLedger,
        communicationMetrics,
        layaHealth,
        experimentStatus,
        isBackendConnected,
        isLoading,
        error,
        lastSynced,
        refetchAll,
        runTriage,
      }}
    >
      {children}
    </DataContext.Provider>
  );
}

export function useData(): DataContextType {
  const context = useContext(DataContext);
  if (!context) {
    // Provide sensible default fallback so components still work outside provider
    return {
      bankData: DEFAULT_BANKS,
      metricsByParticipation: DEFAULT_METRICS,
      macroResults: DEFAULT_MACRO_RESULTS,
      perBankResults: [],
      comparison: null,
      auditData: DEFAULT_AUDIT,
      auditLedger: DEFAULT_AUDIT_LEDGER,
      communicationMetrics: DEFAULT_COMMUNICATION,
      layaHealth: null,
      experimentStatus: DEFAULT_EXPERIMENT_STATUS,
      isBackendConnected: false,
      isLoading: false,
      error: null,
      lastSynced: null,
      refetchAll: async () => {},
      runTriage: triageTransaction,
    };
  }
  return context;
}
