import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { environment } from '../../environments/environment';

/* ---------- Models ---------- */

export interface PinnMeta {
  domain: { x: number[]; y: number[]; t: number[] };
  well_xy: number[];
  initial_head: number;
  q_min: number;
  q_max: number;
  q_baseline: number;
  percent_bounds: number[];
  reference_anchors: number[];
  model: { type: string; inputs: string; hidden: string; constraint: string };
}

export interface SimulationResult {
  q_rate: number;
  percent_change: number;
  times: number[];
  x: number[];
  y: number[];
  heads: number[][][];
  head_min: number;
  head_max: number;
  max_drawdown: number;
  well_xy: number[];
  well_drawdown_series: number[];
  latency_ms: number;
  initial_head: number;
  saved_as?: string | null;
}

export interface ParseResult {
  is_valid: boolean;
  percent_change: number | null;
  error: string;
  suggestion: string;
  source: string;
  llm_error: string;
  latency_ms: number;
}

export interface ChatLatency {
  llm_parse_ms: number;
  pinn_ms?: number;
  summary_ms?: number;
  total_ms: number;
}

export interface ChatResponse {
  is_valid: boolean;
  parse: ParseResult;
  simulation?: SimulationResult;
  summary?: string;
  summary_source?: string;
  error?: string;
  suggestion?: string;
  latency: ChatLatency;
  saved_as?: string | null;
}

export interface ProbeResult {
  q_rate: number;
  points: { x: number; y: number; t: number; head: number; drawdown: number }[];
  latency_ms: number;
}

export interface ValidationMetrics {
  rmse_m: number;
  mae_m: number;
  max_abs_error_m: number;
  rrmse_pct: number;
  r2: number;
}

export interface ValidateResponse {
  q_rate: number;
  percent_change: number;
  precomputed: boolean;
  times: number[];
  x: number[];
  y: number[];
  pinn_heads_final: number[][];
  modflow_heads_final: number[][];
  error_field_final: number[][];
  metrics: ValidationMetrics;
  rmse_per_time: number[];
  well_drawdown: { pinn: number[]; modflow: number[] };
  latency: { pinn_ms: number; modflow_s: number | null; speedup: number | null };
}

export interface GoalSeekResponse {
  feasible: boolean;
  best_q_rate: number;
  best_percent_change: number;
  predicted_drawdown: number;
  constraint: { x: number; y: number; max_drawdown_m: number; t: number };
  curve: { q_rates: number[]; percent_changes: number[]; drawdowns: number[] };
  n_evaluations: number;
  latency_ms: number;
  modflow_verification?: {
    drawdown_m: number;
    satisfies_constraint: boolean;
    runtime_s: number;
  };
}

export interface ScenarioSummary {
  scenario_name: string;
  instruction: string | null;
  percent_change: number;
  q_rate: number;
  max_drawdown: number | null;
  well_drawdown_final: number | null;
  latency: ChatLatency | null;
  created_at: string | null;
}

export interface SavedScenario extends ScenarioSummary {
  head_min: number | null;
  head_max: number | null;
  head_grid: number[][] | null;
  grid_meta: {
    x: number[]; y: number[]; t: number; resolution: number; well_xy: number[];
  } | null;
  summary_text: string | null;
}

export interface CompareScenario {
  label: string;
  percent_change: number;
  q_rate: number;
  times: number[];
  well_drawdown_series: number[];
  max_drawdown: number;
  head_min: number;
  well_drawdown_final: number;
  latency_ms: number;
}

export interface CompareDelta {
  label: string;
  d_max_drawdown: number;
  d_head_min: number;
  d_well_drawdown_final: number;
}

export interface CompareResponse {
  baseline_label: string;
  scenarios: CompareScenario[];
  deltas: CompareDelta[];
}

export interface CompareItem {
  label?: string;
  percent_change?: number;
  q_rate?: number;
  scenario_name?: string;
}

/* ---------- Service ---------- */

@Injectable({ providedIn: 'root' })
export class PinnService {
  private api = `${environment.apiUrl}/pinn`;

  constructor(private http: HttpClient) {}

  getMeta() {
    return this.http.get<PinnMeta>(`${this.api}/meta`);
  }

  simulate(body: {
    percent_change?: number;
    q_rate?: number;
    scenario_name?: string;
    instruction?: string;
    resolution?: number;
  }) {
    return this.http.post<SimulationResult>(`${this.api}/simulate`, body);
  }

  chat(instruction: string, scenarioName?: string, sessionId?: string) {
    return this.http.post<ChatResponse>(`${this.api}/chat`, {
      instruction,
      scenario_name: scenarioName || null,
      session_id: sessionId || null,
    });
  }

  probe(qRate: number, points: { x: number; y: number; t: number }[]) {
    return this.http.post<ProbeResult>(`${this.api}/probe`, {
      q_rate: qRate,
      points,
    });
  }

  validate(body: { q_rate?: number; percent_change?: number; live?: boolean }) {
    return this.http.post<ValidateResponse>(`${this.api}/validate`, body);
  }

  goalSeek(body: {
    x: number;
    y: number;
    max_drawdown_m: number;
    t?: number;
    verify_with_modflow?: boolean;
  }) {
    return this.http.post<GoalSeekResponse>(`${this.api}/goal-seek`, body);
  }

  compare(items: CompareItem[], includeGrids = false) {
    return this.http.post<CompareResponse>(`${this.api}/compare`, {
      items,
      include_grids: includeGrids,
    });
  }

  listScenarios() {
    return this.http.get<ScenarioSummary[]>(`${this.api}/scenarios`);
  }

  getScenario(name: string) {
    return this.http.get<SavedScenario>(
      `${this.api}/scenarios/${encodeURIComponent(name)}`);
  }

  deleteScenario(name: string) {
    return this.http.delete<{ deleted: string }>(
      `${this.api}/scenarios/${encodeURIComponent(name)}`);
  }
}
