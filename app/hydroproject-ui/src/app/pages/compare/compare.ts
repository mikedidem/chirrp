import { Component, OnInit, computed, signal } from '@angular/core';
import { CommonModule, DecimalPipe } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { PlotlyChart } from '../../components/plotly-chart/plotly-chart';
import {
  CompareItem,
  CompareResponse,
  PinnMeta,
  PinnService,
  ScenarioSummary,
} from '../../services/pinn.service';

const PALETTE = ['#0f766e', '#b45309', '#1d4ed8', '#9333ea'];

@Component({
  selector: 'app-compare',
  imports: [CommonModule, FormsModule, DecimalPipe, PlotlyChart],
  templateUrl: './compare.html',
  styleUrl: './compare.css',
})
export class Compare implements OnInit {
  scenarios = signal<ScenarioSummary[]>([]);
  selected = signal<Set<string>>(new Set());
  adhoc = '';                       // comma-separated percent values
  busy = signal(false);
  error = signal<string | null>(null);
  result = signal<CompareResponse | null>(null);
  meta = signal<PinnMeta | null>(null);

  constructor(private pinn: PinnService) {}

  ngOnInit(): void {
    this.pinn.listScenarios().subscribe({
      next: (s) => this.scenarios.set(s),
      error: () => this.scenarios.set([]),
    });
    this.pinn.getMeta().subscribe({
      next: (m) => this.meta.set(m),
      error: () => {},
    });
  }

  /** Classify a scenario against the trained envelope (results are never
   *  out-of-range — the backend rejects those — so this is within/near-edge). */
  envClass(pc: number): { kind: 'within' | 'near'; label: string } {
    const m = this.meta();
    if (!m) return { kind: 'within', label: 'in range' };
    const [lo, hi] = m.percent_bounds;
    const span = hi - lo;
    const margin = Math.min(pc - lo, hi - pc);
    if (span > 0 && margin < span * 0.12) return { kind: 'near', label: 'near envelope edge' };
    return { kind: 'within', label: 'within envelope' };
  }

  /** Validity-aware stakeholder takeaway. The least-drawdown option is only
   *  framed as the recommendation when it is also inside the trained envelope. */
  takeaway = computed<{ text: string; tone: 'ok' | 'warn' } | null>(() => {
    const r = this.result();
    if (!r || !r.scenarios.length) return null;
    const best = r.scenarios.reduce((a, b) => (b.max_drawdown < a.max_drawdown ? b : a));
    const env = this.envClass(best.percent_change);
    if (env.kind === 'within') {
      return {
        text: `“${best.label}” keeps drawdown lowest at ${best.max_drawdown.toFixed(2)} m while remaining inside the trained scenario envelope.`,
        tone: 'ok',
      };
    }
    return {
      text: `“${best.label}” has the lowest drawdown (${best.max_drawdown.toFixed(2)} m) but sits near the edge of the trained envelope — interpret with caution and verify before relying on it.`,
      tone: 'warn',
    };
  });

  /** Human-recognizable label for a saved scenario: its instruction if present,
   *  else the percent change. Falls back to the raw name only as a last resort. */
  friendlyLabel(sc: ScenarioSummary): string {
    if (sc.instruction && sc.instruction.trim()) {
      const t = sc.instruction.trim();
      return t.length > 48 ? t.slice(0, 45) + '…' : t;
    }
    if (sc.percent_change != null) {
      return `${sc.percent_change > 0 ? '+' : ''}${sc.percent_change.toFixed(1)}% pumping`;
    }
    return sc.scenario_name;
  }

  toggle(name: string): void {
    const next = new Set(this.selected());
    if (next.has(name)) next.delete(name);
    else next.add(name);
    this.selected.set(next);
  }

  isSelected(name: string): boolean {
    return this.selected().has(name);
  }

  /** Build the items array from picked saved scenarios + ad-hoc percentages. */
  private buildItems(): CompareItem[] {
    const items: CompareItem[] = [];
    for (const name of this.selected()) {
      const sc = this.scenarios().find((s) => s.scenario_name === name);
      // Pass a readable label so the chart legend / table aren't cryptic names.
      items.push({ scenario_name: name, label: sc ? this.friendlyLabel(sc) : name });
    }
    for (const tok of this.adhoc.split(',')) {
      const v = parseFloat(tok.trim());
      if (!Number.isNaN(v)) items.push({ percent_change: v });
    }
    return items;
  }

  itemCount = computed(() => this.selected().size +
    this.adhoc.split(',').filter((t) => !Number.isNaN(parseFloat(t.trim()))).length);

  runCompare(): void {
    const items = this.buildItems();
    if (items.length < 2) {
      this.error.set('Pick at least 2 scenarios (saved or ad-hoc %) to compare.');
      return;
    }
    if (items.length > 4) {
      this.error.set('Compare at most 4 scenarios at a time.');
      return;
    }
    this.error.set(null);
    this.busy.set(true);
    this.pinn.compare(items).subscribe({
      next: (r) => {
        this.result.set(r);
        this.busy.set(false);
      },
      error: (err) => {
        this.busy.set(false);
        this.error.set(err?.error?.detail || 'Comparison failed.');
      },
    });
  }

  /** The least-impact scenario (smallest maximum drawdown) — a decision cue. */
  bestLabel = computed(() => {
    const r = this.result();
    if (!r || !r.scenarios.length) return null;
    return r.scenarios.reduce((a, b) => (b.max_drawdown < a.max_drawdown ? b : a)).label;
  });

  /** Overlaid drawdown-at-well series, one line per scenario. */
  overlayData = computed(() => {
    const r = this.result();
    if (!r) return [];
    return r.scenarios.map((s, i) => ({
      type: 'scatter',
      mode: 'lines',
      x: s.times,
      y: s.well_drawdown_series,
      name: s.label,
      line: { color: PALETTE[i % PALETTE.length], width: 2.5 },
      hovertemplate: `${s.label}<br>t %{x} d · %{y:.3f} m<extra></extra>`,
    }));
  });

  overlayLayout = {
    title: { text: 'Drawdown at the pumping well — scenarios overlaid', font: { size: 14 } },
    xaxis: { title: { text: 'time (days)' } },
    yaxis: { title: { text: 'drawdown (m)' } },
    height: 360,
    margin: { l: 55, r: 20, t: 40, b: 45 },
    legend: { orientation: 'h', y: -0.2 },
  };
}
