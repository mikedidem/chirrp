import { Component, OnInit, computed, signal } from '@angular/core';
import { CommonModule, DecimalPipe } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { PlotlyChart } from '../../components/plotly-chart/plotly-chart';
import {
  GoalSeekResponse,
  PinnMeta,
  PinnService,
} from '../../services/pinn.service';

@Component({
  selector: 'app-goal-seek',
  imports: [CommonModule, FormsModule, DecimalPipe, PlotlyChart],
  templateUrl: './goal-seek.html',
  styleUrl: './goal-seek.css',
})
export class GoalSeek implements OnInit {
  meta = signal<PinnMeta | null>(null);
  result = signal<GoalSeekResponse | null>(null);
  busy = signal(false);
  error = signal<string | null>(null);

  // Defaults: the pumping well, end of simulation.
  x = 201.67;
  y = -98.33;
  maxDrawdown: number | null = 6;
  t: number | null = 30;
  verify = false;

  constructor(private pinn: PinnService) {}

  ngOnInit(): void {
    this.pinn.getMeta().subscribe({
      next: (m) => this.meta.set(m),
      error: () => {},
    });
  }

  run(): void {
    if (this.busy() || this.maxDrawdown === null) return;
    this.busy.set(true);
    this.error.set(null);
    this.pinn
      .goalSeek({
        x: this.x,
        y: this.y,
        max_drawdown_m: this.maxDrawdown,
        t: this.t ?? undefined,
        verify_with_modflow: this.verify,
      })
      .subscribe({
        next: (r) => {
          this.busy.set(false);
          this.result.set(r);
        },
        error: (err) => {
          this.busy.set(false);
          const detail = err?.error?.detail;
          this.error.set(
            typeof detail === 'string'
              ? detail
              : err?.status === 0
                ? 'Cannot reach the backend.'
                : `Goal seek failed (HTTP ${err?.status ?? '?'}).`,
          );
        },
      });
  }

  /** Where the recommended rate sits relative to the trained envelope. */
  envelopeStatus = computed<
    { label: string; tone: 'ok' | 'warn' | 'danger' } | null
  >(() => {
    const r = this.result();
    const m = this.meta();
    if (!r || !m || !r.feasible) return null;
    const [lo, hi] = m.percent_bounds;
    const pc = r.best_percent_change;
    if (pc < lo - 1e-6 || pc > hi + 1e-6) return { label: 'Outside trained range', tone: 'danger' };
    const span = hi - lo;
    const margin = Math.min(pc - lo, hi - pc);
    if (span > 0 && margin < span * 0.12) return { label: 'Near envelope edge', tone: 'warn' };
    return { label: 'Within trained envelope', tone: 'ok' };
  });

  /** Confidence from how deep inside the trained range the answer sits
   *  (a real validity margin, not a placeholder). */
  confidence = computed<
    { pct: number; label: string; tone: 'ok' | 'warn' | 'danger' } | null
  >(() => {
    const r = this.result();
    const m = this.meta();
    if (!r || !m || !r.feasible) return null;
    const [lo, hi] = m.percent_bounds;
    const pc = r.best_percent_change;
    const half = (hi - lo) / 2 || 1;
    const margin = Math.min(pc - lo, hi - pc);
    const pct = Math.round(Math.max(0, Math.min(1, margin / half)) * 100);
    const es = this.envelopeStatus();
    if (es?.tone === 'warn') return { pct, label: 'Moderate', tone: 'warn' };
    if (es?.tone === 'danger') return { pct, label: 'Low', tone: 'danger' };
    return { pct, label: 'High', tone: 'ok' };
  });

  /** One-line stakeholder takeaway, validity-aware. */
  takeaway = computed<{ text: string; tone: 'ok' | 'warn' | 'danger' } | null>(() => {
    const r = this.result();
    if (!r) return null;
    if (!r.feasible) {
      return {
        text: 'No admissible pumping rate satisfies this drawdown constraint within the trained envelope. The closest achievable drawdown is shown below.',
        tone: 'danger',
      };
    }
    const q = Math.round(r.best_q_rate).toLocaleString();
    const limit = r.constraint.max_drawdown_m;
    if (this.envelopeStatus()?.tone === 'warn') {
      return {
        text: `The maximum pumping rate satisfying the ${limit} m drawdown limit here is Q = ${q} m³/day, but it is near the trained envelope boundary and should be interpreted with caution.`,
        tone: 'warn',
      };
    }
    return {
      text: `The maximum pumping rate that satisfies the ${limit} m drawdown limit at this location is Q = ${q} m³/day, and the result remains within the trained envelope.`,
      tone: 'ok',
    };
  });

  /** Short "what it means for planning" sentence for the readout. */
  planningMeaning = computed<string>(() => {
    const r = this.result();
    if (!r) return '';
    if (!r.feasible) {
      return 'Consider allowing a larger drawdown, choosing a different point of interest, or accepting a shorter pumping duration.';
    }
    return this.envelopeStatus()?.tone === 'warn'
      ? 'Usable as a planning limit, but it sits near the model’s validated boundary — confirm with MODFLOW before relying on it.'
      : 'This rate keeps drawdown within the limit and lies inside the model’s validated range — suitable for planning use.';
  });

  curve = computed(() => {
    const r = this.result();
    if (!r) return null;
    return {
      data: [
        {
          type: 'scatter',
          mode: 'lines',
          x: r.curve.percent_changes,
          y: r.curve.drawdowns,
          line: { color: '#0f766e', width: 2.5 },
          name: 'predicted drawdown',
          hovertemplate: 'Δ %{x:.1f}% · drawdown %{y:.3f} m<extra></extra>',
        },
        {
          type: 'scatter',
          mode: 'lines',
          x: [
            Math.min(...r.curve.percent_changes),
            Math.max(...r.curve.percent_changes),
          ],
          y: [r.constraint.max_drawdown_m, r.constraint.max_drawdown_m],
          line: { color: '#b91c1c', width: 1.5, dash: 'dot' },
          name: 'constraint',
          hoverinfo: 'skip',
        },
        ...(r.feasible
          ? [
              {
                type: 'scatter',
                mode: 'markers',
                x: [r.best_percent_change],
                y: [r.predicted_drawdown],
                marker: { size: 12, color: '#b45309', symbol: 'star' },
                name: 'recommended',
                hovertemplate:
                  'recommended Δ %{x:.1f}% · %{y:.3f} m<extra></extra>',
              },
            ]
          : []),
      ],
      layout: {
        title: {
          text: `Drawdown at (${r.constraint.x.toFixed(0)}, ${r.constraint.y.toFixed(0)}) m, t = ${r.constraint.t} d — vs pumping change`,
          font: { size: 13 },
        },
        xaxis: { title: { text: 'pumping change (%)' } },
        yaxis: { title: { text: 'drawdown (m)' } },
        height: 380,
        legend: { orientation: 'h', y: -0.22 },
      },
    };
  });
}
