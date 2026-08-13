import { Component, OnInit, computed, signal } from '@angular/core';
import { CommonModule, DecimalPipe } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { PlotlyChart } from '../../components/plotly-chart/plotly-chart';
import {
  PinnMeta,
  PinnService,
  ValidateResponse,
} from '../../services/pinn.service';

@Component({
  selector: 'app-validate',
  imports: [CommonModule, FormsModule, DecimalPipe, PlotlyChart],
  templateUrl: './validate.html',
  styleUrl: './validate.css',
})
export class Validate implements OnInit {
  meta = signal<PinnMeta | null>(null);
  result = signal<ValidateResponse | null>(null);
  busy = signal(false);
  error = signal<string | null>(null);
  selectedQ = signal<number | null>(null);

  customPercent: number | null = null;
  live = false;

  constructor(private pinn: PinnService) {}

  ngOnInit(): void {
    this.pinn.getMeta().subscribe({
      next: (m) => this.meta.set(m),
      error: () => {},
    });
  }

  runAnchor(q: number): void {
    this.selectedQ.set(q);
    this.run({ q_rate: q, live: this.live });
  }

  runCustom(): void {
    if (this.customPercent === null) return;
    this.selectedQ.set(null);
    this.run({ percent_change: this.customPercent, live: this.live });
  }

  private run(body: { q_rate?: number; percent_change?: number; live: boolean }): void {
    if (this.busy()) return;
    this.busy.set(true);
    this.error.set(null);
    this.pinn.validate(body).subscribe({
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
              : `Validation failed (HTTP ${err?.status ?? '?'}).`,
        );
      },
    });
  }

  /** Validation confidence derived from the REAL metrics of the last run
   *  (R² against MODFLOW, with RRMSE as a tie-breaker). Not a placeholder. */
  reliability = computed<
    { pct: number; label: string; tone: 'ok' | 'warn' | 'danger' } | null
  >(() => {
    const r = this.result();
    if (!r) return null;
    const r2 = r.metrics.r2;
    const pct = Math.max(0, Math.min(100, r2 * 100));
    if (r2 >= 0.97 && r.metrics.rrmse_pct < 5) return { pct, label: 'High', tone: 'ok' };
    if (r2 >= 0.9) return { pct, label: 'Good', tone: 'ok' };
    if (r2 >= 0.75) return { pct, label: 'Moderate', tone: 'warn' };
    return { pct, label: 'Low', tone: 'danger' };
  });

  /** Whether the chosen Q will need a live MODFLOW run (no anchor match). */
  needsLiveRun = computed(() => {
    const m = this.meta();
    if (!m || this.customPercent === null) return false;
    const q = m.q_baseline * (1 + this.customPercent / 100);
    return !m.reference_anchors.some((a) => Math.abs(a - q) <= 1.0);
  });

  private heatmap(z: number[][], title: string, colorscale: string, opts: any = {}) {
    const r = this.result()!;
    return {
      data: [
        {
          type: 'heatmap',
          x: r.x,
          y: r.y,
          z,
          colorscale,
          ...opts,
          colorbar: { thickness: 12, len: 0.9 },
          hovertemplate: 'x %{x:.0f} · y %{y:.0f}<br>%{z:.3f} m<extra></extra>',
        },
      ],
      layout: {
        title: { text: title, font: { size: 13 } },
        xaxis: { title: { text: 'x (m)' }, constrain: 'domain' },
        yaxis: { scaleanchor: 'x', scaleratio: 1 },
        height: 330,
        margin: { l: 45, r: 10, t: 34, b: 40 },
      },
    };
  }

  pinnMap = computed(() => {
    const r = this.result();
    return r ? this.heatmap(r.pinn_heads_final, 'PINN surrogate — head, t = 30 d', 'Viridis') : null;
  });

  modflowMap = computed(() => {
    const r = this.result();
    return r ? this.heatmap(r.modflow_heads_final, 'MODFLOW-2005 — head, t = 30 d', 'Viridis') : null;
  });

  errorMap = computed(() => {
    const r = this.result();
    if (!r) return null;
    const maxAbs = Math.max(
      ...r.error_field_final.map((row) => Math.max(...row.map(Math.abs))),
    );
    return this.heatmap(
      r.error_field_final,
      'Error (PINN − MODFLOW), t = 30 d',
      'RdBu',
      { zmin: -maxAbs, zmax: maxAbs, reversescale: true },
    );
  });

  rmseSeries = computed(() => {
    const r = this.result();
    if (!r) return null;
    return {
      data: [
        {
          type: 'scatter',
          mode: 'lines+markers',
          x: r.times,
          y: r.rmse_per_time,
          line: { color: '#1d4ed8', width: 2 },
          marker: { size: 4 },
          hovertemplate: 't %{x} d · RMSE %{y:.4f} m<extra></extra>',
        },
      ],
      layout: {
        title: { text: 'RMSE vs simulation time', font: { size: 13 } },
        xaxis: { title: { text: 'time (days)' } },
        yaxis: { title: { text: 'RMSE (m)' } },
        height: 270,
        showlegend: false,
      },
    };
  });

  wellSeries = computed(() => {
    const r = this.result();
    if (!r) return null;
    return {
      data: [
        {
          type: 'scatter',
          mode: 'lines',
          x: r.times,
          y: r.well_drawdown.modflow,
          name: 'MODFLOW',
          line: { color: '#16324f', width: 2.5 },
        },
        {
          type: 'scatter',
          mode: 'lines',
          x: r.times,
          y: r.well_drawdown.pinn,
          name: 'PINN',
          line: { color: '#0f766e', width: 2.5, dash: 'dash' },
        },
      ],
      layout: {
        title: { text: 'Drawdown at the well — engine comparison', font: { size: 13 } },
        xaxis: { title: { text: 'time (days)' } },
        yaxis: { title: { text: 'drawdown (m)' } },
        height: 270,
        legend: { orientation: 'h', y: -0.25 },
      },
    };
  });

  latencyBar = computed(() => {
    const r = this.result();
    if (!r || r.latency.modflow_s == null) return null;
    return {
      data: [
        {
          type: 'bar',
          x: ['MODFLOW-2005', 'PINN surrogate'],
          y: [r.latency.modflow_s * 1000, r.latency.pinn_ms],
          marker: { color: ['#16324f', '#0f766e'] },
          text: [
            `${r.latency.modflow_s.toFixed(1)} s`,
            `${r.latency.pinn_ms.toFixed(0)} ms`,
          ],
          textposition: 'outside',
          hoverinfo: 'skip',
        },
      ],
      layout: {
        title: { text: 'Wall-clock latency (log scale)', font: { size: 13 } },
        yaxis: { title: { text: 'milliseconds' }, type: 'log' },
        height: 270,
        showlegend: false,
        margin: { l: 60, r: 20, t: 40, b: 40 },
      },
    };
  });
}
