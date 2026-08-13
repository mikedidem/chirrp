import {
  Component,
  ElementRef,
  OnDestroy,
  PLATFORM_ID,
  effect,
  inject,
  input,
  output,
  viewChild,
} from '@angular/core';
import { isPlatformBrowser } from '@angular/common';

/**
 * Thin Plotly wrapper: lazy-loads plotly.js in the browser only (SSR-safe),
 * re-renders reactively from signal inputs, applies the CHIRRP theme, and
 * surfaces click events for map-probe interactions.
 */

let plotlyPromise: Promise<any> | null = null;

function loadPlotly(): Promise<any> {
  if (!plotlyPromise) {
    plotlyPromise = import('plotly.js-dist-min').then(
      (m: any) => m.default ?? m,
    );
  }
  return plotlyPromise;
}

// Deep-water dark theme for charts. Axis styling goes in `template` so that
// per-page layouts (which set xaxis/yaxis titles) merge with — rather than
// overwrite — these defaults.
const AXIS = {
  gridcolor: 'rgba(140, 190, 210, 0.12)',
  zerolinecolor: 'rgba(140, 190, 210, 0.22)',
  linecolor: 'rgba(140, 190, 210, 0.25)',
  tickcolor: 'rgba(140, 190, 210, 0.25)',
  tickfont: { color: '#a6c0cc' },
  title: { font: { color: '#c7dde6' } },
};

const BASE_LAYOUT = {
  font: {
    family: "'Inter', system-ui, sans-serif",
    color: '#cfe6ee',
    size: 12,
  },
  paper_bgcolor: 'rgba(0,0,0,0)',
  plot_bgcolor: 'rgba(0,0,0,0)',
  margin: { l: 55, r: 20, t: 36, b: 45 },
  colorway: ['#22d3ee', '#f59e0b', '#7cc4ff', '#a78bfa', '#34d399'],
  template: {
    layout: {
      font: { color: '#cfe6ee' },
      xaxis: AXIS,
      yaxis: AXIS,
      colorway: ['#22d3ee', '#f59e0b', '#7cc4ff', '#a78bfa', '#34d399'],
    },
  },
};

@Component({
  selector: 'app-plotly-chart',
  template: `<div #host class="plot-host"></div>`,
  styles: [
    ':host { display: block; width: 100%; }',
    '.plot-host { width: 100%; }',
  ],
})
export class PlotlyChart implements OnDestroy {
  data = input<any[]>([]);
  layout = input<any>({});
  plotConfig = input<any>({ displayModeBar: false, responsive: true });

  plotClick = output<{ x: number; y: number }>();

  private host = viewChild.required<ElementRef<HTMLDivElement>>('host');
  private platformId = inject(PLATFORM_ID);
  private el: HTMLDivElement | null = null;
  private clickBound = false;

  constructor() {
    effect(() => {
      const data = this.data();
      const layout = this.layout();
      const config = this.plotConfig();
      if (!isPlatformBrowser(this.platformId)) return;
      const el = this.host().nativeElement;
      this.el = el;
      loadPlotly().then((Plotly) => {
        Plotly.react(el, data, { ...BASE_LAYOUT, ...layout }, config);
        if (!this.clickBound && (el as any).on) {
          (el as any).on('plotly_click', (ev: any) => {
            const p = ev?.points?.[0];
            if (p && p.x !== undefined && p.y !== undefined) {
              this.plotClick.emit({ x: Number(p.x), y: Number(p.y) });
            }
          });
          this.clickBound = true;
        }
      });
    });
  }

  ngOnDestroy(): void {
    if (this.el) {
      const el = this.el;
      loadPlotly().then((Plotly) => Plotly.purge(el));
    }
  }
}
