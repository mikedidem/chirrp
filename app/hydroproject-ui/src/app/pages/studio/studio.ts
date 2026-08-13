import { Component, ElementRef, OnInit, ViewChild, computed, effect, signal } from '@angular/core';
import { CommonModule, DecimalPipe } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { ActivatedRoute, RouterLink } from '@angular/router';
import { PlotlyChart } from '../../components/plotly-chart/plotly-chart';
import {
  ChatResponse,
  PinnMeta,
  PinnService,
  ScenarioSummary,
  SimulationResult,
} from '../../services/pinn.service';
import {
  SessionDetail,
  SessionMessage,
  SessionService,
  SessionSummary,
} from '../../services/session.service';
import {
  Participant,
  ParticipantService,
  SessionNote,
} from '../../services/participant.service';
import { ExportService } from '../../services/export.service';
import { RagService, ScenarioContext } from '../../services/rag.service';
import { MicButton } from '../../components/mic-button/mic-button';

const ACTIVE_PARTICIPANT_KEY = 'chirrp.activeParticipantId';

interface ChatMessage {
  role: 'user' | 'assistant';
  text: string;
  valid?: boolean;
  parse?: ChatResponse['parse'];
  latency?: ChatResponse['latency'];
  suggestion?: string;
  savedAs?: string | null;
  percentChange?: number | null;
}

@Component({
  selector: 'app-studio',
  imports: [CommonModule, FormsModule, DecimalPipe, PlotlyChart, MicButton, RouterLink],
  templateUrl: './studio.html',
  styleUrl: './studio.css',
})
export class Studio implements OnInit {
  meta = signal<PinnMeta | null>(null);
  messages = signal<ChatMessage[]>([]);
  busy = signal(false);
  sim = signal<SimulationResult | null>(null);
  timeIdx = signal(0);
  viewMode = signal<'head' | 'drawdown'>('head');
  probeReadout = signal<{ x: number; y: number; t: number; head: number; drawdown: number } | null>(null);
  scenarios = signal<ScenarioSummary[]>([]);

  // Sessions
  sessions = signal<SessionSummary[]>([]);
  activeSessionId = signal<string | null>(null);
  summary = signal<string | null>(null);
  summarizing = signal(false);
  exporting = signal(false);

  // Participants (local profiles) + decision log
  participants = signal<Participant[]>([]);
  activeParticipantId = signal<string | null>(null);
  scope = signal<'mine' | 'all'>('all');
  notes = signal<SessionNote[]>([]);
  noteText = '';

  // RAG↔PINN: regulatory context for the current scenario
  policyContext = signal<ScenarioContext | null>(null);
  policyLoading = signal(false);

  instruction = '';
  manualPercent: number | null = null;
  saveName = '';

  @ViewChild('resultsPanel') resultsPanel?: ElementRef<HTMLElement>;

  constructor(
    private pinn: PinnService,
    private sessionSvc: SessionService,
    private participantSvc: ParticipantService,
    private exportSvc: ExportService,
    private ragSvc: RagService,
    private route: ActivatedRoute,
  ) {
    // Whenever a scenario is (re)computed, fetch its regulatory context.
    effect(() => {
      const s = this.sim();
      if (s) this.loadPolicyContext(s);
      else this.policyContext.set(null);
    });
  }

  /** Best-effort RAG lookup for the current scenario; never blocks the result. */
  private loadPolicyContext(sim: SimulationResult): void {
    this.policyLoading.set(true);
    this.policyContext.set(null);
    this.ragSvc
      .scenarioContext({
        percent_change: sim.percent_change,
        q_rate: sim.q_rate,
        max_drawdown: sim.max_drawdown,
      })
      .subscribe({
        next: (ctx) => {
          this.policyContext.set(ctx);
          this.policyLoading.set(false);
        },
        error: () => {
          this.policyContext.set({
            available: false,
            query: '',
            note: 'Regulatory context is unavailable right now.',
            sources: [],
          });
          this.policyLoading.set(false);
        },
      });
  }

  ngOnInit(): void {
    this.pinn.getMeta().subscribe({
      next: (m) => this.meta.set(m),
      error: () => {},
    });
    const stored = typeof localStorage !== 'undefined'
      ? localStorage.getItem(ACTIVE_PARTICIPANT_KEY) : null;
    if (stored) this.activeParticipantId.set(stored);
    this.refreshParticipants();
    this.refreshScenarios();
    this.refreshSessions();

    // A scenario typed in the Overview hero arrives as ?q=… — run it.
    const q = this.route.snapshot.queryParamMap.get('q');
    if (q && q.trim()) {
      this.instruction = q.trim();
      setTimeout(() => this.sendInstruction());
    }
  }

  /* ----------------------- Participants ----------------------- */

  refreshParticipants(): void {
    this.participantSvc.list().subscribe({
      next: (p) => this.participants.set(p),
      error: () => this.participants.set([]),
    });
  }

  activeParticipant = computed(() =>
    this.participants().find((p) => p.id === this.activeParticipantId()) ?? null);

  participantName(id: string | null): string {
    if (!id) return 'shared';
    return this.participants().find((p) => p.id === id)?.name ?? 'unknown';
  }

  setActiveParticipant(id: string | null): void {
    this.activeParticipantId.set(id);
    if (typeof localStorage !== 'undefined') {
      if (id) localStorage.setItem(ACTIVE_PARTICIPANT_KEY, id);
      else localStorage.removeItem(ACTIVE_PARTICIPANT_KEY);
    }
    this.refreshSessions();
  }

  newParticipant(): void {
    const name = window.prompt('Your name (participant profile)');
    if (!name || !name.trim()) return;
    this.participantSvc.create(name.trim()).subscribe({
      next: (p) => {
        this.refreshParticipants();
        this.setActiveParticipant(p.id);
      },
      error: () => {},
    });
  }

  setScope(s: 'mine' | 'all'): void {
    this.scope.set(s);
    this.refreshSessions();
  }

  refreshScenarios(): void {
    this.pinn.listScenarios().subscribe({
      next: (s) => this.scenarios.set(s),
      error: () => this.scenarios.set([]),
    });
  }

  refreshSessions(): void {
    this.sessionSvc.list(this.activeParticipantId(), this.scope()).subscribe({
      next: (s) => this.sessions.set(s),
      error: () => this.sessions.set([]),
    });
  }

  activeSession = computed(() =>
    this.sessions().find((s) => s.id === this.activeSessionId()) ?? null,
  );

  /* ----------------------- Session management ----------------------- */

  newSession(): void {
    this.activeSessionId.set(null);
    this.messages.set([]);
    this.sim.set(null);
    this.summary.set(null);
    this.probeReadout.set(null);
    this.notes.set([]);
    this.policyContext.set(null);
    this.policyLoading.set(false);
  }

  loadSession(s: SessionSummary): void {
    if (this.busy()) return;
    this.busy.set(true);
    this.sessionSvc.get(s.id).subscribe({
      next: (detail: SessionDetail) => {
        this.activeSessionId.set(detail.id);
        this.summary.set(detail.summary_text);
        this.messages.set(detail.messages.map(this.toChatMessage));
        this.restoreLastScenario(detail.messages);
        this.loadNotes(detail.id);
        this.busy.set(false);
      },
      error: () => this.busy.set(false),
    });
  }

  /* ----------------------- Decision log ----------------------- */

  loadNotes(sessionId: string): void {
    this.participantSvc.listNotes(sessionId).subscribe({
      next: (n) => this.notes.set(n),
      error: () => this.notes.set([]),
    });
  }

  addNote(kind: 'note' | 'decision'): void {
    const text = this.noteText.trim();
    const sid = this.activeSessionId();
    if (!text || !sid) return;
    this.participantSvc.addNote(sid, text, kind, this.activeParticipantId()).subscribe({
      next: (n) => {
        this.notes.update((arr) => [...arr, n]);
        this.noteText = '';
      },
      error: () => {},
    });
  }

  renameSession(s: SessionSummary): void {
    const title = window.prompt('Rename session', s.title);
    if (!title || !title.trim()) return;
    this.sessionSvc.rename(s.id, title.trim()).subscribe({
      next: () => this.refreshSessions(),
      error: () => {},
    });
  }

  deleteSession(s: SessionSummary, ev: Event): void {
    ev.stopPropagation();
    this.sessionSvc.remove(s.id).subscribe({
      next: () => {
        if (this.activeSessionId() === s.id) this.newSession();
        this.refreshSessions();
      },
      error: () => {},
    });
  }

  generateSummary(): void {
    const id = this.activeSessionId();
    if (!id || this.summarizing()) return;
    this.summarizing.set(true);
    this.sessionSvc.summarize(id).subscribe({
      next: (s) => {
        this.summary.set(s.summary_text);
        this.summarizing.set(false);
        this.refreshSessions();
      },
      error: () => this.summarizing.set(false),
    });
  }

  private toChatMessage = (m: SessionMessage): ChatMessage => ({
    role: m.role,
    text: m.content,
    valid: m.valid ?? undefined,
    parse: m.parse_json ?? undefined,
    latency: m.latency_json ?? undefined,
    savedAs: m.scenario_name,
    percentChange: m.sim_meta?.percent_change ?? null,
  });

  /** Re-run the surrogate for the most recent valid scenario to restore charts. */
  private restoreLastScenario(messages: SessionMessage[]): void {
    const last = [...messages].reverse().find(
      (m) => m.role === 'assistant' && m.sim_meta?.percent_change != null,
    );
    if (!last || last.sim_meta == null) {
      this.sim.set(null);
      return;
    }
    this.pinn.simulate({ percent_change: last.sim_meta.percent_change }).subscribe({
      next: (sim) => {
        this.sim.set(sim);
        this.timeIdx.set(sim.times.length - 1);
        this.probeReadout.set(null);
      },
      error: () => {},
    });
  }

  /* ----------------------- Chart computeds ----------------------- */

  currentTime = computed(() => {
    const s = this.sim();
    return s ? s.times[this.timeIdx()] : 0;
  });

  /** Where the current scenario sits relative to the trained planning envelope. */
  envelopeStatus = computed<{ label: string; tone: 'ok' | 'warn' | 'danger' } | null>(() => {
    const s = this.sim();
    const m = this.meta();
    if (!s || !m) return null;
    const [lo, hi] = m.percent_bounds;
    const pc = s.percent_change;
    if (pc < lo - 1e-6 || pc > hi + 1e-6) {
      return { label: 'Outside trained range — not reliable', tone: 'danger' };
    }
    const span = hi - lo;
    const margin = Math.min(pc - lo, hi - pc);
    if (span > 0 && margin < span * 0.12) {
      return { label: 'Near envelope edge — interpret with caution', tone: 'warn' };
    }
    return { label: 'Within trained envelope', tone: 'ok' };
  });

  /** Short plain-language decision-support summary shown above the charts. */
  decisionReadout = computed<
    { scenario: string; response: string; status: string;
      tone: 'ok' | 'warn' | 'danger'; meaning: string } | null
  >(() => {
    const s = this.sim();
    const es = this.envelopeStatus();
    if (!s || !es) return null;
    const pc = s.percent_change;
    const verb = pc > 0 ? 'increased' : pc < 0 ? 'reduced' : 'unchanged';
    const lastUser = [...this.messages()].reverse()
      .find((m) => m.role === 'user')?.text;
    const scenario = lastUser && lastUser.trim()
      ? lastUser.trim()
      : `Pumping ${verb} by ${Math.abs(pc).toFixed(1)}% (Q = ${Math.round(s.q_rate).toLocaleString()} m³/day)`;
    const wellFinal = s.well_drawdown_series[s.well_drawdown_series.length - 1];
    const response =
      `${s.max_drawdown.toFixed(2)} m maximum drawdown · ${wellFinal.toFixed(2)} m at the well after 30 days`;
    const meaning = es.tone === 'ok'
      ? 'Within the model’s validated range — the predicted drawdown is suitable for planning use.'
      : es.tone === 'warn'
        ? 'Near the edge of the validated range — treat the result as indicative and confirm before relying on it.'
        : 'Outside the model’s validated range — this prediction is not reliable and should not guide planning.';
    return { scenario, response, status: es.label, tone: es.tone, meaning };
  });

  mapData = computed(() => {
    const s = this.sim();
    if (!s) return [];
    const idx = Math.min(this.timeIdx(), s.times.length - 1);
    const mode = this.viewMode();
    const z =
      mode === 'head'
        ? s.heads[idx]
        : s.heads[idx].map((row) => row.map((h) => s.initial_head - h));
    return [
      {
        type: 'heatmap',
        x: s.x,
        y: s.y,
        z,
        colorscale: mode === 'head' ? 'Viridis' : 'Reds',
        reversescale: mode === 'drawdown' ? false : false,
        colorbar: {
          title: { text: mode === 'head' ? 'Head (m)' : 'Drawdown (m)', side: 'right' },
          thickness: 14,
          len: 0.9,
        },
        hovertemplate:
          'x %{x:.0f} m · y %{y:.0f} m<br>' +
          (mode === 'head' ? 'head' : 'drawdown') +
          ' %{z:.3f} m<extra></extra>',
      },
      {
        type: 'scatter',
        mode: 'markers',
        x: [s.well_xy[0]],
        y: [s.well_xy[1]],
        marker: { symbol: 'x', size: 11, color: mode === 'head' ? '#ffffff' : '#16324f', line: { width: 2 } },
        name: 'well',
        hovertemplate: 'pumping well<extra></extra>',
        showlegend: false,
      },
    ];
  });

  mapLayout = computed(() => ({
    title: {
      text:
        (this.viewMode() === 'head' ? 'Hydraulic head' : 'Drawdown') +
        ` — t = ${this.currentTime()} d`,
      font: { size: 14 },
    },
    xaxis: { title: { text: 'x (m)' }, constrain: 'domain' },
    yaxis: { title: { text: 'y (m)' }, scaleanchor: 'x', scaleratio: 1 },
    height: 480,
  }));

  seriesData = computed(() => {
    const s = this.sim();
    if (!s) return [];
    const idx = Math.min(this.timeIdx(), s.times.length - 1);
    return [
      {
        type: 'scatter',
        mode: 'lines',
        x: s.times,
        y: s.well_drawdown_series,
        line: { color: '#0f766e', width: 2.5 },
        name: 'drawdown at well',
        hovertemplate: 't %{x} d · %{y:.3f} m<extra></extra>',
      },
      {
        type: 'scatter',
        mode: 'markers',
        x: [s.times[idx]],
        y: [s.well_drawdown_series[idx]],
        marker: { size: 9, color: '#b45309' },
        showlegend: false,
        hoverinfo: 'skip',
      },
    ];
  });

  seriesLayout = {
    title: { text: 'Drawdown at the pumping well', font: { size: 14 } },
    xaxis: { title: { text: 'time (days)' } },
    yaxis: { title: { text: 'drawdown (m)' } },
    height: 240,
    margin: { l: 55, r: 20, t: 36, b: 42 },
    showlegend: false,
  };

  /* ----------------------- Chat ----------------------- */

  sendInstruction(): void {
    const text = this.instruction.trim();
    if (!text || this.busy()) return;
    this.busy.set(true);
    this.messages.update((m) => [...m, { role: 'user', text }]);
    this.instruction = '';

    const send = (sessionId: string) => {
      this.pinn.chat(text, this.saveName.trim() || undefined, sessionId).subscribe({
        next: (resp) => this.handleChatResponse(resp),
        error: (err) => this.handleChatError(err),
      });
    };

    // Auto-create a session on first message so work is always persisted.
    if (this.activeSessionId()) {
      send(this.activeSessionId()!);
    } else {
      const title = text.length > 60 ? text.slice(0, 57) + '…' : text;
      this.sessionSvc.create(title, this.activeParticipantId()).subscribe({
        next: (s) => {
          this.activeSessionId.set(s.id);
          this.refreshSessions();
          send(s.id);
        },
        // If session creation fails (DB down) still run the chat, unpersisted.
        error: () => {
          this.pinn.chat(text, this.saveName.trim() || undefined).subscribe({
            next: (resp) => this.handleChatResponse(resp),
            error: (err) => this.handleChatError(err),
          });
        },
      });
    }
  }

  private handleChatResponse(resp: ChatResponse): void {
    this.busy.set(false);
    if (resp.is_valid && resp.simulation) {
      this.sim.set(resp.simulation);
      this.timeIdx.set(resp.simulation.times.length - 1);
      this.probeReadout.set(null);
      this.messages.update((m) => [
        ...m,
        {
          role: 'assistant',
          text: resp.summary || 'Scenario simulated.',
          valid: true,
          parse: resp.parse,
          latency: resp.latency,
          savedAs: resp.saved_as,
          percentChange: resp.parse?.percent_change ?? null,
        },
      ]);
      this.saveName = '';
      this.refreshScenarios();
      this.refreshSessions();
    } else {
      this.messages.update((m) => [
        ...m,
        {
          role: 'assistant',
          text: resp.error || 'The instruction could not be applied.',
          valid: false,
          parse: resp.parse,
          latency: resp.latency,
          suggestion: resp.suggestion,
        },
      ]);
    }
  }

  private handleChatError(err: any): void {
    this.busy.set(false);
    this.messages.update((m) => [
      ...m,
      { role: 'assistant', text: this.describeError(err), valid: false },
    ]);
  }

  runManual(): void {
    if (this.manualPercent === null || this.busy()) return;
    this.busy.set(true);
    this.pinn
      .simulate({
        percent_change: this.manualPercent,
        scenario_name: this.saveName.trim() || undefined,
      })
      .subscribe({
        next: (sim) => {
          this.busy.set(false);
          this.sim.set(sim);
          this.timeIdx.set(sim.times.length - 1);
          this.probeReadout.set(null);
          this.refreshScenarios();
        },
        error: (err) => {
          this.busy.set(false);
          this.messages.update((m) => [
            ...m,
            { role: 'assistant', text: this.describeError(err), valid: false },
          ]);
        },
      });
  }

  onMapClick(pt: { x: number; y: number }): void {
    const s = this.sim();
    if (!s) return;
    const t = this.currentTime();
    this.pinn.probe(s.q_rate, [{ x: pt.x, y: pt.y, t }]).subscribe({
      next: (r) => this.probeReadout.set(r.points[0]),
      error: () => {},
    });
  }

  loadScenario(sc: ScenarioSummary): void {
    if (this.busy()) return;
    this.busy.set(true);
    this.pinn.simulate({ percent_change: sc.percent_change }).subscribe({
      next: (sim) => {
        this.busy.set(false);
        this.sim.set(sim);
        this.timeIdx.set(sim.times.length - 1);
        this.probeReadout.set(null);
      },
      error: () => this.busy.set(false),
    });
  }

  deleteScenario(sc: ScenarioSummary, ev: Event): void {
    ev.stopPropagation();
    this.pinn.deleteScenario(sc.scenario_name).subscribe({
      next: () => this.refreshScenarios(),
      error: () => {},
    });
  }

  onSliderInput(ev: Event): void {
    this.timeIdx.set(Number((ev.target as HTMLInputElement).value));
  }

  /* ----------------------- Export ----------------------- */

  async exportPdf(): Promise<void> {
    const s = this.sim();
    if (!s || this.exporting()) return;
    this.exporting.set(true);
    const wellFinal = s.well_drawdown_series[s.well_drawdown_series.length - 1];
    const lastSummary = [...this.messages()]
      .reverse()
      .find((m) => m.role === 'assistant' && m.valid)?.text;
    const sections = this.summary()
      ? [{ heading: 'Session recap', body: this.summary()! }]
      : [];
    try {
      await this.exportSvc.exportScenarioPdf({
        title: this.activeSession()?.title || 'CHIRRP scenario',
        subtitle: `${s.percent_change > 0 ? '+' : ''}${s.percent_change.toFixed(1)}% pumping · Q = ${Math.round(s.q_rate)} m³/day`,
        summary: lastSummary ?? null,
        metrics: [
          { label: 'Max drawdown', value: `${s.max_drawdown.toFixed(2)} m` },
          { label: 'Min head', value: `${s.head_min.toFixed(2)} m` },
          { label: 'Drawdown at well (30 d)', value: `${wellFinal.toFixed(2)} m` },
          { label: 'PINN latency', value: `${Math.round(s.latency_ms)} ms` },
        ],
        sections,
        captureEl: this.resultsPanel?.nativeElement ?? null,
      });
    } finally {
      this.exporting.set(false);
    }
  }

  private describeError(err: any): string {
    if (err?.status === 0) {
      return 'Cannot reach the backend at the configured API URL. Is the server running?';
    }
    const detail = err?.error?.detail;
    if (typeof detail === 'string') return detail;
    return `Request failed (HTTP ${err?.status ?? '?'}).`;
  }
}
