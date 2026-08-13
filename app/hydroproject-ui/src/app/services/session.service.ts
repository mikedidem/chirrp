import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { environment } from '../../environments/environment';
import { ChatLatency, ParseResult } from './pinn.service';

/* ---------- Models ---------- */

export interface SessionSummary {
  id: string;
  title: string;
  participant_id: string | null;
  shared: boolean;
  has_summary: boolean;
  summary_text: string | null;
  created_at: string | null;
  updated_at: string | null;
}

export interface SessionMessage {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  valid: boolean | null;
  scenario_name: string | null;
  parse_json: ParseResult | null;
  latency_json: ChatLatency | null;
  sim_meta: { percent_change: number; q_rate: number } | null;
  created_at: string | null;
}

export interface SessionDetail extends SessionSummary {
  messages: SessionMessage[];
}

/* ---------- Service ---------- */

@Injectable({ providedIn: 'root' })
export class SessionService {
  private api = `${environment.apiUrl}/sessions`;

  constructor(private http: HttpClient) {}

  list(participantId?: string | null, scope: 'mine' | 'all' = 'all') {
    let url = this.api;
    if (participantId) {
      const p = new URLSearchParams({ participant_id: participantId, scope });
      url = `${this.api}?${p.toString()}`;
    }
    return this.http.get<SessionSummary[]>(url);
  }

  create(title?: string, participantId?: string | null, shared = true) {
    return this.http.post<SessionSummary>(this.api, {
      title: title ?? null,
      participant_id: participantId ?? null,
      shared,
    });
  }

  get(id: string) {
    return this.http.get<SessionDetail>(`${this.api}/${encodeURIComponent(id)}`);
  }

  rename(id: string, title: string) {
    return this.http.patch<SessionSummary>(
      `${this.api}/${encodeURIComponent(id)}`, { title });
  }

  remove(id: string) {
    return this.http.delete<{ deleted: string }>(
      `${this.api}/${encodeURIComponent(id)}`);
  }

  summarize(id: string) {
    return this.http.post<SessionSummary>(
      `${this.api}/${encodeURIComponent(id)}/summarize`, {});
  }
}
