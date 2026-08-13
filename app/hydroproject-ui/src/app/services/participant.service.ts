import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { environment } from '../../environments/environment';

export interface Participant {
  id: string;
  name: string;
  role: string | null;
  created_at: string | null;
}

export interface SessionNote {
  id: string;
  session_id: string;
  participant_id: string | null;
  kind: 'note' | 'decision';
  content: string;
  created_at: string | null;
}

@Injectable({ providedIn: 'root' })
export class ParticipantService {
  private api = environment.apiUrl;

  constructor(private http: HttpClient) {}

  list() {
    return this.http.get<Participant[]>(`${this.api}/participants`);
  }

  create(name: string, role?: string) {
    return this.http.post<Participant>(`${this.api}/participants`, {
      name,
      role: role ?? null,
    });
  }

  listNotes(sessionId: string) {
    return this.http.get<SessionNote[]>(
      `${this.api}/sessions/${encodeURIComponent(sessionId)}/notes`);
  }

  addNote(sessionId: string, content: string, kind: 'note' | 'decision',
          participantId?: string | null) {
    return this.http.post<SessionNote>(
      `${this.api}/sessions/${encodeURIComponent(sessionId)}/notes`, {
        content,
        kind,
        participant_id: participantId ?? null,
      });
  }
}
