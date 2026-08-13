import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { environment } from '../../environments/environment';

export interface Source {
  title: string;
  url: string;
  relevance_score?: number;
  score?: number;
  excerpt?: string;
  chunk_index?: number;
}

export interface AskResponse {
  question: string;
  answer: string;
  hallucination_risk: string;
  confidence_score: number;
  sources_used: Source[];
  show_sources: boolean;
}

export interface CompareResponse {
  question: string;
  rag_answer: string;
  rag_risk: string;
  rag_confidence: number;
  rag_sources: Source[];
  general_answer: string;
}

export interface Document {
  title: string;
  doc_id: string;
  url: string;
  type: string;
}

export interface HealthResponse {
  status: string;
  docs_indexed?: number;
  chunks_indexed?: number;
  detail?: string;
}

export interface ReindexResponse {
  status: string;
  docs_indexed: number;
  chunks_indexed: number;
  detail: string;
}

export interface ScenarioContext {
  available: boolean;
  query: string;
  note: string;
  sources: Source[];
}

@Injectable({
  providedIn: 'root'
})
export class RagService {
  private ragApiUrl = `${environment.apiUrl}/rag`;

  constructor(private http: HttpClient) {}

  getHealth() {
    return this.http.get<HealthResponse>(`${this.ragApiUrl}/health`);
  }

  getSources() {
    return this.http.get<Document[]>(`${this.ragApiUrl}/sources`);
  }

  reindex() {
    return this.http.post<ReindexResponse>(`${this.ragApiUrl}/reindex`, {});
  }

  ask(question: string, topK: number = 8, minScore: number = 0.45, showSources: boolean = true) {
    return this.http.post<AskResponse>(`${this.ragApiUrl}/ask`, {
      question,
      top_k: topK,
      min_score: minScore,
      show_sources: showSources,
    });
  }

  compare(question: string, topK: number = 8) {
    return this.http.post<CompareResponse>(`${this.ragApiUrl}/compare`, {
      question,
      top_k: topK,
    });
  }

  /** Regulations relevant to a PINN scenario (RAG↔PINN decision link). */
  scenarioContext(body: {
    percent_change: number;
    q_rate?: number;
    max_drawdown?: number;
    instruction?: string;
  }) {
    return this.http.post<ScenarioContext>(
      `${this.ragApiUrl}/scenario-context`, body);
  }
}
