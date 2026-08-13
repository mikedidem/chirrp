import { Component, OnInit, signal } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { firstValueFrom } from 'rxjs';
import { RagService, Document, Source } from '../../services/rag.service';
import { MicButton } from '../../components/mic-button/mic-button';

@Component({
  selector: 'app-rag-assistant',
  standalone: true,
  imports: [CommonModule, FormsModule, MicButton],
  templateUrl: './rag-assistant.html',
  styleUrl: './rag-assistant.css',
})
export class RagAssistant implements OnInit {
  constructor(private ragService: RagService) {}

  currentMode = signal<'ask' | 'compare'>('ask');
  questionInput = signal('');

  // Status
  statusConnected = signal(false);
  statusText = signal('Checking...');
  docsIndexed = signal(0);
  chunksIndexed = signal(0);
  sourcesList = signal<Document[]>([]);

  // Chat messages
  messages = signal<any[]>([]);
  showLocalSources = signal(true);
  isLoading = signal(false);
  reindexing = signal(false);
  reindexError = signal<string | null>(null);
  
  // Suggestions
  suggestions = [
    'What permits are needed to drill a well?',
    'What is the Nebraska Groundwater Management Act?',
    'What are CPNRD rules for groundwater use?',
    'How are recharge zones defined?',
  ];

  ngOnInit(): void {
    this.boot();
  }

  async boot(): Promise<void> {
    try {
      const response = await firstValueFrom(this.ragService.getHealth());
      if (response?.status === 'ok') {
        this.statusConnected.set(true);
        this.statusText.set('Connected');
        this.docsIndexed.set(response.docs_indexed || 0);
        this.chunksIndexed.set(response.chunks_indexed || 0);
      } else {
        this.setErr();
      }
    } catch {
      this.setErr();
    }

    try {
      const docs = await firstValueFrom(this.ragService.getSources());
      this.sourcesList.set(docs || []);
    } catch {
      this.sourcesList.set([]);
    }
  }

  setErr(): void {
    this.statusConnected.set(false);
    this.statusText.set('API offline');
  }

  setMode(mode: 'ask' | 'compare'): void {
    this.currentMode.set(mode);
  }

  clearChat(): void {
    this.messages.set([]);
    this.questionInput.set('');
  }

  async rebuildRagIndex(): Promise<void> {
    if (this.reindexing()) return;

    this.reindexing.set(true);
    this.reindexError.set(null);
    this.statusText.set('Reindexing...');

    try {
      const result = await firstValueFrom(this.ragService.reindex());
      this.statusConnected.set(true);
      this.statusText.set('Index rebuilt');
      this.docsIndexed.set(result.docs_indexed || 0);
      this.chunksIndexed.set(result.chunks_indexed || 0);

      const docs = await firstValueFrom(this.ragService.getSources());
      this.sourcesList.set(docs || []);
    } catch (error: any) {
      this.reindexError.set(error?.error?.detail || error?.message || 'Failed to rebuild the RAG index');
      this.statusText.set('Reindex failed');
    } finally {
      this.reindexing.set(false);
    }
  }

  autoResize(event: any): void {
    const el = event.target;
    el.style.height = 'auto';
    el.style.height = Math.min(el.scrollHeight, 120) + 'px';
  }

  handleKey(event: KeyboardEvent): void {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault();
      this.sendQuestion();
    }
  }

  suggest(suggestion: string): void {
    this.questionInput.set(suggestion);
    setTimeout(() => this.sendQuestion(), 0);
  }

  esc(s: string): string {
    return String(s)
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/\n/g, '<br>');
  }

  riskClass(r: string): string {
    const map: { [key: string]: string } = {
      low: 'risk-low',
      medium: 'risk-medium',
      high: 'risk-high',
    };
    return map[r] || 'risk-high';
  }

  async sendQuestion(): Promise<void> {
    const question = this.questionInput().trim();
    if (!question) return;

    this.isLoading.set(true);
    this.questionInput.set('');

    // Add user message
    this.messages.update((messages) => [...messages, {
      type: 'user',
      content: question,
    }]);

    // Add thinking indicator
    const thinkingIndex = this.messages().length;
    this.messages.update((messages) => [...messages, {
      type: 'thinking',
      label: this.currentMode() === 'compare'
        ? 'Running RAG and General LLM in parallel...'
        : 'Searching policy documents...',
    }]);

    try {
      if (this.currentMode() === 'compare') {
        const data = await firstValueFrom(this.ragService.compare(question));

        this.messages.update((messages) => messages.filter((_, index) => index !== thinkingIndex));

        if (data) {
          this.messages.update((messages) => [...messages, {
            type: 'compare',
            data,
          }]);
        } else {
          this.messages.update((messages) => [...messages, {
            type: 'error',
            content: 'Server error: No response data',
          }]);
        }
      } else {
        const data = await firstValueFrom(this.ragService.ask(
          question,
          8,
          0.45,
          this.showLocalSources()
        ));

        this.messages.update((messages) => messages.filter((_, index) => index !== thinkingIndex));

        if (data) {
          this.messages.update((messages) => [...messages, {
            type: 'ask',
            data,
          }]);
        } else {
          this.messages.update((messages) => [...messages, {
            type: 'error',
            content: 'Server error: No response data',
          }]);
        }
      }
    } catch (e: any) {
      this.messages.update((messages) => messages.filter((_, index) => index !== thinkingIndex));
      this.messages.update((messages) => [...messages, {
        type: 'error',
        content: `Could not reach API. Is the server running? (${e.message})`,
      }]);
    }

    this.isLoading.set(false);
  }

  buildSourcesPanel(sources: Source[]): any[] {
    return sources || [];
  }

  getRelevanceScore(source: Source): number {
    return source.relevance_score ?? source.score ?? 0;
  }
}
