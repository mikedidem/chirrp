import {
  Component,
  EventEmitter,
  Input,
  OnDestroy,
  Output,
  PLATFORM_ID,
  inject,
  signal,
} from '@angular/core';
import { isPlatformBrowser } from '@angular/common';

/**
 * Stakeholder-friendly voice input using the browser-native Web Speech API.
 * Transcribes speech into a bound text field (append, never replace) and never
 * touches the backend, sends, or stores audio. Falls back to a disabled button
 * with a tooltip when the browser has no SpeechRecognition support.
 *
 * Usage: <app-mic-button [text]="value" (textChange)="value = $event" [disabled]="busy" />
 */
@Component({
  selector: 'app-mic-button',
  standalone: true,
  template: `
    <button
      type="button"
      class="mic-btn"
      [class.listening]="listening()"
      [class.has-error]="!!error()"
      [disabled]="disabled || !supported()"
      [attr.aria-label]="listening() ? 'Stop voice input' : 'Start voice input'"
      [title]="tooltip()"
      (click)="toggle()"
    >
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" aria-hidden="true">
        <rect x="9" y="3" width="6" height="11" rx="3" />
        <path d="M5 11a7 7 0 0 0 14 0M12 18v3" />
      </svg>
      @if (listening()) {
        <span class="mic-ping" aria-hidden="true"></span>
      }
    </button>
  `,
  styles: [`
    :host { display: inline-flex; flex-shrink: 0; }
    .mic-btn {
      position: relative;
      width: 36px; height: 36px;
      display: inline-flex; align-items: center; justify-content: center;
      border-radius: 50%;
      border: 1px solid var(--border-strong);
      background: rgba(255, 255, 255, 0.05);
      color: var(--ink-soft);
      cursor: pointer;
      transition: color .15s, border-color .15s, background .15s, box-shadow .2s;
    }
    .mic-btn svg { width: 18px; height: 18px; }
    .mic-btn:hover:not(:disabled) {
      color: var(--accent-strong);
      border-color: var(--accent-border);
    }
    .mic-btn:disabled { opacity: .45; cursor: not-allowed; }
    .mic-btn.has-error { color: var(--danger); border-color: rgba(251, 113, 133, .45); }
    .mic-btn.listening {
      color: var(--accent-deep);
      background: linear-gradient(180deg, #3ee9ff, #14bcdd);
      border-color: transparent;
      box-shadow: 0 0 0 3px rgba(57, 223, 242, .2), 0 0 18px rgba(57, 223, 242, .5);
    }
    .mic-ping {
      position: absolute; inset: -1px;
      border-radius: 50%;
      border: 1.5px solid var(--accent);
      animation: mic-ping 1.4s ease-out infinite;
    }
    @keyframes mic-ping {
      0% { transform: scale(1); opacity: .7; }
      100% { transform: scale(1.8); opacity: 0; }
    }
    @media (prefers-reduced-motion: reduce) { .mic-ping { animation: none; } }
  `],
})
export class MicButton implements OnDestroy {
  @Input() text = '';
  @Input() disabled = false;
  @Output() textChange = new EventEmitter<string>();

  private platformId = inject(PLATFORM_ID);
  supported = signal(false);
  listening = signal(false);
  error = signal<string | null>(null);

  private recog: any = null;
  private baseText = '';

  constructor() {
    if (isPlatformBrowser(this.platformId)) {
      const SR = (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition;
      this.supported.set(!!SR);
    }
  }

  tooltip(): string {
    if (!this.supported()) return 'Speech input is not supported in this browser';
    if (this.error()) return this.error()!;
    return this.listening() ? 'Listening… — click to stop' : 'Voice input — speak your scenario';
  }

  toggle(): void {
    if (this.disabled || !this.supported()) return;
    if (this.listening()) { this.stop(); return; }
    this.start();
  }

  private start(): void {
    const SR = (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition;
    if (!SR) return;
    this.error.set(null);
    const r = new SR();
    r.lang = 'en-US';
    r.interimResults = true;
    r.continuous = false;
    // Capture whatever is already typed so speech is appended, not replaced.
    this.baseText = (this.text || '').trim();

    r.onresult = (e: any) => {
      let finalT = '';
      let interimT = '';
      for (let i = 0; i < e.results.length; i++) {
        const res = e.results[i];
        if (res.isFinal) finalT += res[0].transcript;
        else interimT += res[0].transcript;
      }
      const spoken = (finalT + interimT).trim();
      const sep = this.baseText ? ' ' : '';
      this.textChange.emit(this.baseText + sep + spoken);
      if (finalT) this.baseText = (this.baseText + sep + finalT).trim();
    };
    r.onerror = (e: any) => {
      this.listening.set(false);
      const code = e?.error;
      this.error.set(
        code === 'not-allowed' || code === 'service-not-allowed'
          ? 'Microphone permission denied'
          : 'Speech input unavailable',
      );
    };
    r.onend = () => this.listening.set(false);

    this.recog = r;
    try {
      r.start();
      this.listening.set(true);
    } catch {
      this.listening.set(false);
    }
  }

  private stop(): void {
    try { this.recog?.stop(); } catch { /* ignore */ }
    this.listening.set(false);
  }

  ngOnDestroy(): void {
    this.stop();
  }
}
