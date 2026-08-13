import { Injectable } from '@angular/core';
import type jsPDF from 'jspdf';

export interface PdfMetric {
  label: string;
  value: string;
}

export interface ScenarioPdfOptions {
  title: string;
  subtitle?: string;
  summary?: string | null;
  metrics?: PdfMetric[];
  /** DOM node (results panel) snapshotted into the PDF, e.g. charts. */
  captureEl?: HTMLElement | null;
  /** Extra paragraphs, e.g. a session "what was achieved" recap. */
  sections?: { heading: string; body: string }[];
}

/**
 * Client-side PDF export so stakeholders can save and share planning results.
 * Renders text with jsPDF and snapshots rendered charts with html2canvas, so
 * no server-side rendering is required.
 */
@Injectable({ providedIn: 'root' })
export class ExportService {
  async exportScenarioPdf(opts: ScenarioPdfOptions): Promise<void> {
    // Lazy-load the heavy PDF/canvas libs so they stay out of the initial bundle.
    const { default: JsPDF } = await import('jspdf');
    const doc = new JsPDF({ unit: 'pt', format: 'a4' });
    const pageW = doc.internal.pageSize.getWidth();
    const pageH = doc.internal.pageSize.getHeight();
    const margin = 40;
    const contentW = pageW - margin * 2;
    let y = margin;

    const ensureSpace = (needed: number) => {
      if (y + needed > pageH - margin) {
        doc.addPage();
        y = margin;
      }
    };

    // Header
    doc.setFont('helvetica', 'bold');
    doc.setFontSize(16);
    doc.text(opts.title, margin, y);
    y += 20;

    doc.setFont('helvetica', 'normal');
    doc.setFontSize(10);
    doc.setTextColor(110);
    const stamp = opts.subtitle
      ? `${opts.subtitle}  ·  ${new Date().toLocaleString()}`
      : new Date().toLocaleString();
    doc.text(stamp, margin, y);
    doc.setTextColor(20);
    y += 22;

    // Metrics
    if (opts.metrics?.length) {
      doc.setFont('helvetica', 'bold');
      doc.setFontSize(11);
      for (const m of opts.metrics) {
        ensureSpace(16);
        doc.text(`${m.label}: `, margin, y);
        const w = doc.getTextWidth(`${m.label}: `);
        doc.setFont('helvetica', 'normal');
        doc.text(m.value, margin + w, y);
        doc.setFont('helvetica', 'bold');
        y += 16;
      }
      y += 8;
    }

    // Summary
    if (opts.summary) {
      y = this.writeParagraph(doc, 'Summary', opts.summary, margin, y,
        contentW, pageH, ensureSpace);
    }

    // Extra sections (e.g. session recap)
    for (const s of opts.sections ?? []) {
      y = this.writeParagraph(doc, s.heading, s.body, margin, y,
        contentW, pageH, ensureSpace);
    }

    // Chart snapshot
    if (opts.captureEl) {
      try {
        const { default: html2canvas } = await import('html2canvas');
        const canvas = await html2canvas(opts.captureEl, {
          backgroundColor: '#ffffff',
          scale: 2,
        });
        const imgData = canvas.toDataURL('image/png');
        const imgW = contentW;
        const imgH = (canvas.height / canvas.width) * imgW;
        ensureSpace(imgH + 10);
        doc.addImage(imgData, 'PNG', margin, y, imgW, imgH);
        y += imgH + 10;
      } catch {
        // Chart capture is best-effort; the text PDF still exports.
      }
    }

    const safe = opts.title.replace(/[^a-z0-9-_]+/gi, '_').slice(0, 60);
    doc.save(`${safe || 'chirrp-export'}.pdf`);
  }

  private writeParagraph(
    doc: jsPDF,
    heading: string,
    body: string,
    margin: number,
    y: number,
    contentW: number,
    pageH: number,
    ensureSpace: (n: number) => void,
  ): number {
    ensureSpace(28);
    doc.setFont('helvetica', 'bold');
    doc.setFontSize(12);
    doc.text(heading, margin, y);
    y += 16;
    doc.setFont('helvetica', 'normal');
    doc.setFontSize(10.5);
    const lines = doc.splitTextToSize(body, contentW) as string[];
    for (const line of lines) {
      ensureSpace(14);
      doc.text(line, margin, y);
      y += 14;
    }
    return y + 10;
  }
}
