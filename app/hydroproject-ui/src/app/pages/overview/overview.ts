import { Component, OnInit, signal } from '@angular/core';
import { CommonModule, DecimalPipe } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { Router, RouterLink } from '@angular/router';
import { PinnMeta, PinnService } from '../../services/pinn.service';
import { MicButton } from '../../components/mic-button/mic-button';

@Component({
  selector: 'app-overview',
  imports: [CommonModule, DecimalPipe, FormsModule, RouterLink, MicButton],
  templateUrl: './overview.html',
  styleUrl: './overview.css',
})
export class Overview implements OnInit {
  meta = signal<PinnMeta | null>(null);
  heroQuery = '';

  constructor(private pinn: PinnService, private router: Router) {}

  ngOnInit(): void {
    this.pinn.getMeta().subscribe({
      next: (m) => this.meta.set(m),
      error: () => {},
    });
  }

  /** Send the hero's natural-language scenario into the Studio and auto-run it. */
  runHeroScenario(): void {
    const q = this.heroQuery.trim();
    this.router.navigate(['/studio'], q ? { queryParams: { q } } : {});
  }
}
