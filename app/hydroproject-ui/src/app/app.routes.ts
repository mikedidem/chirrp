import { Routes } from '@angular/router';
import { Overview } from './pages/overview/overview';
import { Studio } from './pages/studio/studio';
import { Compare } from './pages/compare/compare';
import { Validate } from './pages/validate/validate';
import { GoalSeek } from './pages/goal-seek/goal-seek';
import { RagAssistant } from './pages/rag-assistant/rag-assistant';
import { Guide } from './pages/guide/guide';

export const routes: Routes = [
  { path: '', component: Overview },
  { path: 'studio', component: Studio },
  { path: 'compare', component: Compare },
  { path: 'validate', component: Validate },
  { path: 'goal-seek', component: GoalSeek },
  { path: 'policy', component: RagAssistant },
  { path: 'guide', component: Guide },
  // Label-based URL aliases so the page names match their routes.
  { path: 'accuracy', component: Validate },
  { path: 'find-limits', component: GoalSeek },
  { path: 'regulations', component: RagAssistant },
  { path: '**', redirectTo: '' },
];
