import { Component } from '@angular/core';
import { RouterLink, RouterLinkActive } from '@angular/router';

@Component({
  selector: 'app-navbar',
  imports: [RouterLink, RouterLinkActive],
  templateUrl: './navbar.component.html',
  styleUrl: './navbar.component.css',
})
export class NavbarComponent {
  links = [
    {
      path: '/', label: 'Overview', exact: true,
      desc: 'What CHIRRP is and the headline results.',
    },
    {
      path: '/studio', label: 'Explore Scenarios', exact: false,
      desc: 'Describe a pumping change in plain language and see the groundwater result in milliseconds.',
    },
    {
      path: '/compare', label: 'Compare', exact: false,
      desc: 'Put 2–4 scenarios side by side — overlaid curves and a delta table.',
    },
    {
      path: '/validate', label: 'Accuracy', exact: false,
      desc: 'How the fast surrogate compares to MODFLOW (RMSE, R², and speed).',
    },
    {
      path: '/goal-seek', label: 'Find Limits', exact: false,
      desc: 'Set a drawdown limit and find the most pumping that stays under it.',
    },
    {
      path: '/policy', label: 'Regulations', exact: false,
      desc: 'Ask about Nebraska groundwater law — answers with cited sources.',
    },
    {
      path: '/guide', label: 'How to use', exact: false,
      desc: 'A two-minute orientation: the workflow, what each section does, and tips.',
    },
  ];
}
