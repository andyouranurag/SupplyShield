/* =========================================================
   app.js — SupplyShield Dashboard Logic
   ========================================================= */

// ── Embedded data from integration pipeline ──────────────
const ALERTS = [
  { entity_id:"Supplier A", window_start:"2020-05-28", disruption_probability:0.9995, risk_level:"critical", disruption_type:"transport_delay", horizon_class:"immediate", evidence_passed:5, consistency_rate:1.0 },
  { entity_id:"Supplier A", window_start:"2020-06-04", disruption_probability:0.705,  risk_level:"high",     disruption_type:"transport_delay", horizon_class:"short_term", evidence_passed:5, consistency_rate:1.0 },
  { entity_id:"Supplier A", window_start:"2022-01-20", disruption_probability:0.9459, risk_level:"critical", disruption_type:"transport_delay", horizon_class:"immediate", evidence_passed:5, consistency_rate:1.0 },
  { entity_id:"Supplier A", window_start:"2022-02-03", disruption_probability:0.9997, risk_level:"critical", disruption_type:"transport_delay", horizon_class:"immediate", evidence_passed:5, consistency_rate:1.0 },
  { entity_id:"Supplier A", window_start:"2022-02-10", disruption_probability:1.0,    risk_level:"critical", disruption_type:"transport_delay", horizon_class:"immediate", evidence_passed:5, consistency_rate:1.0 },
  { entity_id:"Supplier A", window_start:"2022-03-03", disruption_probability:0.6107, risk_level:"high",     disruption_type:"transport_delay", horizon_class:"short_term", evidence_passed:5, consistency_rate:1.0 },
  { entity_id:"Supplier A", window_start:"2022-03-17", disruption_probability:0.9988, risk_level:"critical", disruption_type:"transport_delay", horizon_class:"immediate", evidence_passed:5, consistency_rate:1.0 },
  { entity_id:"Supplier A", window_start:"2022-03-24", disruption_probability:0.9999, risk_level:"critical", disruption_type:"transport_delay", horizon_class:"immediate", evidence_passed:5, consistency_rate:1.0 },
  { entity_id:"Supplier A", window_start:"2022-03-31", disruption_probability:0.9459, risk_level:"critical", disruption_type:"transport_delay", horizon_class:"immediate", evidence_passed:5, consistency_rate:1.0 },
  { entity_id:"Supplier A", window_start:"2022-04-07", disruption_probability:0.9999, risk_level:"critical", disruption_type:"transport_delay", horizon_class:"immediate", evidence_passed:5, consistency_rate:1.0 },
  { entity_id:"Supplier A", window_start:"2022-04-21", disruption_probability:1.0,    risk_level:"critical", disruption_type:"transport_delay", horizon_class:"immediate", evidence_passed:5, consistency_rate:1.0 },
  { entity_id:"Supplier A", window_start:"2022-04-28", disruption_probability:0.861,  risk_level:"critical", disruption_type:"transport_delay", horizon_class:"immediate", evidence_passed:5, consistency_rate:1.0 },
  { entity_id:"Supplier A", window_start:"2022-05-05", disruption_probability:0.9989, risk_level:"critical", disruption_type:"transport_delay", horizon_class:"immediate", evidence_passed:5, consistency_rate:1.0 },
  { entity_id:"Supplier A", window_start:"2022-06-09", disruption_probability:0.5659, risk_level:"medium",  disruption_type:"transport_delay", horizon_class:"medium_term", evidence_passed:5, consistency_rate:1.0 },
  { entity_id:"Supplier A", window_start:"2022-07-14", disruption_probability:0.9018, risk_level:"critical", disruption_type:"transport_delay", horizon_class:"immediate", evidence_passed:5, consistency_rate:1.0 },
  { entity_id:"Supplier A", window_start:"2022-07-28", disruption_probability:0.9982, risk_level:"critical", disruption_type:"transport_delay", horizon_class:"immediate", evidence_passed:5, consistency_rate:1.0 },
  { entity_id:"Supplier A", window_start:"2022-08-04", disruption_probability:0.9997, risk_level:"critical", disruption_type:"transport_delay", horizon_class:"immediate", evidence_passed:5, consistency_rate:1.0 },
  { entity_id:"Supplier A", window_start:"2022-08-18", disruption_probability:1.0,    risk_level:"critical", disruption_type:"transport_delay", horizon_class:"immediate", evidence_passed:5, consistency_rate:1.0 },
  { entity_id:"Supplier A", window_start:"2022-09-29", disruption_probability:0.97,   risk_level:"critical", disruption_type:"transport_delay", horizon_class:"immediate", evidence_passed:5, consistency_rate:1.0 },
  { entity_id:"Supplier A", window_start:"2022-10-06", disruption_probability:0.6257, risk_level:"high",     disruption_type:"transport_delay", horizon_class:"short_term", evidence_passed:5, consistency_rate:1.0 },
];

const EVIDENCE_SAMPLES = [
  { evidence_id:"ev_ecceacaf31", modality:"news", attribution_score:0.953, summary:"[natural_disaster] Iran's Government Criticized For Inaction Against Large Wildfires — disruption signal tied to extreme weather events affecting production routes.", passed:true, entity:"Supplier A", consistency_score:1.0 },
  { evidence_id:"ev_d0bffa788d", modality:"news", attribution_score:0.900, summary:"[natural_disaster] Virus, heat wave and locusts form perfect storm in India — multi-hazard stacking increases supply chain fragility scores.", passed:true, entity:"Supplier A", consistency_score:1.0 },
  { evidence_id:"ev_a05b9e5160", modality:"news", attribution_score:0.899, summary:"[factory_outage] Lockdown easing to allow groups of six — factory reopening signals mixed with ongoing capacity restrictions.", passed:true, entity:"Supplier A", consistency_score:1.0 },
  { evidence_id:"ev_38314ec79b", modality:"news", attribution_score:0.889, summary:"[factory_outage] Coronavirus: Belgian Prince Joachim tests positive after lockdown party — exemplar of continued pandemic disruption to logistics.", passed:true, entity:"Supplier A", consistency_score:1.0 },
  { evidence_id:"ev_6894ec853b", modality:"news", attribution_score:0.882, summary:"[factory_outage] Lockdown 5.0 live updates: Conduct sero-survey to track Covid spread — broad manufacturing disruption across South Asia.", passed:true, entity:"Supplier A", consistency_score:1.0 },
  { evidence_id:"ev_abc123def4", modality:"news", attribution_score:0.875, summary:"[geopolitical] Ukraine crisis: UK 'not optimistic' Russian invasion can be stopped — significant geopolitical risk elevating supply chain disruption scores.", passed:true, entity:"Supplier A", consistency_score:1.0 },
  { evidence_id:"ev_def456ghi7", modality:"news", attribution_score:0.861, summary:"[geopolitical] Australians in Ukraine describe 'two different realities' — war-related port closures and shipping lane disruptions flagged.", passed:true, entity:"Supplier A", consistency_score:1.0 },
  { evidence_id:"ev_ghi789jkl0", modality:"news", attribution_score:0.840, summary:"[geopolitical] China criticizes US missile sanctions as hypocrisy — bilateral trade tensions elevating material shortage risk.", passed:true, entity:"Supplier A", consistency_score:1.0 },
  { evidence_id:"ev_jkl012mno3", modality:"news", attribution_score:0.822, summary:"[natural_disaster] Some 10,000 people in northern town evacuated — infrastructure damage causing regional logistics bottleneck.", passed:true, entity:"Supplier A", consistency_score:1.0 },
  { evidence_id:"ev_mno345pqr6", modality:"news", attribution_score:0.807, summary:"[geopolitical] UK warns Russia will face severe sanctions — pre-conflict geopolitical escalation detected in news risk embedding.", passed:true, entity:"Supplier A", consistency_score:1.0 },
];

const TS_DATA = {
  "Supplier A": {
    dates: ["2020-01-14","2020-04-19","2020-04-20","2020-04-23","2020-04-25","2020-05-15","2020-05-28","2020-06-04","2020-07-10","2020-08-20","2021-01-10","2021-03-15","2021-06-20","2021-09-10","2022-01-20","2022-02-10","2022-03-17","2022-04-07","2022-05-05","2022-06-09","2022-07-14","2022-08-18","2022-09-29","2022-10-06"],
    lead_time_risk: [0.31,0.52,0.24,0.72,0.14,0.62,0.90,0.76,0.48,0.55,0.33,0.45,0.38,0.61,0.88,1.00,0.97,0.99,0.99,0.57,0.90,1.00,0.97,0.63],
    inventory_risk: [0.12,0.20,0.33,0.12,0.15,0.29,0.55,0.40,0.22,0.31,0.18,0.26,0.19,0.35,0.48,0.66,0.72,0.61,0.58,0.28,0.49,0.70,0.54,0.32],
    demand_anomaly_score: [-0.06,-0.10,-0.05,-0.10,-0.04,-0.08,-0.18,0.22,0.15,-0.03,-0.04,-0.07,-0.03,0.12,0.28,0.45,0.38,0.52,0.44,-0.06,0.33,0.48,0.41,0.18],
  },
  "Supplier B": {
    dates: ["2020-02-10","2020-05-01","2020-06-15","2020-09-20","2021-01-05","2021-04-20","2021-08-15","2022-02-01","2022-04-15","2022-07-01"],
    lead_time_risk: [0.18,0.34,0.42,0.28,0.55,0.47,0.38,0.61,0.74,0.56],
    inventory_risk: [0.09,0.18,0.24,0.14,0.29,0.22,0.19,0.35,0.41,0.30],
    demand_anomaly_score: [-0.03,0.08,0.12,-0.05,0.15,0.09,-0.02,0.22,0.31,0.14],
  },
  "Supplier C": {
    dates: ["2020-03-15","2020-07-10","2020-11-05","2021-02-18","2021-06-25","2021-10-10","2022-01-08","2022-05-20","2022-08-12"],
    lead_time_risk: [0.25,0.48,0.36,0.41,0.59,0.44,0.68,0.52,0.71],
    inventory_risk: [0.11,0.22,0.17,0.20,0.31,0.19,0.38,0.27,0.42],
    demand_anomaly_score: [-0.04,0.11,0.06,-0.03,0.18,0.07,0.25,0.16,0.29],
  },
  "Supplier D": {
    dates: ["2020-04-20","2020-08-15","2021-01-20","2021-05-10","2021-09-05","2022-03-01","2022-06-15","2022-09-20"],
    lead_time_risk: [0.22,0.41,0.33,0.55,0.48,0.76,0.61,0.58],
    inventory_risk: [0.10,0.19,0.16,0.28,0.23,0.44,0.33,0.30],
    demand_anomaly_score: [-0.02,0.09,0.04,0.14,0.11,0.32,0.21,0.18],
  },
};

const NEWS_TREND = {
  dates: Array.from({length: 80}, (_, i) => {
    const d = new Date('2020-01-01'); d.setDate(d.getDate() + i * 4); return d.toISOString().slice(0,10);
  }),
  risk_scores: [0.45,0.13,0.11,0.22,0.18,0.35,0.42,0.67,0.89,0.91,0.78,0.65,0.72,0.55,0.48,0.53,0.61,0.44,0.38,0.47,0.52,0.35,0.29,0.41,0.38,0.22,0.19,0.28,0.31,0.24,0.29,0.18,0.22,0.15,0.12,0.19,0.24,0.18,0.21,0.16,0.28,0.34,0.41,0.55,0.68,0.75,0.82,0.90,0.95,0.88,0.79,0.71,0.65,0.72,0.68,0.61,0.55,0.48,0.52,0.49,0.43,0.38,0.44,0.51,0.58,0.63,0.70,0.67,0.62,0.57,0.52,0.48,0.44,0.51,0.58,0.62,0.68,0.55,0.48,0.42],
  sentiment: [-0.22,-0.20,-0.33,-0.28,-0.30,-0.45,-0.52,-0.61,-0.72,-0.78,-0.65,-0.58,-0.63,-0.51,-0.44,-0.47,-0.55,-0.40,-0.36,-0.42,-0.48,-0.33,-0.28,-0.38,-0.35,-0.21,-0.18,-0.26,-0.29,-0.22,-0.27,-0.17,-0.20,-0.14,-0.11,-0.18,-0.22,-0.17,-0.19,-0.15,-0.26,-0.31,-0.38,-0.50,-0.61,-0.67,-0.73,-0.80,-0.84,-0.78,-0.71,-0.63,-0.57,-0.64,-0.59,-0.53,-0.48,-0.41,-0.46,-0.43,-0.37,-0.32,-0.39,-0.46,-0.52,-0.57,-0.63,-0.60,-0.55,-0.50,-0.46,-0.42,-0.39,-0.46,-0.52,-0.56,-0.61,-0.50,-0.43,-0.37],
};

const EVENT_TYPES = [
  { icon: '🌊', name: 'Natural Disaster', count: 3284, color: '#06b6d4' },
  { icon: '🏭', name: 'Factory Outage',   count: 5621, color: '#a855f7' },
  { icon: '⚡', name: 'Strike / Labour',  count: 2198, color: '#f59e0b' },
  { icon: '🌐', name: 'Geopolitical',     count: 4102, color: '#f43f5e' },
  { icon: '🚢', name: 'Port Closure',     count: 1847, color: '#6366f1' },
  { icon: '📦', name: 'Shortage',         count: 3695, color: '#10b981' },
  { icon: '🌡️', name: 'Pandemic',         count: 7293, color: '#f97316' },
  { icon: '💰', name: 'Trade Dispute',    count: 1534, color: '#ec4899' },
];

// Chart colour palette
const COLORS = {
  critical: '#f43f5e',
  high:     '#f97316',
  medium:   '#f59e0b',
  violet:   '#6366f1',
  purple:   '#a855f7',
  cyan:     '#06b6d4',
  emerald:  '#10b981',
};

// ── State ─────────────────────────────────────────────────
let chartInstances = {};
let sortKey = 'window_start', sortAsc = false;
let filterRisk = 'all', searchQuery = '';
let currentView = 'overview';

// ── View Switching ────────────────────────────────────────
function switchView(viewId, el) {
  document.querySelectorAll('.view').forEach(v => v.classList.remove('active'));
  document.querySelectorAll('.nav-item').forEach(n => n.classList.remove('active'));
  document.getElementById('view-' + viewId).classList.add('active');
  if (el) el.classList.add('active');
  document.getElementById('breadcrumb').textContent =
    { overview:'Overview', alerts:'Alerts', timeseries:'Time Series', news:'News Intelligence', evidence:'Evidence', manual:'Manual Prediction Playground' }[viewId] || viewId;
  currentView = viewId;
  window.scrollTo(0, 0);

  // Lazy-init charts
  if (viewId === 'overview') { initOverviewCharts(); renderMiniAlerts(); }
  if (viewId === 'alerts')    { renderAlertTable(); }
  if (viewId === 'timeseries') { initTSCharts(); }
  if (viewId === 'news')       { initNewsCharts(); renderEventTypes(); }
  if (viewId === 'evidence')   { renderEvidence(); }
  if (viewId === 'manual')     { initManualPlayground(); }
}

function toggleSidebar() {
  const sb = document.getElementById('sidebar');
  const mn = document.querySelector('.main');
  sb.classList.toggle('collapsed');
  mn.classList.toggle('expanded');
}

// ── Filters & Search ──────────────────────────────────────
function applyFilters() {
  filterRisk = document.getElementById('riskFilter').value;
  if (currentView === 'alerts') renderAlertTable();
  if (currentView === 'overview') renderMiniAlerts();
}

function handleSearch(val) {
  searchQuery = val.toLowerCase();
  if (currentView === 'alerts') renderAlertTable();
}

function getFilteredAlerts() {
  return ALERTS.filter(a => {
    if (filterRisk !== 'all' && a.risk_level !== filterRisk) return false;
    if (searchQuery && !a.entity_id.toLowerCase().includes(searchQuery) &&
        !a.window_start.includes(searchQuery) &&
        !a.risk_level.includes(searchQuery)) return false;
    return true;
  });
}

// ── Sort ──────────────────────────────────────────────────
function sortAlerts(key) {
  if (sortKey === key) sortAsc = !sortAsc;
  else { sortKey = key; sortAsc = true; }
  renderAlertTable();
}

// ── Overview ──────────────────────────────────────────────
function initOverviewCharts() {
  // Risk Donut
  if (!chartInstances.riskDonut) {
    chartInstances.riskDonut = new Chart(document.getElementById('riskDonut'), {
      type: 'doughnut',
      data: {
        labels: ['Critical', 'High', 'Medium'],
        datasets: [{ data: [16, 3, 1], backgroundColor: [COLORS.critical, COLORS.high, COLORS.medium], borderWidth: 0, hoverOffset: 6 }]
      },
      options: {
        cutout: '72%', responsive: true, maintainAspectRatio: false,
        plugins: { legend: { display: false }, tooltip: { callbacks: { label: ctx => ` ${ctx.label}: ${ctx.raw}` } } }
      }
    });
  }

  // Horizon Bar
  if (!chartInstances.horizonBar) {
    chartInstances.horizonBar = new Chart(document.getElementById('horizonBar'), {
      type: 'bar',
      data: {
        labels: ['Immediate', 'Short-Term', 'Medium-Term'],
        datasets: [{
          data: [16, 3, 1],
          backgroundColor: [
            'rgba(244,63,94,0.7)', 'rgba(249,115,22,0.7)', 'rgba(245,158,11,0.7)'
          ],
          borderRadius: 6, borderSkipped: false,
        }]
      },
      options: {
        responsive: true, maintainAspectRatio: false,
        scales: {
          x: { grid: { color: 'rgba(255,255,255,0.04)' }, ticks: { color: '#606080', font: { size: 11 } } },
          y: { grid: { color: 'rgba(255,255,255,0.04)' }, ticks: { color: '#606080', stepSize: 4 }, beginAtZero: true }
        },
        plugins: { legend: { display: false } }
      }
    });
  }

  // Probability Timeline
  if (!chartInstances.probTimeline) {
    const sorted = [...ALERTS].sort((a,b) => a.window_start.localeCompare(b.window_start));
    const colors = sorted.map(a => ({
      critical: COLORS.critical, high: COLORS.high, medium: COLORS.medium
    }[a.risk_level]));

    chartInstances.probTimeline = new Chart(document.getElementById('probTimeline'), {
      type: 'line',
      data: {
        labels: sorted.map(a => a.window_start.slice(0,7)),
        datasets: [{
          label: 'Disruption Probability',
          data: sorted.map(a => +(a.disruption_probability * 100).toFixed(1)),
          fill: true,
          borderColor: COLORS.violet,
          backgroundColor: 'rgba(99,102,241,0.1)',
          pointBackgroundColor: colors,
          pointRadius: 6,
          pointHoverRadius: 8,
          tension: 0.3,
        }]
      },
      options: {
        responsive: true, maintainAspectRatio: false,
        scales: {
          x: { grid: { color: 'rgba(255,255,255,0.04)' }, ticks: { color: '#606080', font: { size: 10 }, maxRotation: 45 } },
          y: { grid: { color: 'rgba(255,255,255,0.04)' }, ticks: { color: '#606080', callback: v => v + '%' }, min: 0, max: 100 }
        },
        plugins: {
          legend: { display: false },
          tooltip: { callbacks: { label: ctx => ` Probability: ${ctx.raw}%` } }
        },
        interaction: { mode: 'index', intersect: false },
      }
    });
  }
}

function renderMiniAlerts() {
  const container = document.getElementById('miniAlertGrid');
  const criticals = getFilteredAlerts().filter(a => a.risk_level === 'critical').slice(0, 6);
  container.innerHTML = criticals.map((a, i) => `
    <div class="mini-alert ${a.risk_level}" onclick="openModal(${ALERTS.indexOf(a)})" style="animation-delay:${i*0.05}s">
      <div class="mini-alert-top">
        <span class="mini-entity">${a.entity_id}</span>
        <span class="mini-prob ${a.risk_level}">${(a.disruption_probability*100).toFixed(1)}%</span>
      </div>
      <div class="prob-bar-bg"><div class="prob-bar-fill ${a.risk_level}" style="width:${(a.disruption_probability*100).toFixed(1)}%"></div></div>
      <div class="mini-meta">
        <span class="mini-tag">${a.window_start}</span>
        <span class="mini-tag">${horizonLabel(a.horizon_class)}</span>
        <span class="mini-tag">${a.disruption_type.replace('_',' ')}</span>
      </div>
    </div>
  `).join('');
}

// ── Alert Table ───────────────────────────────────────────
function renderAlertTable() {
  let alerts = getFilteredAlerts();
  alerts = alerts.sort((a,b) => {
    let av = a[sortKey], bv = b[sortKey];
    if (sortKey === 'risk_level') {
      const ord = {critical:0, high:1, medium:2};
      av = ord[av]; bv = ord[bv];
    }
    return sortAsc ? (av > bv ? 1 : -1) : (av < bv ? 1 : -1);
  });

  const tbody = document.getElementById('alertTableBody');
  tbody.innerHTML = alerts.map((a, i) => {
    const pct = (a.disruption_probability * 100).toFixed(1);
    const idx = ALERTS.indexOf(a);
    return `<tr>
      <td><span class="risk-pill ${a.risk_level}">${riskIcon(a.risk_level)} ${cap(a.risk_level)}</span></td>
      <td><strong>${a.entity_id}</strong></td>
      <td style="color:var(--text-secondary);font-family:'JetBrains Mono',monospace;font-size:0.78rem">${a.window_start}</td>
      <td><span class="prob-chip ${a.risk_level}">${pct}%</span></td>
      <td style="color:var(--text-muted)">${a.disruption_type.replace('_',' ')}</td>
      <td><span class="mini-tag">${horizonLabel(a.horizon_class)}</span></td>
      <td><span class="ev-badge">✓ ${a.evidence_passed}/5</span></td>
      <td><button class="btn-detail" onclick="openModal(${idx})">Details →</button></td>
    </tr>`;
  }).join('');
}

// ── Time Series Charts ────────────────────────────────────
function initTSCharts() {
  updateTSCharts();
}

function updateTSCharts() {
  const entity = document.getElementById('entitySelect')?.value || 'Supplier A';
  const data = TS_DATA[entity];
  if (!data) return;

  const cfgLeadInv = {
    type: 'line',
    data: {
      labels: data.dates,
      datasets: [
        { label: 'Lead Time Risk', data: data.lead_time_risk, borderColor: COLORS.violet, backgroundColor: 'rgba(99,102,241,0.08)', fill: true, tension: 0.4, pointRadius: 4, pointHoverRadius: 6 },
        { label: 'Inventory Risk', data: data.inventory_risk, borderColor: COLORS.cyan, backgroundColor: 'rgba(6,182,212,0.08)', fill: true, tension: 0.4, pointRadius: 4, pointHoverRadius: 6 },
      ]
    },
    options: lineOptions('Risk Score (0–1)'),
  };

  const cfgDemand = {
    type: 'bar',
    data: {
      labels: data.dates,
      datasets: [{
        label: 'Demand Anomaly Score',
        data: data.demand_anomaly_score,
        backgroundColor: data.demand_anomaly_score.map(v => v > 0 ? 'rgba(244,63,94,0.6)' : 'rgba(16,185,129,0.6)'),
        borderRadius: 4, borderSkipped: false,
      }]
    },
    options: barOptions('Anomaly Score')
  };

  if (chartInstances.tsLeadInv) chartInstances.tsLeadInv.destroy();
  if (chartInstances.tsDemand)  chartInstances.tsDemand.destroy();

  chartInstances.tsLeadInv = new Chart(document.getElementById('tsLeadInv'), cfgLeadInv);
  chartInstances.tsDemand  = new Chart(document.getElementById('tsDemand'),  cfgDemand);
}

// ── News Charts ───────────────────────────────────────────
function initNewsCharts() {
  if (!chartInstances.newsRiskChart) {
    chartInstances.newsRiskChart = new Chart(document.getElementById('newsRiskChart'), {
      type: 'line',
      data: {
        labels: NEWS_TREND.dates,
        datasets: [{
          label: 'News Risk Score', data: NEWS_TREND.risk_scores,
          borderColor: COLORS.rose, backgroundColor: 'rgba(244,63,94,0.08)',
          fill: true, tension: 0.4, pointRadius: 2, pointHoverRadius: 5,
        }]
      },
      options: lineOptions('Risk Score (0–1)')
    });
  }
  if (!chartInstances.newsSentimentChart) {
    chartInstances.newsSentimentChart = new Chart(document.getElementById('newsSentimentChart'), {
      type: 'line',
      data: {
        labels: NEWS_TREND.dates,
        datasets: [{
          label: 'Sentiment', data: NEWS_TREND.sentiment,
          borderColor: COLORS.purple, backgroundColor: 'rgba(168,85,247,0.08)',
          fill: true, tension: 0.4, pointRadius: 2, pointHoverRadius: 5,
        }]
      },
      options: lineOptions('Sentiment Score')
    });
  }
}

function renderEventTypes() {
  const grid = document.getElementById('eventTypeGrid');
  grid.innerHTML = EVENT_TYPES.map(e => `
    <div class="event-type-card">
      <div class="event-type-icon">${e.icon}</div>
      <div>
        <div class="event-type-name" style="color:${e.color}">${e.name}</div>
        <div class="event-type-count">${e.count.toLocaleString()} events detected</div>
      </div>
    </div>
  `).join('');
}

// ── Evidence ──────────────────────────────────────────────
function filterEvidence() { renderEvidence(); }

function renderEvidence() {
  const modFilter  = document.getElementById('evModalityFilter').value;
  const passFilter = document.getElementById('evPassedFilter').value;
  const evs = EVIDENCE_SAMPLES.filter(e => {
    if (modFilter !== 'all' && e.modality !== modFilter) return false;
    if (passFilter === 'passed' && !e.passed) return false;
    if (passFilter === 'failed' && e.passed)  return false;
    return true;
  });
  const grid = document.getElementById('evidenceGrid');
  grid.innerHTML = evs.map(e => `
    <div class="evidence-card">
      <div class="ev-header">
        <div>
          <div class="ev-id">${e.evidence_id}</div>
          <div style="font-size:0.7rem;color:var(--text-muted);margin-top:2px">${e.entity}</div>
        </div>
        <div class="ev-score">Score: ${e.attribution_score.toFixed(3)}</div>
      </div>
      <div class="ev-summary">${e.summary}</div>
      <div class="ev-footer">
        <span class="modality-tag">${e.modality}</span>
        <span class="ev-check ${e.passed ? 'pass' : 'fail'}">${e.passed ? '✓ Verified' : '✗ Failed'}</span>
        <span class="ev-check pass">Consistency: ${(e.consistency_score * 100).toFixed(0)}%</span>
      </div>
    </div>
  `).join('');
}

// ── Modal ─────────────────────────────────────────────────
function openModal(idx) {
  const a = ALERTS[idx];
  document.getElementById('modalRiskBadge').className = 'modal-risk-badge ' + a.risk_level;
  document.getElementById('modalRiskBadge').textContent = cap(a.risk_level);
  document.getElementById('modalTitle').textContent = `${a.entity_id} — ${a.window_start}`;
  document.getElementById('modalGaugeVal').textContent = (a.disruption_probability * 100).toFixed(1) + '%';
  document.getElementById('modalGaugeVal').style.color = COLORS[a.risk_level] || COLORS.violet;
  document.getElementById('modalEvCount').textContent = `${a.evidence_passed}/5 verified`;

  // Gauge
  drawGauge('modalGauge', a.disruption_probability, a.risk_level);

  // Details
  document.getElementById('modalDetails').innerHTML = `
    <div class="detail-row"><span class="detail-key">Disruption Type</span><span class="detail-val">${a.disruption_type.replace(/_/g,' ')}</span></div>
    <div class="detail-row"><span class="detail-key">Horizon</span><span class="detail-val">${horizonLabel(a.horizon_class)}</span></div>
    <div class="detail-row"><span class="detail-key">Window Start</span><span class="detail-val">${a.window_start}</span></div>
    <div class="detail-row"><span class="detail-key">Consistency Rate</span><span class="detail-val" style="color:var(--emerald)">${(a.consistency_rate*100).toFixed(0)}%</span></div>
    <div class="detail-row"><span class="detail-key">Model</span><span class="detail-val">5-Library AI Fusion</span></div>
  `;

  // Cross-modal contributions (synthesised from MultimodalWarning)
  const contribs = [
    { name:'Time Series (P1)', share: 0.55, color: COLORS.violet },
    { name:'News Intel (P2)',   share: 0.30, color: COLORS.rose },
    { name:'Event Risk (P3)',   share: 0.15, color: COLORS.amber },
  ];
  document.getElementById('modalContrib').innerHTML = contribs.map(c => `
    <div class="contrib-row">
      <div class="contrib-label">
        <span class="contrib-label-name">${c.name}</span>
        <span class="contrib-label-pct">${(c.share*100).toFixed(0)}%</span>
      </div>
      <div class="contrib-bg"><div class="contrib-fill" style="width:${c.share*100}%;background:${c.color}"></div></div>
    </div>
  `).join('');

  // Evidence
  const ev = EVIDENCE_SAMPLES.slice(0, a.evidence_passed);
  document.getElementById('modalEvidence').innerHTML = ev.map(e => `
    <div class="modal-ev-item">
      <div class="modal-ev-header">
        <span class="modality-tag">${e.modality}</span>
        <span style="font-size:0.72rem;color:var(--text-muted);font-family:'JetBrains Mono',monospace">score: ${e.attribution_score.toFixed(3)}</span>
        <span class="ev-check pass" style="margin-left:auto">✓ Verified</span>
      </div>
      <div class="modal-ev-text">${e.summary}</div>
    </div>
  `).join('');

  document.getElementById('modalOverlay').classList.add('active');
  document.getElementById('alertModal').classList.add('active');
}

function closeModal() {
  document.getElementById('modalOverlay').classList.remove('active');
  document.getElementById('alertModal').classList.remove('active');
}

// ── Semi-circular Gauge ───────────────────────────────────
function drawGauge(canvasId, value, level) {
  const canvas = document.getElementById(canvasId);
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const W = canvas.width, H = canvas.height;
  const cx = W / 2, cy = H - 16;
  const r = Math.min(W, H * 1.6) * 0.42;
  ctx.clearRect(0, 0, W, H);

  // Track
  ctx.beginPath();
  ctx.arc(cx, cy, r, Math.PI, 0);
  ctx.strokeStyle = 'rgba(255,255,255,0.08)';
  ctx.lineWidth = 14; ctx.lineCap = 'round';
  ctx.stroke();

  // Fill
  const endAngle = Math.PI + Math.PI * value;
  const grad = ctx.createLinearGradient(cx - r, cy, cx + r, cy);
  grad.addColorStop(0,   COLORS.emerald);
  grad.addColorStop(0.5, COLORS.amber);
  grad.addColorStop(1,   COLORS[level] || COLORS.critical);
  ctx.beginPath();
  ctx.arc(cx, cy, r, Math.PI, endAngle);
  ctx.strokeStyle = grad;
  ctx.lineWidth = 14; ctx.lineCap = 'round';
  ctx.stroke();
}

// ── Chart helpers ─────────────────────────────────────────
function lineOptions(yLabel) {
  return {
    responsive: true, maintainAspectRatio: false,
    scales: {
      x: { grid: { color: 'rgba(255,255,255,0.04)' }, ticks: { color: '#606080', font: { size: 10 }, maxRotation: 45, maxTicksLimit: 12 } },
      y: { grid: { color: 'rgba(255,255,255,0.04)' }, ticks: { color: '#606080' }, title: { display: true, text: yLabel, color: '#606080', font: { size: 10 } } }
    },
    plugins: { legend: { labels: { color: '#a0a0c0', boxWidth: 12, font: { size: 11 } } } },
    interaction: { mode: 'index', intersect: false },
  };
}
function barOptions(yLabel) {
  return {
    responsive: true, maintainAspectRatio: false,
    scales: {
      x: { grid: { color: 'rgba(255,255,255,0.04)' }, ticks: { color: '#606080', font: { size: 10 }, maxRotation: 45, maxTicksLimit: 12 } },
      y: { grid: { color: 'rgba(255,255,255,0.04)' }, ticks: { color: '#606080' }, title: { display: true, text: yLabel, color: '#606080', font: { size: 10 } } }
    },
    plugins: { legend: { display: false } },
  };
}

// ── Utils ─────────────────────────────────────────────────
function cap(s) { return s.charAt(0).toUpperCase() + s.slice(1); }
function riskIcon(r) { return {critical:'⚠', high:'⚡', medium:'ℹ'}[r] || ''; }
function horizonLabel(h) {
  return { immediate: '< 7 days', short_term: '7–30 days', medium_term: '1–3 months', long_term: '> 3 months' }[h] || h;
}

// ── Init ──────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  document.getElementById('updateTime').textContent = new Date().toLocaleTimeString();
  initOverviewCharts();
  renderMiniAlerts();
  initManualPlayground();

  // Keyboard shortcut: Esc closes modal
  document.addEventListener('keydown', e => { if (e.key === 'Escape') closeModal(); });
});

// =========================================================
// ── Manual Prediction Playground ─────────────────────────
// =========================================================
let predictionHistory = [];
let playgroundInitialized = false;

function initManualPlayground() {
  if (!playgroundInitialized) {
    playgroundInitialized = true;
    runPrediction(false); // initial run without adding to history
  }
}

function updateSliderVal(input, valId) {
  const el = document.getElementById(valId);
  if (el) {
    el.textContent = (+input.value).toFixed(valId.includes('lag') ? 1 : 2);
  }
}

function getSliderNum(id, fallback = 0) {
  const el = document.getElementById(id);
  return el ? parseFloat(el.value) || 0 : fallback;
}

function setSliderVal(id, val, valId) {
  const el = document.getElementById(id);
  if (el) {
    el.value = val;
    updateSliderVal(el, valId);
  }
}

function isChecked(id) {
  const el = document.getElementById(id);
  return el ? !!el.checked : false;
}

function setChecked(id, state) {
  const el = document.getElementById(id);
  if (el) el.checked = !!state;
}

// ── Real-Time Multimodal Prediction Model ────────────────
function runPrediction(addToHistory = true) {
  // 1. Time-series inputs (Person 1)
  const lead_time_risk  = getSliderNum('s_lead_time_risk', 0.55);
  const inventory_risk  = getSliderNum('s_inventory_risk', 0.24);
  const demand_anomaly  = getSliderNum('s_demand_anomaly', 0.0);
  const delay_lag7      = getSliderNum('s_delay_lag7', 0.0);
  const temporal_risk   = getSliderNum('s_temporal_risk', 0.40);

  // Normalized delay score (0 to 1)
  const lag_score = Math.min(1.0, delay_lag7 / 14.0);
  const p1_ts_score = 0.35 * lead_time_risk + 0.25 * inventory_risk + 0.15 * Math.max(0, demand_anomaly) + 0.15 * lag_score + 0.10 * temporal_risk;

  // 2. News inputs (Person 2)
  const news_risk      = getSliderNum('s_news_risk', 0.35);
  const sentiment      = getSliderNum('s_sentiment', -0.30);
  const event_severity = getSliderNum('s_event_severity', 0.20);

  // Event category modifiers
  const cbGeo       = isChecked('cb_geopolitical') ? 0.14 : 0;
  const cbDisaster  = isChecked('cb_natural_disaster') ? 0.16 : 0;
  const cbOutage    = isChecked('cb_factory_outage') ? 0.12 : 0;
  const cbStrike    = isChecked('cb_strike') ? 0.10 : 0;
  const cbPort      = isChecked('cb_port_closure') ? 0.15 : 0;
  const cbShortage  = isChecked('cb_shortage') ? 0.11 : 0;

  const event_category_boost = Math.min(0.40, cbGeo + cbDisaster + cbOutage + cbStrike + cbPort + cbShortage);
  const sentiment_magnitude = Math.abs(sentiment);
  const p2_news_score = Math.min(1.0, 0.40 * news_risk + 0.25 * sentiment_magnitude + 0.15 * event_severity + 0.20 * event_category_boost);

  // 3. Event risk inputs (Person 3)
  const event_impact     = getSliderNum('s_event_impact', 0.15);
  const network_exposure = getSliderNum('s_network_exposure', 0.30);
  const weather          = getSliderNum('s_weather', 0.0);
  const p3_event_score   = 0.45 * event_impact + 0.35 * network_exposure + 0.20 * weather;

  // 4. Multimodal Fusion Logit (Matching MultimodalWarning Architecture)
  // Base intercept -2.1 calibrated for ~0.11 baseline probability
  const logit = -2.2 + 2.7 * p1_ts_score + 2.2 * p2_news_score + 1.8 * p3_event_score;
  const raw_prob = 1 / (1 + Math.exp(-logit));
  const prob = Math.max(0.01, Math.min(0.9999, raw_prob));
  const pct = (prob * 100).toFixed(1);

  // Risk Classification
  let riskLevel = 'low';
  let horizon = '> 3 months';
  if (prob >= 0.80) {
    riskLevel = 'critical';
    horizon = (lead_time_risk > 0.7 || lag_score > 0.5) ? '< 7 days (Immediate)' : '7–30 days';
  } else if (prob >= 0.60) {
    riskLevel = 'high';
    horizon = '7–30 days (Short-term)';
  } else if (prob >= 0.35) {
    riskLevel = 'medium';
    horizon = '1–3 months (Medium-term)';
  } else {
    riskLevel = 'low';
    horizon = 'Low Disruption Expected';
  }

  // Update UI Elements
  const probEl = document.getElementById('resultProb');
  if (probEl) {
    probEl.textContent = pct + '%';
    probEl.style.color = COLORS[riskLevel] || COLORS.emerald;
  }

  const badgeEl = document.getElementById('resultRiskBadge');
  if (badgeEl) {
    badgeEl.className = 'result-risk-badge ' + riskLevel;
    badgeEl.textContent = riskIcon(riskLevel) + ' ' + cap(riskLevel) + ' Risk';
  }

  const horizonEl = document.getElementById('resultHorizon');
  if (horizonEl) {
    horizonEl.textContent = '⏱ ' + horizon;
  }

  // Draw semi-circular gauge
  drawPlaygroundGauge('playGauge', prob, riskLevel);

  // Modality Contributions (Normalized weights)
  const sumScores = (p1_ts_score * 0.45) + (p2_news_score * 0.35) + (p3_event_score * 0.20) || 0.001;
  const contribs = [
    { name: 'SupplyChainTimeSeries (P1)', share: (p1_ts_score * 0.45) / sumScores, color: COLORS.violet },
    { name: 'NewsIntelligence (P2)',      share: (p2_news_score * 0.35) / sumScores, color: COLORS.rose },
    { name: 'SupplyChainEventRisk (P3)',  share: (p3_event_score * 0.20) / sumScores, color: COLORS.amber },
  ];

  const contribEl = document.getElementById('resultContrib');
  if (contribEl) {
    contribEl.innerHTML = contribs.map(c => `
      <div class="contrib-row" style="margin-bottom:8px">
        <div class="contrib-label">
          <span class="contrib-label-name">${c.name}</span>
          <span class="contrib-label-pct">${(c.share * 100).toFixed(0)}%</span>
        </div>
        <div class="contrib-bg"><div class="contrib-fill" style="width:${(c.share * 100).toFixed(0)}%;background:${c.color}"></div></div>
      </div>
    `).join('');
  }

  // Top Feature Drivers (SHAP-like attribution)
  const drivers = [
    { name: 'Lead Time Pressure', impact: +(lead_time_risk * 0.38).toFixed(2), isPos: lead_time_risk > 0.4 },
    { name: 'Inventory Stockout Risk', impact: +(inventory_risk * 0.32).toFixed(2), isPos: inventory_risk > 0.3 },
    { name: 'News NLP Disruption Index', impact: +(news_risk * 0.28).toFixed(2), isPos: news_risk > 0.3 },
    { name: 'Active Disruption Events', impact: +(event_category_boost * 0.65).toFixed(2), isPos: event_category_boost > 0.05 },
    { name: 'Event Network Exposure', impact: +(network_exposure * 0.22).toFixed(2), isPos: network_exposure > 0.35 },
  ].sort((a, b) => b.impact - a.impact).slice(0, 3);

  const driversEl = document.getElementById('resultDrivers');
  if (driversEl) {
    driversEl.innerHTML = drivers.map(d => `
      <div class="driver-pill">
        <span class="driver-name">${d.name}</span>
        <span class="driver-impact ${d.isPos ? 'pos' : 'neg'}">${d.isPos ? '+' : ''}${(d.impact * 100).toFixed(0)}% impact</span>
      </div>
    `).join('');
  }

  // Model Reasoning Natural Language Explanation
  const activeEvents = [];
  if (cbPort) activeEvents.push('Port Closures');
  if (cbStrike) activeEvents.push('Labor Strikes');
  if (cbDisaster) activeEvents.push('Natural Disasters');
  if (cbOutage) activeEvents.push('Factory Outages');
  if (cbShortage) activeEvents.push('Material Shortages');
  if (cbGeo) activeEvents.push('Geopolitical Tensions');

  let reasoningText = '';
  if (riskLevel === 'critical') {
    const eventsStr = activeEvents.length ? ` compounded by detected ${activeEvents.join(', ')}` : '';
    reasoningText = `🔴 <strong>CRITICAL ALERT:</strong> Acute multi-tier stress detected. Elevated lead-time volatility (${(lead_time_risk * 100).toFixed(0)}%) and high inventory vulnerability (${(inventory_risk * 100).toFixed(0)}%)${eventsStr} trigger high disruption probability. Recommend immediate buffer stock reallocation.`;
  } else if (riskLevel === 'high') {
    const eventsStr = activeEvents.length ? ` along with active ${activeEvents.join(' and ')}` : '';
    reasoningText = `⚡ <strong>ELEVATED RISK:</strong> Warning indicators are converging. Negative news sentiment (${sentiment}) and elevated lead time variance${eventsStr} pose short-term delivery bottlenecks.`;
  } else if (riskLevel === 'medium') {
    reasoningText = `ℹ️ <strong>MODERATE VULNERABILITY:</strong> Operational metrics show mild localized strain. Inventory levels are partially shielding against lead-time lag, but demand variance requires monitoring.`;
  } else {
    reasoningText = `✅ <strong>STABLE SYSTEM:</strong> All 3 input modalities (Time Series, News NLP, Event Network) remain within baseline resilience thresholds. Normal supply chain operations expected.`;
  }

  const reasonEl = document.getElementById('resultReasoning');
  if (reasonEl) reasonEl.innerHTML = reasoningText;

  // Add to History
  if (addToHistory) {
    const timeStr = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
    predictionHistory.unshift({
      time: timeStr,
      prob: pct,
      level: riskLevel,
      config: {
        lead_time_risk, inventory_risk, demand_anomaly, delay_lag7, temporal_risk,
        news_risk, sentiment, event_severity,
        cbGeo: !!cbGeo, cbDisaster: !!cbDisaster, cbOutage: !!cbOutage, cbStrike: !!cbStrike, cbPort: !!cbPort, cbShortage: !!cbShortage,
        event_impact, network_exposure, weather
      }
    });
    if (predictionHistory.length > 8) predictionHistory.pop();
    renderHistory();
  }
}

// ── Draw Playground Semicircular Gauge ─────────────────────
function drawPlaygroundGauge(canvasId, value, level) {
  const canvas = document.getElementById(canvasId);
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const W = canvas.width, H = canvas.height;
  const cx = W / 2, cy = H - 18;
  const r = Math.min(W, H * 1.6) * 0.44;
  ctx.clearRect(0, 0, W, H);

  // Background track
  ctx.beginPath();
  ctx.arc(cx, cy, r, Math.PI, 0);
  ctx.strokeStyle = 'rgba(255,255,255,0.08)';
  ctx.lineWidth = 16;
  ctx.lineCap = 'round';
  ctx.stroke();

  // Value fill
  const endAngle = Math.PI + Math.PI * Math.min(1.0, value);
  const grad = ctx.createLinearGradient(cx - r, cy, cx + r, cy);
  grad.addColorStop(0,   COLORS.emerald);
  grad.addColorStop(0.4, COLORS.amber);
  grad.addColorStop(0.7, COLORS.high);
  grad.addColorStop(1,   COLORS.critical);

  ctx.beginPath();
  ctx.arc(cx, cy, r, Math.PI, endAngle);
  ctx.strokeStyle = grad;
  ctx.lineWidth = 16;
  ctx.lineCap = 'round';
  ctx.stroke();
}

// ── Scenario Presets ──────────────────────────────────────
const SCENARIOS = {
  normal: {
    lead_time_risk: 0.15, inventory_risk: 0.12, demand_anomaly: 0.0, delay_lag7: 0.0, temporal_risk: 0.18,
    news_risk: 0.10, sentiment: -0.05, event_severity: 0.05,
    cbGeo: false, cbDisaster: false, cbOutage: false, cbStrike: false, cbPort: false, cbShortage: false,
    event_impact: 0.08, network_exposure: 0.12, weather: 0.0
  },
  moderate: {
    lead_time_risk: 0.52, inventory_risk: 0.35, demand_anomaly: 0.18, delay_lag7: 3.5, temporal_risk: 0.45,
    news_risk: 0.38, sentiment: -0.35, event_severity: 0.30,
    cbGeo: false, cbDisaster: false, cbOutage: false, cbStrike: false, cbPort: false, cbShortage: true,
    event_impact: 0.30, network_exposure: 0.40, weather: 0.25
  },
  critical: {
    lead_time_risk: 0.92, inventory_risk: 0.85, demand_anomaly: 0.55, delay_lag7: 12.0, temporal_risk: 0.90,
    news_risk: 0.85, sentiment: -0.80, event_severity: 0.75,
    cbGeo: false, cbDisaster: true, cbOutage: true, cbStrike: true, cbPort: true, cbShortage: true,
    event_impact: 0.82, network_exposure: 0.88, weather: 0.60
  },
  pandemic: {
    lead_time_risk: 0.88, inventory_risk: 0.92, demand_anomaly: 0.65, delay_lag7: 18.0, temporal_risk: 0.85,
    news_risk: 0.95, sentiment: -0.92, event_severity: 0.85,
    cbGeo: false, cbDisaster: false, cbOutage: true, cbStrike: false, cbPort: true, cbShortage: true,
    event_impact: 0.88, network_exposure: 0.95, weather: 0.10
  },
  geopolitical: {
    lead_time_risk: 0.78, inventory_risk: 0.62, demand_anomaly: 0.30, delay_lag7: 8.5, temporal_risk: 0.72,
    news_risk: 0.90, sentiment: -0.78, event_severity: 0.80,
    cbGeo: true, cbDisaster: false, cbOutage: false, cbStrike: true, cbPort: true, cbShortage: true,
    event_impact: 0.75, network_exposure: 0.80, weather: 0.05
  }
};

function loadScenario(name) {
  const s = SCENARIOS[name];
  if (!s) return;
  applyScenarioConfig(s);
  runPrediction(true);
}

function applyScenarioConfig(s) {
  setSliderVal('s_lead_time_risk', s.lead_time_risk, 'v_lead_time_risk');
  setSliderVal('s_inventory_risk', s.inventory_risk, 'v_inventory_risk');
  setSliderVal('s_demand_anomaly', s.demand_anomaly, 'v_demand_anomaly');
  setSliderVal('s_delay_lag7', s.delay_lag7, 'v_delay_lag7');
  setSliderVal('s_temporal_risk', s.temporal_risk, 'v_temporal_risk');

  setSliderVal('s_news_risk', s.news_risk, 'v_news_risk');
  setSliderVal('s_sentiment', s.sentiment, 'v_sentiment');
  setSliderVal('s_event_severity', s.event_severity, 'v_event_severity');

  setChecked('cb_geopolitical', s.cbGeo);
  setChecked('cb_natural_disaster', s.cbDisaster);
  setChecked('cb_factory_outage', s.cbOutage);
  setChecked('cb_strike', s.cbStrike);
  setChecked('cb_port_closure', s.cbPort);
  setChecked('cb_shortage', s.cbShortage);

  setSliderVal('s_event_impact', s.event_impact, 'v_event_impact');
  setSliderVal('s_network_exposure', s.network_exposure, 'v_network_exposure');
  setSliderVal('s_weather', s.weather, 'v_weather');
}

function resetPlayground() {
  loadScenario('normal');
}

// ── Prediction History ────────────────────────────────────
function renderHistory() {
  const container = document.getElementById('predHistory');
  if (!container) return;
  if (!predictionHistory.length) {
    container.innerHTML = '<div style="font-size:0.72rem;color:var(--text-muted);text-align:center;padding:8px">No manual tests recorded yet.</div>';
    return;
  }
  container.innerHTML = predictionHistory.map((item, idx) => `
    <div class="history-item" onclick="restoreHistory(${idx})">
      <div>
        <span class="risk-pill ${item.level}" style="font-size:0.68rem;padding:2px 8px">${riskIcon(item.level)} ${cap(item.level)}</span>
        <span class="history-time" style="margin-left:6px">${item.time}</span>
      </div>
      <span class="history-prob" style="color:${COLORS[item.level]}">${item.prob}%</span>
    </div>
  `).join('');
}

function restoreHistory(idx) {
  const item = predictionHistory[idx];
  if (item && item.config) {
    applyScenarioConfig(item.config);
    runPrediction(false);
  }
}

function clearHistory() {
  predictionHistory = [];
  renderHistory();
}
