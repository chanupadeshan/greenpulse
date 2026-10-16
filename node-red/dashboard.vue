<template>
  <main class="gp-shell">
    <header class="gp-heading">
      <div><div class="gp-eyebrow">GREENPULSE / GREENHOUSE INTELLIGENCE</div><h1>Continuous prediction.</h1><p>Stored DynamoDB readings, sensor trends, and model results updated automatically.</p></div>
      <span class="gp-pill" :class="connected ? 'gp-green' : 'gp-amber'">{{ connected ? 'Backend connected' : 'Waiting for backend' }}</span>
    </header>
    <nav class="gp-tabs" aria-label="Dashboard tabs"><a class="gp-tab-active" href="/dashboard/greenhouse" aria-current="page">Prediction</a><a href="/dashboard/assistant">AI assistant</a></nav>
    <div class="gp-toolbar">
      <div><strong>{{ config.device_id || 'Configure device ID' }}</strong><span>{{ config.region }} · {{ config.table }}</span></div>
      <div class="gp-buttons"><button @click="run('prediction')" :disabled="busy || !connected">Refresh prediction</button></div>
    </div>
    <div v-if="notice" class="gp-alert" role="status">{{ notice }}</div>
    <div v-if="config.demo" class="gp-alert"><strong>DEMO MODE</strong> — simulated sensors and weather with the real trained model. AWS, OpenWeather, and Groq are not called.</div>
    <div v-if="!connected" class="gp-alert">Start the GreenPulse backend to load readings. Waiting for a successful status response.</div>
    <div v-if="sensor.status === 'error'" class="gp-alert">{{ sensor.error }}</div>
    <div v-if="sensor.warnings?.length" class="gp-alert"><strong>Measurement checks</strong><ul><li v-for="w in sensor.warnings" :key="w">{{ w }}</li></ul></div>
    <section class="gp-grid gp-metrics" aria-label="Latest sensor readings">
      <article v-for="metric in metrics" :key="metric.key" class="gp-card">
        <div class="gp-label">{{ metric.label }}</div><div class="gp-value">{{ sensorValue(latest[metric.key]) }}<small v-if="latest[metric.key] != null">{{ metric.unit }}</small></div>
        <div class="gp-subtle">{{ latest[metric.key] == null ? 'Measurement unavailable' : 'Stored device measurement' }}</div>
      </article>
    </section>
    <div class="gp-meta">Latest reading: {{ date(latest.timestamp) }} <span v-if="sensor.age_minutes != null"> · {{ Math.round(sensor.age_minutes) }} minutes old</span> · {{ sensor.row_count || 0 }} recent readings</div>
    <section class="gp-grid gp-two">
      <article class="gp-card gp-prediction">
        <div class="gp-card-heading"><h2>Model prediction</h2><span class="gp-pill" :class="ml.status === 'ok' ? severityClass : 'gp-amber'">{{ ml.status === 'ok' ? (ml.is_anomaly ? ml.severity + ' severity' : 'Normal') : (ml.status || 'Waiting') }}</span></div>
        <div class="gp-result" :class="ml.status === 'ok' ? severityClass + '-text' : ''">{{ ml.status === 'ok' ? (ml.is_anomaly ? 'Unusual sensor pattern' : 'Normal sensor pattern') : 'Prediction unavailable' }}</div>
        <p v-if="ml.status !== 'ok'">{{ ml.error || 'Waiting for the first reading.' }}</p>
        <div v-if="ml.status === 'ok'" class="gp-score"><div><span>ANOMALY SCORE</span><strong>{{ number(ml.score) }}</strong></div><div><span>ACTIVE THRESHOLD</span><strong>{{ number(ml.threshold) }}</strong></div><div><span>SEVERITY</span><strong :class="severityClass + '-text'">{{ ml.severity }}</strong></div></div>
        <div v-if="ml.status === 'ok'" class="gp-meter" :class="severityClass + '-meter'" role="img" :aria-label="ml.severity + ' severity; score ' + number(ml.score) + ', threshold ' + number(ml.threshold)"><div :style="{width: Math.max(0, Math.min(100, ml.score / Math.max(ml.threshold * 2, 0.000001) * 100)) + '%'}"></div><span></span></div>
      </article>
      <article class="gp-card">
        <div class="gp-card-heading"><h2>Recent sensor trends</h2><select v-model="trend" aria-label="Select trend"><option value="air_temp_c">Air temperature</option><option value="humidity">Humidity</option><option value="soil_moisture_l1">Moisture · layer 1</option><option value="soil_moisture_l2">Moisture · layer 2</option><option value="soil_temp_l1_c">Soil temperature · layer 1</option><option value="soil_temp_l2_c">Soil temperature · layer 2</option></select></div>
        <div v-if="chartPoints.length" class="gp-chart"><div class="gp-chart-scale"><span>{{ chartMax.toFixed(1) }}</span><span>{{ chartMin.toFixed(1) }}</span></div><svg viewBox="0 0 600 140" preserveAspectRatio="none" role="img" :aria-label="'Recent ' + trend + ' trend'"><line x1="0" y1="139" x2="600" y2="139" stroke="#d9e6df"/><polyline :points="chartLine" fill="none" stroke="#217454" stroke-width="3" vector-effect="non-scaling-stroke"/><circle v-for="p in chartPoints" :key="p.x" :cx="p.x" :cy="p.y" r="3" fill="#217454"/></svg></div>
        <p v-else class="gp-empty">No measured values for this sensor yet.</p>
        <div class="gp-chart-dates"><span>{{ date(sensor.readings?.[0]?.timestamp) }}</span><span>{{ date(latest.timestamp) }}</span></div>
        <p class="gp-subtle">Most recent device records, ordered by timestamp. Null readings break the line.</p>
      </article>
    </section>
    <section class="gp-card gp-history"><h2>Run history</h2><div class="gp-table-scroll"><table><thead><tr><th>Completed</th><th>Workflow</th><th>Result</th><th>Run ID</th></tr></thead><tbody><tr v-for="run in predictionHistory" :key="run.id"><td>{{ date(run.completed_at) }}</td><td>{{ run.mode }}</td><td>{{ run.status }}</td><td class="gp-mono">{{ run.id.slice(0, 8) }}</td></tr><tr v-if="!predictionHistory.length"><td colspan="4">Waiting for the first analysis.</td></tr></tbody></table></div></section>
    <footer class="gp-footer">GREENPULSE AI <span>DynamoDB → sensor validation → saved anomaly model · AI chat runs separately</span></footer>
  </main>
</template>
<script>
export default {
  data() { return { data: {}, connected: false, notice: '', trend: 'air_temp_c', metrics: [
    {key:'air_temp_c',label:'Air temperature',unit:'°C'},{key:'humidity',label:'Air humidity',unit:'%'},
    {key:'soil_moisture_l1',label:'Moisture · layer 1',unit:'%'},{key:'soil_moisture_l2',label:'Moisture · layer 2',unit:'%'},
    {key:'soil_temp_l1_c',label:'Soil temp · layer 1',unit:'°C'},{key:'soil_temp_l2_c',label:'Soil temp · layer 2',unit:'°C'}
  ]} },
  computed: {
    config() { return this.data.config || {} },
    current() { return this.data.prediction || {} },
    predictionHistory() { return (this.data.history || []).filter(run => run.mode === 'prediction') },
    sensor() { return this.current.sensor_report || {} }, latest() { return this.sensor.latest || {} },
    ml() { return this.current.ml_report || {} },
    severityClass() { return !this.ml.is_anomaly ? 'gp-green' : this.ml.severity === 'high' ? 'gp-red' : 'gp-amber' },
    busy() { return !!this.data.running },
    chartRows() { return (this.sensor.readings || []).map(r => r[this.trend]) },
    chartValues() { return this.chartRows.filter(v => v != null && Number.isFinite(v)) },
    chartMin() { return this.chartValues.length ? Math.min(...this.chartValues) : 0 },
    chartMax() { return this.chartValues.length ? Math.max(...this.chartValues) : 1 },
    chartPoints() { const span = this.chartMax - this.chartMin; return this.chartRows.map((v,i) => v == null ? null : ({x:this.chartRows.length === 1 ? 300 : i / (this.chartRows.length - 1) * 600, y:span === 0 ? 70 : 130 - (v - this.chartMin) / span * 120})).filter(Boolean) },
    chartLine() { if (this.chartRows.some(v => v == null)) return ''; return this.chartPoints.map(p => p.x + ',' + p.y).join(' ') }
  },
  watch: { msg: { immediate: true, handler(msg) {
    if (msg?.topic === 'status' && msg.payload?.config) { this.data = msg.payload; this.connected = true; this.notice = '' }
    if (msg?.topic === 'request-error') { this.notice = msg.payload?.error || 'Request failed'; if(msg.payload?.disconnected) this.connected = false }
    if (msg?.topic === 'accepted') this.notice = 'Prediction refresh started. Results will appear automatically.'
  } } },
  methods: {
    run(mode) { this.send({topic: mode}); this.notice = 'Requesting analysis…' },
    sensorValue(v) { return v == null ? '—' : Number(v).toFixed(1) },
    number(v) { return v == null ? '—' : Number(v).toFixed(4) },
    date(v) { if(!v) return '—'; return new Date(v).toLocaleString('en-GB', {timeZone:this.config.timezone || 'Asia/Colombo',month:'short',day:'numeric',hour:'2-digit',minute:'2-digit',second:'2-digit'}) }
  }
}
</script>
<style>
.gp-tabs{display:flex;gap:8px;margin:0 0 22px;border-bottom:1px solid #dbe6df;padding-bottom:10px}.gp-tabs a{padding:9px 16px;color:#61766c;text-decoration:none;border-radius:7px;font-size:13px;font-weight:600}.gp-tabs .gp-tab-active{background:#217454;color:white}

.gp-shell{--gp-green:#217454;--gp-ink:#183b30;--gp-muted:#61766c;--gp-border:#dbe6df;max-width:1450px;margin:auto;padding:24px 16px 32px;color:var(--gp-ink);font-family:Inter,system-ui,sans-serif;background:#f6f8f5}.gp-heading{display:flex;justify-content:space-between;align-items:flex-start;gap:20px;margin-bottom:24px}.gp-eyebrow{font-size:11px;font-weight:700;letter-spacing:2px;color:var(--gp-green)}.gp-heading h1{font-size:clamp(25px,3vw,38px);font-weight:650;letter-spacing:-1px;margin:8px 0}.gp-heading p,.gp-subtle,.gp-meta{font-size:13px;color:var(--gp-muted)}.gp-pill{display:inline-flex;align-items:center;align-self:flex-start;flex:0 0 auto;line-height:1.4;padding:6px 10px;border-radius:20px;font-size:11px;white-space:nowrap;font-weight:600}.gp-green{background:#e0f0e6;color:#216146}.gp-amber{background:#fff0d3;color:#825812}.gp-toolbar{display:flex;justify-content:space-between;gap:16px;align-items:center;padding:18px 20px;background:#edf3ee;border:1px solid var(--gp-border);border-radius:12px;margin-bottom:20px}.gp-toolbar strong,.gp-toolbar span{display:block}.gp-toolbar strong{font-size:14px}.gp-toolbar span{font-size:12px;color:var(--gp-muted);margin-top:4px}.gp-buttons{display:flex;gap:10px}.gp-buttons button{padding:10px 16px;border:1px solid #b7c9bf;border-radius:7px;background:white;color:var(--gp-ink);font-weight:600;font-size:12px}.gp-buttons .gp-primary{background:var(--gp-green);border-color:var(--gp-green);color:white}.gp-buttons button:disabled{opacity:.5;cursor:wait}.gp-buttons button:focus-visible,select:focus-visible{outline:3px solid #81b89a;outline-offset:2px}.gp-alert{padding:12px 16px;background:#fff6e7;border:1px solid #efd6a4;border-radius:8px;color:#72521f;font-size:13px;margin:12px 0}.gp-alert ul{margin:6px 0 0;padding-left:20px}.gp-grid{display:grid;gap:16px}.gp-metrics{grid-template-columns:repeat(6,minmax(0,1fr))}.gp-card{background:#fff;border:1px solid var(--gp-border);border-radius:12px;padding:20px}.gp-label{font-size:12px;color:var(--gp-muted)}.gp-value{font-size:32px;font-weight:650;letter-spacing:-1px;margin:12px 0}.gp-value small{font-size:16px;font-weight:400;margin-left:4px}.gp-card .gp-subtle{font-size:12px;line-height:1.6}.gp-meta{margin:14px 2px 22px}.gp-two{grid-template-columns:1fr 1fr;margin-bottom:16px}.gp-card h2{font-size:17px;font-weight:650;margin:0}.gp-card-heading{display:flex;justify-content:space-between;gap:12px;align-items:center;margin-bottom:18px}.gp-result{font-size:26px;font-weight:600;letter-spacing:-.5px;margin:20px 0 10px}.gp-prediction p{font-size:13px;line-height:1.7}.gp-score{display:flex;gap:30px;margin:22px 0}.gp-score span{display:block;font-size:9px;letter-spacing:1px;color:var(--gp-muted)}.gp-score strong{display:block;font-size:22px;margin-top:7px}.gp-meter{position:relative;height:8px;background:#edf3ee;border-radius:4px;overflow:hidden;margin-bottom:15px}.gp-meter div{height:100%;background:var(--gp-green)}.gp-meter span{position:absolute;left:50%;height:100%;top:0;border-left:2px solid #cb963e}.gp-card select{font-size:11px;border:1px solid var(--gp-border);border-radius:6px;padding:6px;max-width:190px;background:white;color:var(--gp-ink)}.gp-chart{height:160px;display:flex;gap:8px;margin-top:20px}.gp-chart svg{width:100%;height:140px;overflow:visible}.gp-chart-scale{display:flex;flex-direction:column;justify-content:space-between;height:140px;font-size:10px;color:var(--gp-muted);min-width:28px}.gp-chart-dates{display:flex;justify-content:space-between;font-size:10px;color:var(--gp-muted);margin:8px 0 15px}.gp-empty{color:var(--gp-muted);height:140px;display:flex;align-items:center;justify-content:center;font-size:13px}.gp-agent{display:flex;gap:12px;align-items:center;padding:12px 0;border-bottom:1px solid #edf1ee}.gp-agent strong{font-size:13px}.gp-agent p{font-size:11px;color:var(--gp-muted);margin:2px 0}.gp-agent-status{margin-left:auto;font-size:11px;color:var(--gp-muted)}.gp-dot{height:9px;width:9px;border-radius:50%;flex-shrink:0}.gp-dot-ok{background:#398966}.gp-dot-pending{background:#cda255}.gp-context{background:#f5f8f5;padding:12px;border-radius:8px;margin-top:16px;font-size:12px}.gp-context p{margin:5px 0;color:var(--gp-muted);line-height:1.5}.gp-action{display:flex;gap:12px;padding:12px 0;border-bottom:1px solid #edf1ee}.gp-action-number{background:#edf4ef;color:var(--gp-green);width:26px;height:26px;display:flex;align-items:center;justify-content:center;border-radius:6px;flex-shrink:0;font-size:12px}.gp-priority{font-size:10px;text-transform:uppercase;letter-spacing:1px;color:var(--gp-muted)}.gp-action p{font-size:13px;line-height:1.6;margin:4px 0}.gp-evidence{font-size:12px;line-height:1.6;margin:16px 0}.gp-evidence p{margin:4px 0}.gp-evidence small{color:var(--gp-muted)}.gp-card ul{font-size:12px;padding-left:20px;line-height:1.7}.gp-weather{display:flex;justify-content:space-between;align-items:center;gap:15px;margin-bottom:16px}.gp-weather p{font-size:12px;color:var(--gp-muted);margin:8px 0 0}.gp-weather-values{display:flex;flex-direction:column;gap:4px;font-size:12px}.gp-weather-values strong{font-size:22px}.gp-weather-values small{color:var(--gp-muted)}.gp-table-scroll{overflow:auto}.gp-history table{width:100%;border-collapse:collapse;font-size:12px;margin-top:15px}.gp-history th{text-align:left;color:var(--gp-muted);font-weight:500}.gp-history td,.gp-history th{padding:10px 12px;border-bottom:1px solid #edf1ee}.gp-mono{font-family:monospace}.gp-footer{display:flex;justify-content:space-between;margin-top:24px;font-size:10px;letter-spacing:1px;color:var(--gp-muted)}.gp-footer span{letter-spacing:0}@media(max-width:1100px){.gp-metrics{grid-template-columns:repeat(3,1fr)}}@media(max-width:700px){.gp-metrics{grid-template-columns:repeat(2,1fr)}.gp-two{grid-template-columns:1fr}.gp-heading,.gp-toolbar,.gp-weather{align-items:flex-start;flex-direction:column}.gp-buttons{width:100%}.gp-buttons button{flex:1;padding:10px}.gp-card{padding:16px}.gp-footer{gap:10px;flex-direction:column}.gp-heading h1{font-size:28px}}

.gp-red{background:#fde8e7;color:#b42318}.gp-red-text{color:#b42318}.gp-amber-text{color:#825812}.gp-green-text{color:#216146}.gp-red-meter div{background:#c43226}.gp-amber-meter div{background:#b77912}.gp-green-meter div{background:#217454}.gp-score{flex-wrap:wrap;gap:18px 28px}
</style>
