<template>
  <main class="gp-shell gp-chat-shell">
    <header class="gp-heading"><div><div class="gp-eyebrow">GREENPULSE / PLANT ASSISTANT</div><h1>Ask. Understand. Grow.</h1><p>Plant questions and advice about your stored readings, whenever you need it.</p></div><span class="gp-pill" :class="connected ? 'gp-green' : 'gp-amber'">{{ connected ? 'Backend connected' : 'Waiting for backend' }}</span></header>
    <nav class="gp-tabs" aria-label="Dashboard tabs"><a href="/dashboard/greenhouse">Prediction</a><a class="gp-tab-active" href="/dashboard/assistant" aria-current="page">AI assistant</a></nav>
    <div v-if="data.config?.demo" class="gp-alert"><strong>DEMO MODE</strong> — questions are received, but no AI service is called.</div>
    <div v-if="connected && !data.config?.groq_ready && !data.config?.demo" class="gp-alert">Add GROQ_API_KEY in .env to enable AI answers.</div>
    <div v-if="notice" class="gp-alert" role="status">{{ notice }}</div>
    <section class="gp-chat-layout">
      <article class="gp-card gp-conversation">
        <div class="gp-card-heading"><h2>Your plant conversation</h2><span class="gp-pill gp-green">Answers on request</span></div>
        <div v-if="!conversation.length" class="gp-chat-empty"><div class="gp-chat-leaf">🌱</div><h2>What would you like to know?</h2><p>Ask about plant care, greenhouse conditions, or what to check in your sensor readings.</p><div class="gp-suggestions"><button @click="suggest('Why do plant leaves turn yellow?', 'general')" :disabled="pending">Why do leaves turn yellow?</button><button @click="suggest('How can I improve greenhouse ventilation?', 'general')" :disabled="pending">Improve greenhouse ventilation</button><button @click="suggest('Explain the latest model result and suggest checks for any anomalies or missing inputs.', 'greenhouse')" :disabled="pending">Explain my latest readings</button></div></div>
        <div class="gp-messages" role="log" aria-live="polite"><article v-for="turn in conversation" :key="turn.id" class="gp-chat-message" :class="turn.role === 'user' ? 'gp-message-user' : 'gp-message-assistant'"><div class="gp-message-label">{{ turn.role === 'user' ? 'You' : 'GreenPulse assistant' }}<span v-if="turn.scope === 'greenhouse'"> · Stored readings</span></div><div v-if="turn.role === 'assistant'" class="gp-answer" v-html="formatAnswer(turn.content)"></div><p v-else>{{ turn.content }}</p><small v-if="turn.evidence_timestamp">Evidence recorded: {{ date(turn.evidence_timestamp) }}</small></article><div v-if="pending" class="gp-thinking" role="status">Thinking about your question… Prediction monitoring continues independently.</div></div>
        <form class="gp-chat-form" @submit.prevent="ask">
          <fieldset class="gp-chat-mode"><legend>Use</legend><label><input type="radio" v-model="scope" value="general" :disabled="pending"> General plant knowledge</label><label><input type="radio" v-model="scope" value="greenhouse" :disabled="pending"> Greenhouse insights</label></fieldset>
          <label class="gp-question-label" for="gp-plant-question">Your question</label><textarea id="gp-plant-question" v-model="question" maxlength="4000" rows="3" placeholder="Ask a plant question, or describe what you would like to understand…" :disabled="pending" @keydown.ctrl.enter.prevent="ask"></textarea>
          <div class="gp-chat-submit"><button class="gp-chat-send" type="submit" :disabled="pending || !connected || !question.trim()">{{ pending ? 'Answering…' : 'Ask assistant' }}</button></div>
        </form>
      </article>
      <aside class="gp-card gp-chat-context" aria-label="Greenhouse context">
        <div class="gp-context-heading"><h2>Greenhouse context</h2><span class="gp-pill" :class="contextReady ? 'gp-green' : 'gp-amber'">{{ contextStatus }}</span></div>
        <p>Choose <strong>Greenhouse insights</strong> to include these readings and the prediction with your question.</p>
        <dl>
          <div><dt>Prediction</dt><dd>{{ modelSummary }}</dd></div>
          <div><dt>Last reading</dt><dd>{{ date(contextSensor.latest?.timestamp) }}</dd></div>
          <div><dt>Crop</dt><dd>{{ data.config?.crop && data.config.crop !== 'unspecified' ? data.config.crop : 'Not set' }}</dd></div>
          <div><dt>Sensor coverage</dt><dd>{{ availableInputCount }} of 6 measurements available</dd></div>
        </dl>
        <div v-if="missingInputLabels.length" class="gp-context-missing"><strong>Missing measurements</strong><p>{{ missingInputLabels.join(', ') }}</p></div>
      </aside>
    </section>
    <footer class="gp-footer">GREENPULSE AI <span>Plant advice for grower review</span></footer>
  </main>
</template>
<script>
export default {
  data() { return {data:{}, connected:false, notice:'', question:'', scope:'general', conversation:[], pending:false, pendingId:null, pendingScope:'general',
    sensorLabels:{air_temp_c:'Air temperature',humidity:'Air humidity',soil_moisture_l1:'Moisture · layer 1',soil_temp_l1_c:'Soil temperature · layer 1',soil_moisture_l2:'Moisture · layer 2',soil_temp_l2_c:'Soil temperature · layer 2'}
  } },
  computed: {
    contextSensor() { return this.data.prediction?.sensor_report || {} },
    contextModel() { return this.data.prediction?.ml_report || {} },
    availableInputCount() { return Object.keys(this.sensorLabels).filter(key => Number.isFinite(this.contextSensor.latest?.[key])).length },
    missingInputLabels() { return this.contextSensor.latest?.timestamp ? Object.keys(this.sensorLabels).filter(key => !Number.isFinite(this.contextSensor.latest[key])).map(key => this.sensorLabels[key]) : [] },
    contextReady() { return this.connected && this.contextSensor.status === 'ok' && this.availableInputCount === 6 && this.contextModel.status === 'ok' && !this.contextSensor.warnings?.length },
    contextStatus() {
      if(!this.connected) return 'Waiting for connection';
      if(!this.contextSensor.latest?.timestamp) return 'Waiting for readings';
      return this.contextReady ? 'Ready to include' : 'Review measurements';
    },
    modelSummary() {
      if(this.contextModel.status === 'ok') return this.contextModel.is_anomaly ? 'Unusual sensor pattern' : 'Normal sensor pattern';
      if(this.contextModel.status === 'blocked') return 'Needs complete readings';
      if(this.contextModel.status === 'error') return 'Prediction unavailable';
      return 'Waiting for prediction';
    }
  },
  watch: { msg: {immediate:true, handler(msg) {
    if (msg?.topic === 'status' && msg.payload?.config) { this.data=msg.payload; this.connected=true }
    if (msg?.topic === 'chat-reply' && msg.requestId === this.pendingId) {
      const reply=msg.payload || {}; this.pending=false; this.pendingId=null;
      if(reply.answer) { this.conversation.push({id:Date.now() + '-answer',role:'assistant',content:reply.answer,scope:reply.scope,evidence_timestamp:reply.evidence_timestamp}); this.notice='' }
      else this.notice=reply.error || 'The assistant could not answer. Please retry.';
    }
    if (msg?.topic === 'request-error' && (!msg.requestId || msg.requestId === this.pendingId)) { this.notice=msg.payload?.error || 'Request failed'; this.pending=false; this.pendingId=null; if(msg.payload?.disconnected) this.connected=false }
  }}},
  methods: {
    formatAnswer(text) {
      // Escape all model-provided HTML before adding our own formatting tags.
      const escape = value => value.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;').replace(/'/g,'&#39;');
      const inline = value => escape(value).replace(/\*\*(.+?)\*\*/g,'<strong>$1</strong>').replace(/`([^`]+)`/g,'<code>$1</code>').replace(/\*([^*\n]+)\*/g,'<em>$1</em>');
      const lines = String(text).replace(/\r\n/g,'\n').split('\n');
      let html='', paragraph=[], list='';
      const flush = () => { if(paragraph.length) { html+='<p>'+inline(paragraph.join(' '))+'</p>'; paragraph=[] } };
      const closeList = () => { if(list) { html+='</'+list+'>'; list='' } };
      for(const line of lines) {
        const heading=line.match(/^#{1,6}\s+(.+)$/);
        const bullet=line.match(/^\s*(?:[-*+]\s+|\d+[.)]\s+)(.+)$/);
        if(bullet) {
          flush(); const kind=/^\s*\d/.test(line) ? 'ol' : 'ul';
          if(list!==kind) { closeList(); html+='<'+kind+'>'; list=kind }
          html+='<li>'+inline(bullet[1])+'</li>';
        } else {
          closeList();
          if(!line.trim()) flush();
          else if(heading) { flush(); html+='<h3>'+inline(heading[1])+'</h3>' }
          else if(/^\s*([-*_])\1{2,}\s*$/.test(line)) { flush(); html+='<hr>' }
          else paragraph.push(line.trim());
        }
      }
      flush(); closeList(); return html;
    },
    suggest(text,scope) { this.question=text; this.scope=scope },
    ask() {
      if(this.pending || !this.connected || !this.question.trim()) return;
      const text=this.question.trim(); const history=this.conversation.slice(-10).map(t=>({role:t.role,content:t.content.slice(0,12000)}));
      this.pendingId=globalThis.crypto.randomUUID(); this.pending=true; this.pendingScope=this.scope; this.notice='';
      this.conversation.push({id:this.pendingId,role:'user',content:text,scope:this.scope}); this.question='';
      this.send({topic:'chat',requestId:this.pendingId,payload:{message:text,scope:this.scope,history}});
    },
    date(value) { return value ? new Date(value).toLocaleString('en-GB',{timeZone:this.data.config?.timezone || 'Asia/Colombo',month:'short',day:'numeric',hour:'2-digit',minute:'2-digit'}) : '—' }
  }
}
</script>
<style>
.gp-chat-shell{--gp-green:#217454;--gp-ink:#183b30;--gp-muted:#61766c;--gp-border:#dbe6df;max-width:1450px;margin:auto;padding:24px 16px 32px;color:var(--gp-ink);font-family:Inter,system-ui,sans-serif;background:#f6f8f5}.gp-chat-shell .gp-heading{display:flex;justify-content:space-between;align-items:flex-start;gap:20px;margin-bottom:24px}.gp-chat-shell .gp-eyebrow{font-size:11px;font-weight:700;letter-spacing:2px;color:#217454}.gp-chat-shell h1{font-size:clamp(25px,3vw,38px);font-weight:650;letter-spacing:-1px;margin:8px 0}.gp-chat-shell .gp-heading p{font-size:13px;color:#61766c}.gp-chat-shell .gp-tabs{display:flex;gap:8px;margin:0 0 22px;border-bottom:1px solid #dbe6df;padding-bottom:10px}.gp-chat-shell .gp-tabs a{padding:9px 16px;color:#61766c;text-decoration:none;border-radius:7px;font-size:13px;font-weight:600}.gp-chat-shell .gp-tabs .gp-tab-active{background:#217454;color:white}.gp-chat-shell .gp-card{background:white;border:1px solid #dbe6df;border-radius:12px;padding:24px}.gp-chat-shell .gp-card h2{font-size:18px;font-weight:650;margin:0}.gp-chat-shell .gp-card-heading{display:flex;justify-content:space-between;align-items:center;gap:12px;margin-bottom:20px}.gp-chat-shell .gp-pill{display:inline-flex;align-items:center;align-self:flex-start;flex:0 0 auto;line-height:1.4;padding:6px 10px;border-radius:20px;font-size:11px;white-space:nowrap;font-weight:600}.gp-chat-shell .gp-green{background:#e0f0e6;color:#216146}.gp-chat-shell .gp-amber{background:#fff0d3;color:#825812}.gp-chat-shell .gp-alert{padding:12px 16px;background:#fff6e7;border:1px solid #efd6a4;border-radius:8px;color:#72521f;font-size:13px;margin:12px 0}.gp-chat-layout{display:grid;grid-template-columns:minmax(0,3fr) minmax(260px,1fr);gap:20px;align-items:start}.gp-conversation{min-height:550px}.gp-chat-empty{text-align:center;padding:32px 20px 40px}.gp-chat-leaf{font-size:38px;margin-bottom:15px}.gp-chat-empty p{font-size:13px;color:#61766c;max-width:450px;margin:12px auto 22px;line-height:1.7}.gp-suggestions{display:flex;flex-wrap:wrap;gap:8px;justify-content:center}.gp-suggestions button{padding:9px 12px;border:1px solid #dbe6df;border-radius:8px;font-size:12px;background:#f5f8f5;color:#217454}.gp-messages{display:flex;flex-direction:column;gap:16px;margin:24px 0}.gp-chat-message{padding:16px 18px;border-radius:10px;max-width:94%;border:1px solid #e3eae5}.gp-message-user{align-self:flex-end;background:#eaf4ed}.gp-message-assistant{align-self:flex-start;background:#f9fbf8}.gp-message-label{font-size:11px;font-weight:700;color:#217454;margin-bottom:7px}.gp-message-label span{font-weight:400;color:#61766c}.gp-chat-message p{font-size:14px;line-height:1.8;white-space:pre-wrap;overflow-wrap:anywhere;margin:0}.gp-chat-message small{display:block;font-size:10px;color:#61766c;margin-top:10px}.gp-thinking{font-size:12px;color:#61766c;padding:14px}.gp-chat-form{border-top:1px solid #e3eae5;padding-top:20px}.gp-chat-mode{display:flex;flex-wrap:wrap;gap:16px;border:0;padding:0;margin:0 0 16px}.gp-chat-mode legend{font-size:11px;color:#61766c;margin-bottom:8px}.gp-chat-mode label{font-size:12px;display:flex;gap:6px;align-items:center}.gp-chat-mode input{accent-color:#217454}.gp-question-label{display:block;font-size:12px;font-weight:600;margin-bottom:8px}.gp-chat-form textarea{width:100%;box-sizing:border-box;padding:14px;border:1px solid #bbd0c2;border-radius:9px;resize:vertical;background:white;font:inherit;font-size:13px;color:#183b30}.gp-chat-form textarea:focus-visible{outline:2px solid #81b89a;outline-offset:2px}.gp-chat-submit{display:flex;justify-content:flex-end;gap:12px;align-items:center;margin-top:12px}.gp-chat-send{background:#217454;color:white;border:0;border-radius:7px;padding:11px 18px;font-size:12px;font-weight:600}.gp-chat-send:disabled{opacity:.5}.gp-chat-context p{font-size:12px;line-height:1.8;color:#61766c;margin:15px 0}.gp-chat-context dl{font-size:12px;margin:20px 0}.gp-chat-context dt{color:#61766c;margin-top:16px}.gp-chat-context dd{margin:4px 0 0;font-weight:600;overflow-wrap:anywhere}.gp-chat-shell .gp-footer{display:flex;justify-content:space-between;gap:16px;margin-top:24px;font-size:10px;letter-spacing:1px;color:#61766c}.gp-chat-shell .gp-footer span{letter-spacing:0}@media(max-width:850px){.gp-chat-layout{grid-template-columns:1fr}.gp-chat-context{order:2}.gp-chat-shell .gp-heading{flex-direction:column}.gp-chat-shell .gp-card{padding:18px}.gp-chat-submit{align-items:flex-end}.gp-chat-shell .gp-footer{flex-direction:column}}
 .gp-answer{font-size:14px;line-height:1.7;overflow-wrap:anywhere}.gp-chat-message .gp-answer p{white-space:normal;margin:0 0 12px}.gp-answer p:last-child{margin-bottom:0}.gp-answer h3{font-size:15px;margin:16px 0 8px;font-weight:650}.gp-answer h3:first-child{margin-top:0}.gp-answer ul,.gp-answer ol{padding-left:22px;margin:8px 0 14px}.gp-answer li{margin:6px 0}.gp-answer code{background:#eaf0e9;padding:2px 5px;border-radius:4px}.gp-answer hr{border:0;border-top:1px solid #dbe6df;margin:16px 0}

.gp-chat-shell .gp-chat-context{min-width:0;box-sizing:border-box}
.gp-context-heading{display:flex;flex-direction:column;align-items:flex-start;gap:12px}
.gp-chat-context dl{display:grid;gap:0;margin:20px 0 0}
.gp-chat-context dl>div{padding:14px 0;border-bottom:1px solid #e7eee9}
.gp-chat-context dl>div:last-child{border-bottom:0;padding-bottom:0}
.gp-chat-context dt{margin:0;font-size:11px}
.gp-chat-context dd{margin:6px 0 0;font-size:13px;line-height:1.5}
.gp-context-missing{margin-top:20px;padding:12px;background:#fff6e7;border-radius:8px;color:#72521f;font-size:11px}
.gp-context-missing p{margin:6px 0 0;color:inherit;line-height:1.6}
</style>
