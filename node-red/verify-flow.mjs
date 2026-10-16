// End-to-end verification against an explicit demo API: no external service calls.
import assert from 'node:assert/strict';
import { io } from 'socket.io-client';
const api = process.env.GREENPULSE_API_URL || 'http://127.0.0.1:8010';
const dashboard = process.env.GREENPULSE_DASHBOARD_URL || 'http://127.0.0.1:1880';
const status = await fetch(api + '/api/status').then(r => r.json());
assert.equal(status.config.demo, true, 'This test only runs in explicit demo mode');
for(const path of ['/dashboard/greenhouse','/dashboard/assistant']) {
    assert.equal((await fetch(dashboard + path)).status, 200);
}
const socket = io(dashboard, { path: '/dashboard/socket.io', transports: ['polling', 'websocket'],
    reconnection: false, autoConnect: false });
const requestId = crypto.randomUUID();
let sent = false, predictionId, prediction, reply;
const received = new Promise((resolve, reject) => {
    const timeout = setTimeout(() => reject(new Error('Prediction/chat flow did not complete')), 20000);
    function finish() {
        if(reply && prediction) { clearTimeout(timeout); resolve({reply,prediction}); }
    }
    socket.on('connect_error', error => { clearTimeout(timeout); reject(error); });
    socket.on('ui-config', (_id, config) => {
        assert(Object.values(config.pages).some(p => p.path === '/assistant'));
        if (!sent) {
            sent = true;
            socket.emit('widget-send', 'gp-dashboard', {topic:'prediction'});
            socket.emit('widget-send', 'gp-assistant', {topic:'chat',requestId,
                payload:{message:'Why do plant leaves turn yellow?',scope:'general',history:[]}});
        }
    });
    socket.on('msg-input:gp-dashboard', msg => {
        if(msg.topic === 'accepted') predictionId = msg.payload.id;
        if(msg.topic === 'status' && msg.payload?.prediction?.id === predictionId) {
            prediction = msg.payload.prediction;
            finish();
        }
    });
    socket.on('msg-input:gp-assistant', msg => {
        if(msg.topic === 'chat-reply' && msg.requestId === requestId) {
            reply = msg.payload;
            finish();
        }
    });
});
socket.connect();
try {
    const result = await received;
    assert.equal(result.reply.status, 'demo');
    assert.match(result.reply.answer, /Demo mode/);
    assert.equal(result.prediction.sensor_report.source, 'example');
    assert.equal(result.prediction.ml_report.status, 'ok');
    assert(!result.prediction.history.includes('master_agent'));
    console.log('PASS: Two separate tabs, continuous model prediction, and user-requested chat route');
} finally { socket.close(); }
