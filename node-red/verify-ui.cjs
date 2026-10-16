// Compile and server-render the actual template in normal and blocked states.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const { parse } = require('@vue/compiler-sfc');
const { createSSRApp } = require('vue');
const { renderToString } = require('@vue/server-renderer');

async function main() {
    const source = fs.readFileSync(__dirname + '/dashboard.vue', 'utf8');
    const { descriptor, errors } = parse(source);
    assert.deepEqual(errors, []);
    const options = new Function(descriptor.script.content.replace('export default', 'return'))();
    const payload = {
        config: {demo: true, model: 'autoencoder', model_ready: true, timezone: 'Asia/Colombo'},
        prediction: {
            sensor_report: {status:'ok', source:'example', latest:{timestamp:'2026-10-12T00:00:00Z',air_temp_c:25},
                readings:[{timestamp:'2026-10-12T00:00:00Z',air_temp_c:25}],row_count:1},
            ml_report:{status:'ok',score:0.1,threshold:0.2,is_anomaly:false,severity:'normal'},
            decision_report:{status:'ready_for_review',actions:[]}
        }, history:[]
    };
    async function render() {
        // FlowFuse supplies these computed properties before merging custom options.
        // A method named `value` collides with its payload accessor and breaks rendering.
        const reserved = ['value', 'msg', 'messages', 'setup'];
        for (const name of reserved) {
            assert.equal(Object.hasOwn(options.methods || {}, name), false, 'FlowFuse reserves ' + name);
        }
        return renderToString(createSSRApp({...options,props:['msg'],
            computed: { value() { return this.msg.payload; }, ...options.computed },
            template:descriptor.template.content},
            {msg:{topic:'status',payload}}));
    }
    const complete = await render();
    assert.match(complete, /Normal sensor pattern/);
    assert.match(complete, /DEMO MODE/);
    assert.match(complete, /Backend connected/);
    assert.match(complete, /ACTIVE THRESHOLD/);
    payload.prediction.ml_report = {status:'ok',score:55.9746,threshold:0.1786,is_anomaly:true,severity:'high'};
    const anomaly = await render();
    assert.match(anomaly, /Unusual sensor pattern/);
    assert.match(anomaly, /gp-red-text/);
    assert.match(anomaly, /gp-red-meter/);
    assert.match(anomaly, /high severity/);
    assert.match(complete, /gp-green-meter/);
    payload.prediction.ml_report = {status:'ok',score:0.22,threshold:0.1786,is_anomaly:true,severity:'moderate'};
    const moderate = await render();
    assert.match(moderate, /gp-amber-meter/);
    assert.match(moderate, /moderate severity/);
    assert.doesNotMatch(moderate, /gp-red-meter/);
    payload.prediction.ml_report = {status:'ok',score:0.606566,threshold:0.7,trained_threshold:0.178632,
        threshold_source:'override',is_anomaly:false,severity:'normal'};
    const custom = await render();
    assert.match(custom, /Normal sensor pattern/);
    assert.match(custom, /0\.7000/);
    assert.doesNotMatch(custom, /high severity/);
    payload.prediction.ml_report = {status:'blocked',error:'Missing measured soil temperatures'};
    const blocked = await render();
    assert.match(blocked, /Prediction unavailable/);
    assert.match(blocked, /Missing measured soil temperatures/);
    assert.doesNotMatch(blocked, /Normal sensor pattern/);
    assert.doesNotMatch(blocked, /ANOMALY SCORE/);
    assert.doesNotMatch(blocked, /Run AI assessment/);
    assert.match(blocked, /href="\/dashboard\/assistant"/);
    const assistantSource=fs.readFileSync(__dirname+'/assistant.vue','utf8');
    const assistant=parse(assistantSource).descriptor;
    const assistantOptions=new Function(assistant.script.content.replace('export default','return'))();
    for(const reserved of ['value','msg','messages','setup']) {
        assert.equal(Object.hasOwn(assistantOptions.methods || {},reserved),false);
    }
    async function renderAssistant() {
        return renderToString(createSSRApp({...assistantOptions,props:['msg'],
            computed:{value(){return this.msg.payload},...assistantOptions.computed},template:assistant.template.content},
            {msg:{topic:'status',payload}}));
    }
    const html=await renderAssistant();
    assert.match(html,/Your plant conversation/);
    assert.match(html,/General plant knowledge/);
    assert.match(html,/Greenhouse insights/);
    assert.match(html,/Greenhouse context/);
    assert.match(html,/Needs complete readings/);
    assert.match(html,/1 of 6 measurements available/);
    assert.match(html,/Missing measurements/);
    assert.match(html,/Ask assistant/);
    assert.doesNotMatch(html,/ANOMALY SCORE/);
    payload.prediction.sensor_report={status:'ok',warnings:[],latest:{timestamp:'2026-10-15T15:10:40Z',
        air_temp_c:27.3,humidity:87.2,soil_moisture_l1:50.21,soil_temp_l1_c:27.06,
        soil_moisture_l2:42.57,soil_temp_l2_c:27.37}};
    payload.prediction.ml_report={status:'ok',is_anomaly:false};
    const ready=await renderAssistant();
    assert.match(ready,/Ready to include/);
    assert.match(ready,/6 of 6 measurements available/);
    assert.match(ready,/Normal sensor pattern/);
    assert.doesNotMatch(ready,/Missing measurements/);
    payload.prediction.sensor_report.warnings=['Sensor readings are stale'];
    const stale=await renderAssistant();
    assert.match(stale,/Review measurements/);
    assert.doesNotMatch(stale,/Ready to include/);
    payload.prediction={};
    const waiting=await renderAssistant();
    assert.match(waiting,/Waiting for readings/);
    assert.match(waiting,/0 of 6 measurements available/);
    console.log('PASS: Separate prediction and assistant tabs render with FlowFuse reserved properties');
}
main().catch(error => { console.error(error); process.exitCode = 1; });
