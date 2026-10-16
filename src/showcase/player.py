"""Render the synchronized video, event timeline, and pitch map as one component."""
from __future__ import annotations

import base64
import json
from html import escape
from pathlib import Path
from typing import Any, Iterable

from .data import ClipWindow


def video_data_uri(video_path: Path) -> str:
    """Return a browser-playable data URI for a local MP4."""
    encoded = base64.b64encode(video_path.read_bytes()).decode("ascii")
    return f"data:video/mp4;base64,{encoded}"


def build_player_html(
    *,
    video_uri: str,
    events: Iterable[dict[str, Any]],
    clip: ClipWindow,
    bundle_name: str,
    radar_uri: str | None = None,
) -> str:
    """Build an isolated, dependency-free player that synchronizes on currentTime."""
    payload = {
        "events": list(events),
        "clip": {
            "start": clip.start_s,
            "end": clip.end_s,
            "focusIds": list(clip.focus_event_ids),
            "label": clip.label,
        },
        "hasRadar": bool(radar_uri),
    }
    # Escaping the closing tag keeps local event labels from terminating the script.
    payload_json = json.dumps(payload, ensure_ascii=True, separators=(",", ":")).replace("</", "<\\/")
    safe_title = escape(bundle_name)
    safe_video_uri = escape(video_uri, quote=True)
    if radar_uri:
        safe_radar_uri = escape(radar_uri, quote=True)
        radar_markup = (
            f'<video id="radar" muted playsinline preload="metadata" '
            f'src="{safe_radar_uri}" aria-label="Live player tracking map"></video>'
        )
    else:
        radar_markup = '<div class="tracking-empty">—</div>'
    pitch_markup = """
    <svg id="action-pitch" viewBox="0 0 1200 700" role="img" aria-label="Locations of passes, carries, and possession gains">
      <defs>
        <marker id="arrow-team-0" markerWidth="10" markerHeight="10" refX="7" refY="3" orient="auto" markerUnits="strokeWidth"><path d="M0,0 L0,6 L8,3 z" fill="#2896f0"/></marker>
        <marker id="arrow-team-1" markerWidth="10" markerHeight="10" refX="7" refY="3" orient="auto" markerUnits="strokeWidth"><path d="M0,0 L0,6 L8,3 z" fill="#eb3c30"/></marker>
        <marker id="arrow-team-unknown" markerWidth="10" markerHeight="10" refX="7" refY="3" orient="auto" markerUnits="strokeWidth"><path d="M0,0 L0,6 L8,3 z" fill="#8fa59b"/></marker>
        <marker id="arrow-carry" markerWidth="10" markerHeight="10" refX="7" refY="3" orient="auto" markerUnits="strokeWidth"><path d="M0,0 L0,6 L8,3 z" fill="#ff8f70"/></marker>
      </defs>
      <g fill="none" stroke="rgba(255,255,255,.72)" stroke-width="4">
        <rect x="18" y="18" width="1164" height="664" rx="7"/><path d="M600 18V682"/><circle cx="600" cy="350" r="91"/><circle cx="600" cy="350" r="5" fill="rgba(255,255,255,.72)"/>
        <path d="M18 178H170V522H18M1182 178H1030V522H1182M18 258H72V442H18M1182 258H1128V442H1182"/>
        <circle cx="130" cy="350" r="5" fill="rgba(255,255,255,.72)"/><circle cx="1070" cy="350" r="5" fill="rgba(255,255,255,.72)"/>
        <path d="M170 281A91 91 0 0 1 170 419M1030 281A91 91 0 0 0 1030 419"/>
      </g>
      <g id="pitch-events"></g>
    </svg>"""
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<style>
  :root {{
    --ink:#f4f7f5; --muted:#8fa59b; --panel:#051c14; --line:#123a2a;
    --pitch:#073b28; --lime:#73df93; --cyan:#55d9d2; --amber:#ffc857;
    --red:#f04d5e; --blue:#3296f0;
  }}
  * {{ box-sizing:border-box; }}
  html,body {{ margin:0; background:#000; color:var(--ink); font-family:Inter,ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; }}
  body {{ overflow:hidden; }}
  .shell {{ background:#020b08; border:1px solid #123a2a; border-radius:14px; overflow:hidden; }}
  .bar {{ height:50px; padding:0 16px; display:flex; align-items:center; justify-content:space-between; border-bottom:1px solid var(--line); background:#03130d; }}
  .bar-left,.bar-right {{ display:flex; align-items:center; gap:10px; min-width:0; }}
  .live {{ display:inline-flex; align-items:center; gap:7px; color:var(--lime); font-size:11px; font-weight:800; letter-spacing:.12em; }}
  .live:before {{ content:""; width:7px; height:7px; border-radius:50%; background:var(--lime); box-shadow:0 0 0 4px rgba(169,245,107,.12); }}
  .session {{ color:var(--muted); font-size:12px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; max-width:36vw; }}
  .clock {{ font-variant-numeric:tabular-nums; font-size:13px; font-weight:750; letter-spacing:.04em; }}
  .status {{ font-size:10px; text-transform:uppercase; letter-spacing:.08em; color:var(--muted); border:1px solid var(--line); border-radius:999px; padding:5px 8px; }}
  .main {{ display:grid; grid-template-columns:minmax(0,1.7fr) minmax(300px,.9fr); min-height:472px; }}
  .video-wrap {{ position:relative; background:#000; display:flex; align-items:center; justify-content:center; min-width:0; }}
  #video {{ display:block; width:100%; max-height:472px; aspect-ratio:16/9; background:#000; object-fit:contain; }}
  .clip-progress {{ position:absolute; height:3px; left:0; right:0; bottom:0; background:rgba(255,255,255,.12); pointer-events:none; }}
  .clip-progress span {{ display:block; height:100%; width:0; background:var(--lime); transition:width .08s linear; }}
  .side {{ border-left:1px solid var(--line); background:var(--panel); padding:14px; min-width:0; }}
  .section-head {{ display:flex; align-items:center; justify-content:space-between; margin-bottom:9px; }}
  .eyebrow {{ font-size:10px; color:var(--muted); font-weight:800; letter-spacing:.12em; text-transform:uppercase; }}
  .tracking-wrap,.pitch-wrap {{ position:relative; border:1px solid #18543b; background:var(--pitch); border-radius:10px; padding:0; overflow:hidden; }}
  #radar {{ width:100%; display:block; aspect-ratio:13/8; background:#04261a; object-fit:cover; }}
  .tracking-empty {{ min-height:185px; display:grid; place-items:center; color:var(--muted); }}
  .analytics {{ border-top:1px solid var(--line); background:#03130d; padding:13px 16px 14px; display:grid; grid-template-columns:minmax(360px,1.1fr) minmax(0,1.9fr); gap:12px; }}
  .map-card {{ min-width:0; border:1px solid var(--line); border-radius:9px; background:#020b08; padding:10px; }}
  .map-head {{ display:flex; align-items:center; justify-content:space-between; gap:12px; margin-bottom:8px; }}
  .map-legend {{ display:flex; align-items:center; gap:10px; color:var(--muted); font-size:9px; font-weight:800; letter-spacing:.08em; text-transform:uppercase; }}
  .map-legend span {{ display:inline-flex; align-items:center; gap:4px; }}
  .map-legend i {{ width:7px; height:7px; border-radius:50%; display:inline-block; }}
  #action-pitch {{ width:100%; display:block; aspect-ratio:12/7; background:var(--pitch); }}
  .latest {{ border-top:1px solid var(--line); margin-top:13px; padding-top:13px; min-height:118px; }}
  .event-row {{ display:flex; gap:10px; align-items:flex-start; }}
  .event-icon {{ width:35px; height:35px; flex:0 0 35px; border-radius:10px; display:grid; place-items:center; font-size:15px; color:#08100d; background:var(--lime); font-weight:900; }}
  .event-copy {{ min-width:0; flex:1; }}
  .event-name {{ font-size:15px; font-weight:800; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }}
  .event-detail {{ color:var(--muted); margin-top:3px; font-size:11px; line-height:1.45; }}
  .confidence {{ margin-top:9px; height:4px; background:#29332f; border-radius:99px; overflow:hidden; }}
  .confidence span {{ display:block; height:100%; background:var(--lime); border-radius:99px; }}
  .empty {{ color:var(--muted); font-size:12px; line-height:1.5; padding-top:4px; }}
  .charts {{ min-width:0; display:grid; grid-template-columns:repeat(3,1fr); gap:10px; }}
  .chart-card {{ min-width:0; border:1px solid var(--line); border-radius:9px; background:#020b08; padding:10px 11px 6px; }}
  .chart-head {{ display:flex; align-items:baseline; justify-content:space-between; }}
  .chart-head span {{ color:var(--muted); font-size:10px; font-weight:800; letter-spacing:.11em; text-transform:uppercase; }}
  .chart-head b {{ color:var(--ink); font-size:20px; font-variant-numeric:tabular-nums; }}
  .metric-chart {{ width:100%; height:104px; display:block; }}
  .pulse {{ animation:pulse 1.25s ease-out infinite; transform-box:fill-box; transform-origin:center; }}
  @keyframes pulse {{ 0%{{opacity:.9;transform:scale(.8)}} 100%{{opacity:0;transform:scale(2.4)}} }}
  @media (max-width:640px) {{
    body {{ overflow:auto; }} .main,.analytics {{ grid-template-columns:1fr; }} .side {{ border-left:0; border-top:1px solid var(--line); }}
    #video {{ max-height:none; }} .session {{ max-width:42vw; }} .charts {{ grid-template-columns:1fr; }}
  }}
</style>
</head>
<body>
<div class="shell">
  <div class="bar">
    <div class="bar-left"><span class="live">TRACKING LIVE</span><span class="session">{safe_title} · {escape(clip.label)}</span></div>
    <div class="bar-right"><span id="mode" class="status">Ready</span><span id="clock" class="clock">00:00.0</span></div>
  </div>
  <div class="main">
    <div class="video-wrap">
      <video id="video" controls playsinline preload="metadata" src="{safe_video_uri}"></video>
      <div class="clip-progress"><span id="progress"></span></div>
    </div>
    <aside class="side">
      <div class="section-head"><span class="eyebrow">Live player tracking</span><span class="eyebrow">120 × 70 m</span></div>
      <div class="tracking-wrap">{radar_markup}</div>
      <div class="latest"><div class="eyebrow" style="margin-bottom:10px">Now on the ball</div><div id="latest"></div></div>
    </aside>
  </div>
  <section class="analytics">
    <div class="map-card">
      <div class="map-head"><span class="eyebrow">Action locations</span><div class="map-legend"><span><i style="background:#2896f0"></i>Team 0</span><span><i style="background:#eb3c30"></i>Team 1</span><span><i style="background:#ff8f70"></i>Carry</span><span><i style="background:#c68cff"></i>Gain</span><b id="pitch-time">—</b></div></div>
      <div class="pitch-wrap">{pitch_markup}</div>
    </div>
    <div class="charts">
      <div class="chart-card"><div class="chart-head"><span>Passes</span><b id="pass-count">0</b></div><svg id="chart-pass" class="metric-chart" viewBox="0 0 360 104" preserveAspectRatio="none"></svg></div>
      <div class="chart-card"><div class="chart-head"><span>Carries</span><b id="carry-count">0</b></div><svg id="chart-carry" class="metric-chart" viewBox="0 0 360 104" preserveAspectRatio="none"></svg></div>
      <div class="chart-card"><div class="chart-head"><span>Possession gains</span><b id="gain-count">0</b></div><svg id="chart-gain" class="metric-chart" viewBox="0 0 360 104" preserveAspectRatio="none"></svg></div>
    </div>
  </section>
</div>
<script>
const state = {payload_json};
const video = document.getElementById('video');
const radar = document.getElementById('radar');
const NS = 'http://www.w3.org/2000/svg';
const colors = {{pass:'#55d9d2', zone_entry:'#a9f56b', possession:'#ffc857', possession_change:'#c68cff', carry:'#ff8f70'}};
const teamColors = {{0:'#2896f0', 1:'#eb3c30', unknown:'#8fa59b'}};
const focus = new Set(state.clip.focusIds);
const visibleEvents = state.events.filter(e => e.end >= state.clip.start && e.start <= state.clip.end);
let lastRender = -1;

function svg(tag, attrs={{}}) {{ const n=document.createElementNS(NS,tag); for(const [k,v] of Object.entries(attrs)) n.setAttribute(k,String(v)); return n; }}
function esc(value) {{ return String(value ?? '').replace(/[&<>"']/g, c => ({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}}[c])); }}
function clock(t) {{ t=Math.max(0,t||0); const m=Math.floor(t/60); const s=(t%60).toFixed(1).padStart(4,'0'); return String(m).padStart(2,'0')+':'+s; }}
function typeLabel(t) {{ return ({{pass:'Pass',zone_entry:'Entry',possession:'Possession',possession_change:'Possession change',carry:'Carry'}}[t]||t.replaceAll('_',' ')); }}
function actionIcon(t) {{ return ({{pass:'→',zone_entry:'↗',possession:'●',possession_change:'⇄',carry:'↝'}}[t]||'•'); }}
function isMomentActive(e,t) {{ return e.end-e.start < .12 ? Math.abs(t-e.start)<1.0 : t>=e.start && t<=e.end; }}
function eventDetail(e) {{
  const bits=[];
  if(e.team!==null && e.team!==undefined) bits.push('Team '+e.team);
  if(e.players?.length) bits.push(e.players.slice(0,2).map(p=>'#'+p).join(' → '));
  if(e.attributes?.zone) bits.push(e.attributes.zone.replaceAll('_',' '));
  if(e.length>0) bits.push(e.length.toFixed(1)+' m');
  bits.push(Math.round(e.confidence*100)+'% confidence');
  return bits.join(' · ');
}}
function priority(e) {{ return ({{possession_change:5,pass:4,zone_entry:3,carry:2,possession:1}}[e.type]||0); }}
function point(p) {{ if(!p) return null; return [Math.max(18,Math.min(1182,p[0]*10)),Math.max(18,Math.min(682,p[1]*10))]; }}
function teamKey(team) {{ return Number(team)===0?'0':Number(team)===1?'1':'unknown'; }}
function pitchColor(e) {{ return e.type==='pass'?teamColors[teamKey(e.team)]:(colors[e.type]||'#fff'); }}

function drawPitch(t) {{
  const layer=document.getElementById('pitch-events');
  if(!layer) {{ document.getElementById('pitch-time').textContent=clock(t); return; }}
  layer.replaceChildren();
  const occurred=visibleEvents.filter(e=>e.start<=t && ['pass','carry','possession_change'].includes(e.type) && (e.start_xy||e.end_xy));
  occurred.forEach((e,index)=>{{
    const a=point(e.start_xy), b=point(e.end_xy||e.start_xy); if(!a&&!b) return;
    const active=isMomentActive(e,t), highlighted=focus.size===0||focus.has(e.id);
    const opacity=active?1:(highlighted?.62:.25); const color=pitchColor(e);
    if(e.type==='possession_change') {{
      const p=b||a; const ring=svg('circle',{{cx:p[0],cy:p[1],r:active?28:18,fill:'none',stroke:color,'stroke-width':active?8:5,opacity}}); layer.appendChild(ring);
      if(active) layer.appendChild(svg('circle',{{cx:p[0],cy:p[1],r:25,fill:'none',stroke:color,'stroke-width':5,class:'pulse'}}));
      layer.appendChild(svg('circle',{{cx:p[0],cy:p[1],r:7,fill:color,opacity}}));
    }} else if(a&&b) {{
      let fraction=1;
      if(active && e.end>e.start) fraction=Math.max(0,Math.min(1,(t-e.start)/(e.end-e.start)));
      const end=[a[0]+(b[0]-a[0])*fraction,a[1]+(b[1]-a[1])*fraction];
      const line=svg('line',{{x1:a[0],y1:a[1],x2:end[0],y2:end[1],stroke:color,'stroke-width':active?10:6,'stroke-linecap':'round',opacity}});
      if(e.type==='pass') line.setAttribute('marker-end',`url(#arrow-team-${{teamKey(e.team)}})`);
      if(e.type==='carry') {{ line.setAttribute('stroke-dasharray','16 11'); line.setAttribute('marker-end','url(#arrow-carry)'); }}
      layer.appendChild(line);
      layer.appendChild(svg('circle',{{cx:a[0],cy:a[1],r:active?10:7,fill:color,stroke:'#0b2d21','stroke-width':4,opacity}}));
      if(fraction>=.98) layer.appendChild(svg('circle',{{cx:b[0],cy:b[1],r:active?11:7,fill:color,stroke:'#0b2d21','stroke-width':4,opacity}}));
    }}
  }});
  document.getElementById('pitch-time').textContent=clock(t);
}}

function drawAggregateChart(rootId,eventType,t,color) {{
  const root=document.getElementById(rootId); root.replaceChildren();
  const W=360,H=104,x0=5,x1=355,y0=8,y1=91;
  const span=Math.max(.1,state.clip.end-state.clip.start);
  const all=visibleEvents.filter(e=>e.type===eventType).sort((a,b)=>a.start-b.start);
  const occurred=all.filter(e=>e.start<=t);
  const maxCount=Math.max(1,all.length);
  const x=value=>x0+Math.max(0,Math.min(1,(value-state.clip.start)/span))*(x1-x0);
  const y=count=>y1-(count/maxCount)*(y1-y0);
  [0,.5,1].forEach(f=>root.appendChild(svg('line',{{x1:x0,y1:y0+f*(y1-y0),x2:x1,y2:y0+f*(y1-y0),stroke:'#17251f','stroke-width':1}})));
  let count=0,d=`M ${{x0}} ${{y(0)}}`;
  occurred.forEach(e=>{{ const xx=x(e.start); d+=` L ${{xx}} ${{y(count)}}`; count+=1; d+=` L ${{xx}} ${{y(count)}}`; }});
  d+=` L ${{x(t)}} ${{y(count)}}`;
  root.appendChild(svg('path',{{d,fill:'none',stroke:color,'stroke-width':3,'stroke-linejoin':'round','stroke-linecap':'round'}}));
  root.appendChild(svg('circle',{{cx:x(t),cy:y(count),r:4,fill:color}}));
  return count;
}}

function drawCharts(t) {{
  const passes=drawAggregateChart('chart-pass','pass',t,colors.pass);
  const carries=drawAggregateChart('chart-carry','carry',t,colors.carry);
  const gains=drawAggregateChart('chart-gain','possession_change',t,colors.possession_change);
  document.getElementById('pass-count').textContent=passes;
  document.getElementById('carry-count').textContent=carries;
  document.getElementById('gain-count').textContent=gains;
}}

function latestEvent(t) {{
  const active=visibleEvents.filter(e=>isMomentActive(e,t)).sort((a,b)=>priority(b)-priority(a));
  if(active.length) return [active[0],true];
  const past=visibleEvents.filter(e=>e.start<=t).sort((a,b)=>b.start-a.start);
  return [past[0]||null,false];
}}
function drawLatest(t) {{
  const [event,active]=latestEvent(t), root=document.getElementById('latest');
  if(!event) {{ root.innerHTML='<div class="empty">—</div>'; return; }}
  const color=colors[event.type]||'#fff'; const stateLabel=active?'Live now':'Latest action';
  root.innerHTML=`<div class="event-row"><div class="event-icon" style="background:${{color}}">${{actionIcon(event.type)}}</div><div class="event-copy"><div class="event-name">${{esc(typeLabel(event.type))}} <span style="color:${{color}};font-size:10px;text-transform:uppercase;letter-spacing:.08em">${{stateLabel}}</span></div><div class="event-detail">${{esc(eventDetail(event))}}</div><div class="confidence"><span style="width:${{event.confidence*100}}%;background:${{color}}"></span></div></div></div>`;
}}
function sync(force=false) {{
  let t=video.currentTime||state.clip.start;
  if(t>state.clip.end+.05) {{
    video.pause(); video.currentTime=state.clip.end; t=state.clip.end;
  }}
  if(!force && Math.abs(t-lastRender)<.025) return; lastRender=t;
  if(radar && radar.readyState && (force || Math.abs(radar.currentTime-t)>.16)) radar.currentTime=Math.min(t,Math.max(0,radar.duration-.05));
  document.getElementById('clock').textContent=clock(t);
  document.getElementById('mode').textContent=video.paused?'Paused':'Playing';
  const ratio=Math.max(0,Math.min(1,(t-state.clip.start)/Math.max(.1,state.clip.end-state.clip.start)));
  document.getElementById('progress').style.width=(ratio*100)+'%';
  drawPitch(t); drawLatest(t); drawCharts(t);
}}
function tick() {{ sync(); requestAnimationFrame(tick); }}
video.addEventListener('loadedmetadata',()=>{{ video.currentTime=Math.min(state.clip.start,Math.max(0,video.duration-.05)); sync(true); }});
if(radar) radar.addEventListener('loadedmetadata',()=>{{ radar.currentTime=Math.min(video.currentTime||state.clip.start,Math.max(0,radar.duration-.05)); }});
video.addEventListener('play',()=>{{ if(radar) radar.play().catch(()=>{{}}); sync(true); }});
video.addEventListener('pause',()=>{{ if(radar) radar.pause(); sync(true); }});
video.addEventListener('seeked',()=>sync(true));
sync(true); requestAnimationFrame(tick);
</script>
</body>
</html>"""
