"use strict";(self["webpackChunk_jupyterlab_application_top"]=self["webpackChunk_jupyterlab_application_top"]||[]).push([[4765],{44765:(e,t,l)=>{l.r(t);l.d(t,{diagram:()=>L});var n=l(96778);var o=l(34596);var a=l(67058);var s=l(23787);var i=l(29395);var c=l(27484);var r=l.n(c);var d=l(17967);var p=l(27856);var b=l.n(p);var f=l(96001);var y=l(21307);const u=e=>s.e.sanitizeText(e,(0,s.c)());let g={dividerMargin:10,padding:5,textHeight:10,curve:void 0};const v=function(e,t,l,n){const o=Object.keys(e);s.l.info("keys:",o);s.l.info(e);o.forEach((function(o){var a,i;const c=e[o];const r="rect";const d={shape:r,id:c.id,domId:c.domId,labelText:u(c.id),labelStyle:"",style:"fill: none; stroke: black",padding:((a=(0,s.c)().flowchart)==null?void 0:a.padding)??((i=(0,s.c)().class)==null?void 0:i.padding)};t.setNode(c.id,d);h(c.classes,t,l,n,c.id);s.l.info("setNode",d)}))};const h=function(e,t,l,n,o){const a=Object.keys(e);s.l.info("keys:",a);s.l.info(e);a.filter((t=>e[t].parent==o)).forEach((function(l){var a,i;const c=e[l];const r=c.cssClasses.join(" ");const d=(0,s.k)(c.styles);const p=c.label??c.id;const b=0;const f="class_box";const y={labelStyle:d.labelStyle,shape:f,labelText:u(p),classData:c,rx:b,ry:b,class:r,style:d.style,id:c.id,domId:c.domId,tooltip:n.db.getTooltip(c.id,o)||"",haveCallback:c.haveCallback,link:c.link,width:c.type==="group"?500:void 0,type:c.type,padding:((a=(0,s.c)().flowchart)==null?void 0:a.padding)??((i=(0,s.c)().class)==null?void 0:i.padding)};t.setNode(c.id,y);if(o){t.setParent(c.id,o)}s.l.info("setNode",y)}))};const w=function(e,t,l,n){s.l.info(e);e.forEach((function(e,a){var i,c;const r=e;const d="";const p={labelStyle:"",style:""};const b=r.text;const f=0;const y="note";const v={labelStyle:p.labelStyle,shape:y,labelText:u(b),noteData:r,rx:f,ry:f,class:d,style:p.style,id:r.id,domId:r.id,tooltip:"",type:"note",padding:((i=(0,s.c)().flowchart)==null?void 0:i.padding)??((c=(0,s.c)().class)==null?void 0:c.padding)};t.setNode(r.id,v);s.l.info("setNode",v);if(!r.class||!(r.class in n)){return}const h=l+a;const w={id:`edgeNote${h}`,classes:"relation",pattern:"dotted",arrowhead:"none",startLabelRight:"",endLabelLeft:"",arrowTypeStart:"none",arrowTypeEnd:"none",style:"fill:none",labelStyle:"",curve:(0,s.n)(g.curve,o.c_6)};t.setEdge(r.id,r.class,w,h)}))};const k=function(e,t){const l=(0,s.c)().flowchart;let n=0;e.forEach((function(e){var a;n++;const i={classes:"relation",pattern:e.relation.lineType==1?"dashed":"solid",id:`id_${e.id1}_${e.id2}_${n}`,arrowhead:e.type==="arrow_open"?"none":"normal",startLabelRight:e.relationTitle1==="none"?"":e.relationTitle1,endLabelLeft:e.relationTitle2==="none"?"":e.relationTitle2,arrowTypeStart:T(e.relation.type1),arrowTypeEnd:T(e.relation.type2),style:"fill:none",labelStyle:"",curve:(0,s.n)(l==null?void 0:l.curve,o.c_6)};s.l.info(i,e);if(e.style!==void 0){const t=(0,s.k)(e.style);i.style=t.style;i.labelStyle=t.labelStyle}e.text=e.title;if(e.text===void 0){if(e.style!==void 0){i.arrowheadStyle="fill: #333"}}else{i.arrowheadStyle="fill: #333";i.labelpos="c";if(((a=(0,s.c)().flowchart)==null?void 0:a.htmlLabels)??(0,s.c)().htmlLabels){i.labelType="html";i.label='<span class="edgeLabel">'+e.text+"</span>"}else{i.labelType="text";i.label=e.text.replace(s.e.lineBreakRegex,"\n");if(e.style===void 0){i.style=i.style||"stroke: #333; stroke-width: 1.5px;fill:none"}i.labelStyle=i.labelStyle.replace("color:","fill:")}}t.setEdge(e.id1,e.id2,i,n)}))};const x=function(e){g={...g,...e}};const m=async function(e,t,l,n){s.l.info("Drawing class - ",t);const c=(0,s.c)().flowchart??(0,s.c)().class;const r=(0,s.c)().securityLevel;s.l.info("config:",c);const d=(c==null?void 0:c.nodeSpacing)??50;const p=(c==null?void 0:c.rankSpacing)??50;const b=new a.k({multigraph:true,compound:true}).setGraph({rankdir:n.db.getDirection(),nodesep:d,ranksep:p,marginx:8,marginy:8}).setDefaultEdgeLabel((function(){return{}}));const f=n.db.getNamespaces();const y=n.db.getClasses();const u=n.db.getRelations();const g=n.db.getNotes();s.l.info(u);v(f,b,t,n);h(y,b,t,n);k(u,b);w(g,b,u.length+1,y);let x;if(r==="sandbox"){x=(0,o.Ys)("#i"+t)}const m=r==="sandbox"?(0,o.Ys)(x.nodes()[0].contentDocument.body):(0,o.Ys)("body");const T=m.select(`[id="${t}"]`);const S=m.select("#"+t+" g");await(0,i.r)(S,b,["aggregation","extension","composition","dependency","lollipop"],"classDiagram",t);s.u.insertTitle(T,"classTitleText",(c==null?void 0:c.titleTopMargin)??5,n.db.getDiagramTitle());(0,s.o)(b,T,c==null?void 0:c.diagramPadding,c==null?void 0:c.useMaxWidth);if(!(c==null?void 0:c.htmlLabels)){const e=r==="sandbox"?x.nodes()[0].contentDocument:document;const l=e.querySelectorAll('[id="'+t+'"] .edgeLabel .label');for(const t of l){const l=t.getBBox();const n=e.createElementNS("http://www.w3.org/2000/svg","rect");n.setAttribute("rx",0);n.setAttribute("ry",0);n.setAttribute("width",l.width);n.setAttribute("height",l.height);t.insertBefore(n,t.firstChild)}}};function T(e){let t;switch(e){case 0:t="aggregation";break;case 1:t="extension";break;case 2:t="composition";break;case 3:t="dependency";break;case 4:t="lollipop";break;default:t="none"}return t}const S={setConf:x,draw:m};const L={parser:n.p,db:n.d,renderer:S,styles:n.s,init:e=>{if(!e.class){e.class={}}e.class.arrowMarkerAbsolute=e.arrowMarkerAbsolute;n.d.clear()}}}}]);
//# sourceMappingURL=4765.ba0109db39f25ddf020a.js.map?v=ba0109db39f25ddf020a