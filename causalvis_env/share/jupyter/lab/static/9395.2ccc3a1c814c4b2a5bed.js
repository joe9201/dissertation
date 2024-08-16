"use strict";(self["webpackChunk_jupyterlab_application_top"]=self["webpackChunk_jupyterlab_application_top"]||[]).push([[9395],{21307:(e,t,n)=>{n.d(t,{c:()=>c});var r=n(99250);var i=n(15220);var a=4;function o(e){return(0,i.Z)(e,a)}const s=o;var d=n(30014);var l=n(85389);function c(e){var t={options:{directed:e.isDirected(),multigraph:e.isMultigraph(),compound:e.isCompound()},nodes:h(e),edges:g(e)};if(!r.Z(e.graph())){t.value=s(e.graph())}return t}function h(e){return d.Z(e.nodes(),(function(t){var n=e.node(t);var i=e.parent(t);var a={v:t};if(!r.Z(n)){a.value=n}if(!r.Z(i)){a.parent=i}return a}))}function g(e){return d.Z(e.edges(),(function(t){var n=e.edge(t);var i={v:t.v,w:t.w};if(!r.Z(t.name)){i.name=t.name}if(!r.Z(n)){i.value=n}return i}))}function f(e){var t=new Graph(e.options).setGraph(e.value);_.each(e.nodes,(function(e){t.setNode(e.v,e.value);if(e.parent){t.setParent(e.v,e.parent)}}));_.each(e.edges,(function(e){t.setEdge({v:e.v,w:e.w,name:e.name},e.value)}));return t}},29395:(e,t,n)=>{n.d(t,{r:()=>G});var r=n(96001);var i=n(21307);var a=n(27734);var o=n(23787);var s=n(67058);var d=n(66385);var l=n(34596);let c={};let h={};let g={};const f=()=>{h={};g={};c={}};const u=(e,t)=>{o.l.trace("In isDecendant",t," ",e," = ",h[t].includes(e));if(h[t].includes(e)){return true}return false};const w=(e,t)=>{o.l.info("Decendants of ",t," is ",h[t]);o.l.info("Edge is ",e);if(e.v===t){return false}if(e.w===t){return false}if(!h[t]){o.l.debug("Tilt, ",t,",not in decendants");return false}return h[t].includes(e.v)||u(e.v,t)||u(e.w,t)||h[t].includes(e.w)};const p=(e,t,n,r)=>{o.l.warn("Copying children of ",e,"root",r,"data",t.node(e),r);const i=t.children(e)||[];if(e!==r){i.push(e)}o.l.warn("Copying (nodes) clusterId",e,"nodes",i);i.forEach((i=>{if(t.children(i).length>0){p(i,t,n,r)}else{const a=t.node(i);o.l.info("cp ",i," to ",r," with parent ",e);n.setNode(i,a);if(r!==t.parent(i)){o.l.warn("Setting parent",i,t.parent(i));n.setParent(i,t.parent(i))}if(e!==r&&i!==e){o.l.debug("Setting parent",i,e);n.setParent(i,e)}else{o.l.info("In copy ",e,"root",r,"data",t.node(e),r);o.l.debug("Not Setting parent for node=",i,"cluster!==rootId",e!==r,"node!==clusterId",i!==e)}const s=t.edges(i);o.l.debug("Copying Edges",s);s.forEach((i=>{o.l.info("Edge",i);const a=t.edge(i.v,i.w,i.name);o.l.info("Edge data",a,r);try{if(w(i,r)){o.l.info("Copying as ",i.v,i.w,a,i.name);n.setEdge(i.v,i.w,a,i.name);o.l.info("newGraph edges ",n.edges(),n.edge(n.edges()[0]))}else{o.l.info("Skipping copy of edge ",i.v,"--\x3e",i.w," rootId: ",r," clusterId:",e)}}catch(s){o.l.error(s)}}))}o.l.debug("Removing node",i);t.removeNode(i)}))};const v=(e,t)=>{const n=t.children(e);let r=[...n];for(const i of n){g[i]=e;r=[...r,...v(i,t)]}return r};const y=(e,t)=>{o.l.trace("Searching",e);const n=t.children(e);o.l.trace("Searching children of id ",e,n);if(n.length<1){o.l.trace("This is a valid node",e);return e}for(const r of n){const n=y(r,t);if(n){o.l.trace("Found replacement for",e," => ",n);return n}}};const m=e=>{if(!c[e]){return e}if(!c[e].externalConnections){return e}if(c[e]){return c[e].id}return e};const x=(e,t)=>{if(!e||t>10){o.l.debug("Opting out, no graph ");return}else{o.l.debug("Opting in, graph ")}e.nodes().forEach((function(t){const n=e.children(t);if(n.length>0){o.l.warn("Cluster identified",t," Replacement id in edges: ",y(t,e));h[t]=v(t,e);c[t]={id:y(t,e),clusterData:e.node(t)}}}));e.nodes().forEach((function(t){const n=e.children(t);const r=e.edges();if(n.length>0){o.l.debug("Cluster identified",t,h);r.forEach((e=>{if(e.v!==t&&e.w!==t){const n=u(e.v,t);const r=u(e.w,t);if(n^r){o.l.warn("Edge: ",e," leaves cluster ",t);o.l.warn("Decendants of XXX ",t,": ",h[t]);c[t].externalConnections=true}}}))}else{o.l.debug("Not a cluster ",t,h)}}));e.edges().forEach((function(t){const n=e.edge(t);o.l.warn("Edge "+t.v+" -> "+t.w+": "+JSON.stringify(t));o.l.warn("Edge "+t.v+" -> "+t.w+": "+JSON.stringify(e.edge(t)));let r=t.v;let i=t.w;o.l.warn("Fix XXX",c,"ids:",t.v,t.w,"Translating: ",c[t.v]," --- ",c[t.w]);if(c[t.v]&&c[t.w]&&c[t.v]===c[t.w]){o.l.warn("Fixing and trixing link to self - removing XXX",t.v,t.w,t.name);o.l.warn("Fixing and trixing - removing XXX",t.v,t.w,t.name);r=m(t.v);i=m(t.w);e.removeEdge(t.v,t.w,t.name);const a=t.w+"---"+t.v;e.setNode(a,{domId:a,id:a,labelStyle:"",labelText:n.label,padding:0,shape:"labelRect",style:""});const s=structuredClone(n);const d=structuredClone(n);s.label="";s.arrowTypeEnd="none";d.label="";s.fromCluster=t.v;d.toCluster=t.v;e.setEdge(r,a,s,t.name+"-cyclic-special");e.setEdge(a,i,d,t.name+"-cyclic-special")}else if(c[t.v]||c[t.w]){o.l.warn("Fixing and trixing - removing XXX",t.v,t.w,t.name);r=m(t.v);i=m(t.w);e.removeEdge(t.v,t.w,t.name);if(r!==t.v){n.fromCluster=t.v}if(i!==t.w){n.toCluster=t.w}o.l.warn("Fix Replacing with XXX",r,i,t.name);e.setEdge(r,i,n,t.name)}}));o.l.warn("Adjusted Graph",i.c(e));b(e,0);o.l.trace(c)};const b=(e,t)=>{o.l.warn("extractor - ",t,i.c(e),e.children("D"));if(t>10){o.l.error("Bailing out");return}let n=e.nodes();let r=false;for(const i of n){const t=e.children(i);r=r||t.length>0}if(!r){o.l.debug("Done, no node has children",e.nodes());return}o.l.debug("Nodes = ",n,t);for(const a of n){o.l.debug("Extracting node",a,c,c[a]&&!c[a].externalConnections,!e.parent(a),e.node(a),e.children("D")," Depth ",t);if(!c[a]){o.l.debug("Not a cluster",a,t)}else if(!c[a].externalConnections&&e.children(a)&&e.children(a).length>0){o.l.warn("Cluster without external connections, without a parent and with children",a,t);const n=e.graph();let r=n.rankdir==="TB"?"LR":"TB";if(c[a]&&c[a].clusterData&&c[a].clusterData.dir){r=c[a].clusterData.dir;o.l.warn("Fixing dir",c[a].clusterData.dir,r)}const d=new s.k({multigraph:true,compound:true}).setGraph({rankdir:r,nodesep:50,ranksep:50,marginx:8,marginy:8}).setDefaultEdgeLabel((function(){return{}}));o.l.warn("Old graph before copy",i.c(e));p(a,e,d,a);e.setNode(a,{clusterNode:true,id:a,clusterData:c[a].clusterData,labelText:c[a].labelText,graph:d});o.l.warn("New graph after copy node: (",a,")",i.c(d));o.l.debug("Old graph after copy",i.c(e))}else{o.l.warn("Cluster ** ",a," **not meeting the criteria !externalConnections:",!c[a].externalConnections," no parent: ",!e.parent(a)," children ",e.children(a)&&e.children(a).length>0,e.children("D"),t);o.l.debug(c)}}n=e.nodes();o.l.warn("New list of nodes",n);for(const i of n){const n=e.node(i);o.l.warn(" Now next level",i,n);if(n.clusterNode){b(n.graph,t+1)}}};const N=(e,t)=>{if(t.length===0){return[]}let n=Object.assign(t);t.forEach((t=>{const r=e.children(t);const i=N(e,r);n=[...n,...i]}));return n};const E=e=>N(e,e.children());const X=(e,t)=>{o.l.info("Creating subgraph rect for ",t.id,t);const n=e.insert("g").attr("class","cluster"+(t.class?" "+t.class:"")).attr("id",t.id);const r=n.insert("rect",":first-child");const i=(0,o.m)((0,o.c)().flowchart.htmlLabels);const s=n.insert("g").attr("class","cluster-label");const c=t.labelType==="markdown"?(0,d.a)(s,t.labelText,{style:t.labelStyle,useHtmlLabels:i}):s.node().appendChild((0,a.c)(t.labelText,t.labelStyle,void 0,true));let h=c.getBBox();if((0,o.m)((0,o.c)().flowchart.htmlLabels)){const e=c.children[0];const t=(0,l.Ys)(c);h=e.getBoundingClientRect();t.attr("width",h.width);t.attr("height",h.height)}const g=0*t.padding;const f=g/2;const u=t.width<=h.width+g?h.width+g:t.width;if(t.width<=h.width+g){t.diff=(h.width-t.width)/2-t.padding/2}else{t.diff=-t.padding/2}o.l.trace("Data ",t,JSON.stringify(t));r.attr("style",t.style).attr("rx",t.rx).attr("ry",t.ry).attr("x",t.x-u/2).attr("y",t.y-t.height/2-f).attr("width",u).attr("height",t.height+g);if(i){s.attr("transform","translate("+(t.x-h.width/2)+", "+(t.y-t.height/2)+")")}else{s.attr("transform","translate("+t.x+", "+(t.y-t.height/2)+")")}const w=r.node().getBBox();t.width=w.width;t.height=w.height;t.intersect=function(e){return(0,a.i)(t,e)};return n};const C=(e,t)=>{const n=e.insert("g").attr("class","note-cluster").attr("id",t.id);const r=n.insert("rect",":first-child");const i=0*t.padding;const o=i/2;r.attr("rx",t.rx).attr("ry",t.ry).attr("x",t.x-t.width/2-o).attr("y",t.y-t.height/2-o).attr("width",t.width+i).attr("height",t.height+i).attr("fill","none");const s=r.node().getBBox();t.width=s.width;t.height=s.height;t.intersect=function(e){return(0,a.i)(t,e)};return n};const S=(e,t)=>{const n=e.insert("g").attr("class",t.classes).attr("id",t.id);const r=n.insert("rect",":first-child");const i=n.insert("g").attr("class","cluster-label");const s=n.append("rect");const d=i.node().appendChild((0,a.c)(t.labelText,t.labelStyle,void 0,true));let c=d.getBBox();if((0,o.m)((0,o.c)().flowchart.htmlLabels)){const e=d.children[0];const t=(0,l.Ys)(d);c=e.getBoundingClientRect();t.attr("width",c.width);t.attr("height",c.height)}c=d.getBBox();const h=0*t.padding;const g=h/2;const f=t.width<=c.width+t.padding?c.width+t.padding:t.width;if(t.width<=c.width+t.padding){t.diff=(c.width+t.padding*0-t.width)/2}else{t.diff=-t.padding/2}r.attr("class","outer").attr("x",t.x-f/2-g).attr("y",t.y-t.height/2-g).attr("width",f+h).attr("height",t.height+h);s.attr("class","inner").attr("x",t.x-f/2-g).attr("y",t.y-t.height/2-g+c.height-1).attr("width",f+h).attr("height",t.height+h-c.height-3);i.attr("transform","translate("+(t.x-c.width/2)+", "+(t.y-t.height/2-t.padding/3+((0,o.m)((0,o.c)().flowchart.htmlLabels)?5:3))+")");const u=r.node().getBBox();t.height=u.height;t.intersect=function(e){return(0,a.i)(t,e)};return n};const D=(e,t)=>{const n=e.insert("g").attr("class",t.classes).attr("id",t.id);const r=n.insert("rect",":first-child");const i=0*t.padding;const o=i/2;r.attr("class","divider").attr("x",t.x-t.width/2-o).attr("y",t.y-t.height/2).attr("width",t.width+i).attr("height",t.height+i);const s=r.node().getBBox();t.width=s.width;t.height=s.height;t.diff=-t.padding/2;t.intersect=function(e){return(0,a.i)(t,e)};return n};const B={rect:X,roundedWithTitle:S,noteGroup:C,divider:D};let O={};const T=(e,t)=>{o.l.trace("Inserting cluster");const n=t.shape||"rect";O[t.id]=B[n](e,t)};const J=()=>{O={}};const k=async(e,t,n,s,d)=>{o.l.info("Graph in recursive render: XXX",i.c(t),d);const l=t.graph().rankdir;o.l.trace("Dir in recursive render - dir:",l);const h=e.insert("g").attr("class","root");if(!t.nodes()){o.l.info("No nodes found for",t)}else{o.l.info("Recursive render XXX",t.nodes())}if(t.edges().length>0){o.l.trace("Recursive edges",t.edge(t.edges()[0]))}const g=h.insert("g").attr("class","clusters");const f=h.insert("g").attr("class","edgePaths");const u=h.insert("g").attr("class","edgeLabels");const w=h.insert("g").attr("class","nodes");await Promise.all(t.nodes().map((async function(e){const r=t.node(e);if(d!==void 0){const n=JSON.parse(JSON.stringify(d.clusterData));o.l.info("Setting data for cluster XXX (",e,") ",n,d);t.setNode(d.id,n);if(!t.parent(e)){o.l.trace("Setting parent",e,d.id);t.setParent(e,d.id,n)}}o.l.info("(Insert) Node XXX"+e+": "+JSON.stringify(t.node(e)));if(r&&r.clusterNode){o.l.info("Cluster identified",e,r.width,t.node(e));const i=await k(w,r.graph,n,s,t.node(e));const d=i.elem;(0,a.u)(r,d);r.diff=i.diff||0;o.l.info("Node bounds (abc123)",e,r,r.width,r.x,r.y);(0,a.s)(d,r);o.l.warn("Recursive render complete ",d,r)}else{if(t.children(e).length>0){o.l.info("Cluster - the non recursive path XXX",e,r.id,r,t);o.l.info(y(r.id,t));c[r.id]={id:y(r.id,t),node:r}}else{o.l.info("Node - the non recursive path",e,r.id,r);await(0,a.e)(w,t.node(e),l)}}})));t.edges().forEach((function(e){const n=t.edge(e.v,e.w,e.name);o.l.info("Edge "+e.v+" -> "+e.w+": "+JSON.stringify(e));o.l.info("Edge "+e.v+" -> "+e.w+": ",e," ",JSON.stringify(t.edge(e)));o.l.info("Fix",c,"ids:",e.v,e.w,"Translateing: ",c[e.v],c[e.w]);(0,a.f)(u,n)}));t.edges().forEach((function(e){o.l.info("Edge "+e.v+" -> "+e.w+": "+JSON.stringify(e))}));o.l.info("#############################################");o.l.info("###                Layout                 ###");o.l.info("#############################################");o.l.info(t);(0,r.bK)(t);o.l.info("Graph after layout:",i.c(t));let p=0;E(t).forEach((function(e){const n=t.node(e);o.l.info("Position "+e+": "+JSON.stringify(t.node(e)));o.l.info("Position "+e+": ("+n.x,","+n.y,") width: ",n.width," height: ",n.height);if(n&&n.clusterNode){(0,a.p)(n)}else{if(t.children(e).length>0){T(g,n);c[n.id].node=n}else{(0,a.p)(n)}}}));t.edges().forEach((function(e){const r=t.edge(e);o.l.info("Edge "+e.v+" -> "+e.w+": "+JSON.stringify(r),r);const i=(0,a.g)(f,e,r,c,n,t,s);(0,a.h)(r,i)}));t.nodes().forEach((function(e){const n=t.node(e);o.l.info(e,n.type,n.diff);if(n.type==="group"){p=n.diff}}));return{elem:h,diff:p}};const G=async(e,t,n,r,s)=>{(0,a.a)(e,n,r,s);(0,a.b)();(0,a.d)();J();f();o.l.warn("Graph at first:",JSON.stringify(i.c(t)));x(t);o.l.warn("Graph after:",JSON.stringify(i.c(t)));await k(e,t,r,s)}}}]);
//# sourceMappingURL=9395.2ccc3a1c814c4b2a5bed.js.map?v=2ccc3a1c814c4b2a5bed