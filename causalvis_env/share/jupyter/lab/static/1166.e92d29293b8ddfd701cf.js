"use strict";(self["webpackChunk_jupyterlab_application_top"]=self["webpackChunk_jupyterlab_application_top"]||[]).push([[1166],{1497:(t,e,r)=>{r.d(e,{a:()=>a});var n=r(65915);function a(t,e){var r=t.append("foreignObject").attr("width","100000");var a=r.append("xhtml:div");a.attr("xmlns","http://www.w3.org/1999/xhtml");var o=e.label;switch(typeof o){case"function":a.insert(o);break;case"object":a.insert((function(){return o}));break;default:a.html(o)}n.bg(a,e.labelStyle);a.style("display","inline-block");a.style("white-space","nowrap");var i=a.node().getBoundingClientRect();r.attr("width",i.width).attr("height",i.height);return r}},65915:(t,e,r)=>{r.d(e,{$p:()=>d,O1:()=>i,WR:()=>h,bF:()=>o,bg:()=>c});var n=r(30353);var a=r(25069);function o(t,e){return!!t.children(e).length}function i(t){return s(t.v)+":"+s(t.w)+":"+s(t.name)}var l=/:/g;function s(t){return t?String(t).replace(l,"\\:"):""}function c(t,e){if(e){t.attr("style",e)}}function d(t,e,r){if(e){t.attr("class",e).attr("class",r+" "+t.attr("class"))}}function h(t,e){var r=e.graph();if(n.Z(r)){var o=r.transition;if(a.Z(o)){return o(t)}}return t}},31166:(t,e,r)=>{r.r(e);r.d(e,{diagram:()=>Mt});var n=r(71929);var a=r(67058);var o=r(34596);var i=r(23787);var l=r(19055);var s=r(51698);var c=r(81277);var d=r(96001);var h=r(65915);var u={normal:p,vee:g,undirected:b};function f(t){u=t}function p(t,e,r,n){var a=t.append("marker").attr("id",e).attr("viewBox","0 0 10 10").attr("refX",9).attr("refY",5).attr("markerUnits","strokeWidth").attr("markerWidth",8).attr("markerHeight",6).attr("orient","auto");var o=a.append("path").attr("d","M 0 0 L 10 5 L 0 10 z").style("stroke-width",1).style("stroke-dasharray","1,0");h.bg(o,r[n+"Style"]);if(r[n+"Class"]){o.attr("class",r[n+"Class"])}}function g(t,e,r,n){var a=t.append("marker").attr("id",e).attr("viewBox","0 0 10 10").attr("refX",9).attr("refY",5).attr("markerUnits","strokeWidth").attr("markerWidth",8).attr("markerHeight",6).attr("orient","auto");var o=a.append("path").attr("d","M 0 0 L 10 5 L 0 10 L 4 5 z").style("stroke-width",1).style("stroke-dasharray","1,0");h.bg(o,r[n+"Style"]);if(r[n+"Class"]){o.attr("class",r[n+"Class"])}}function b(t,e,r,n){var a=t.append("marker").attr("id",e).attr("viewBox","0 0 10 10").attr("refX",9).attr("refY",5).attr("markerUnits","strokeWidth").attr("markerWidth",8).attr("markerHeight",6).attr("orient","auto");var o=a.append("path").attr("d","M 0 5 L 10 5").style("stroke-width",1).style("stroke-dasharray","1,0");h.bg(o,r[n+"Style"]);if(r[n+"Class"]){o.attr("class",r[n+"Class"])}}var y=r(1497);function v(t,e){var r=t;r.node().appendChild(e.label);h.bg(r,e.labelStyle);return r}function w(t,e){var r=t.append("text");var n=x(e.label).split("\n");for(var a=0;a<n.length;a++){r.append("tspan").attr("xml:space","preserve").attr("dy","1em").attr("x","1").text(n[a])}h.bg(r,e.labelStyle);return r}function x(t){var e="";var r=false;var n;for(var a=0;a<t.length;++a){n=t[a];if(r){switch(n){case"n":e+="\n";break;default:e+=n}r=false}else if(n==="\\"){r=true}else{e+=n}}return e}function k(t,e,r){var n=e.label;var a=t.append("g");if(e.labelType==="svg"){v(a,e)}else if(typeof n!=="string"||e.labelType==="html"){(0,y.a)(a,e)}else{w(a,e)}var o=a.node().getBBox();var i;switch(r){case"top":i=-e.height/2;break;case"bottom":i=e.height/2-o.height;break;default:i=-o.height/2}a.attr("transform","translate("+-o.width/2+","+i+")");return a}var m=function(t,e){var r=e.nodes().filter((function(t){return h.bF(e,t)}));var n=t.selectAll("g.cluster").data(r,(function(t){return t}));h.WR(n.exit(),e).style("opacity",0).remove();var a=n.enter().append("g").attr("class","cluster").attr("id",(function(t){var r=e.node(t);return r.id})).style("opacity",0).each((function(t){var r=e.node(t);var n=o.Ys(this);o.Ys(this).append("rect");var a=n.append("g").attr("class","label");k(a,r,r.clusterLabelPos)}));n=n.merge(a);n=h.WR(n,e).style("opacity",1);n.selectAll("rect").each((function(t){var r=e.node(t);var n=o.Ys(this);h.bg(n,r.style)}));return n};function _(t){m=t}let S=function(t,e){var r=t.selectAll("g.edgeLabel").data(e.edges(),(function(t){return h.O1(t)})).classed("update",true);r.exit().remove();r.enter().append("g").classed("edgeLabel",true).style("opacity",0);r=t.selectAll("g.edgeLabel");r.each((function(t){var r=o.Ys(this);r.select(".label").remove();var n=e.edge(t);var a=k(r,e.edge(t),0).classed("label",true);var i=a.node().getBBox();if(n.labelId){a.attr("id",n.labelId)}if(!l.Z(n,"width")){n.width=i.width}if(!l.Z(n,"height")){n.height=i.height}}));var n;if(r.exit){n=r.exit()}else{n=r.selectAll(null)}h.WR(n,e).style("opacity",0).remove();return r};function T(t){S=t}var L=r(855);var C=r(5315);function A(t,e){return t.intersect(e)}var E=function(t,e,r){var n=t.selectAll("g.edgePath").data(e.edges(),(function(t){return h.O1(t)})).classed("update",true);var a=R(n,e);Y(n,e);var i=n.merge!==undefined?n.merge(a):n;h.WR(i,e).style("opacity",1);i.each((function(t){var r=o.Ys(this);var n=e.edge(t);n.elem=this;if(n.id){r.attr("id",n.id)}h.$p(r,n["class"],(r.classed("update")?"update ":"")+"edgePath")}));i.selectAll("path.path").each((function(t){var r=e.edge(t);r.arrowheadId=L.Z("arrowhead");var n=o.Ys(this).attr("marker-end",(function(){return"url("+$(location.href,r.arrowheadId)+")"})).style("fill","none");h.WR(n,e).attr("d",(function(t){return B(e,t)}));h.bg(n,r.style)}));i.selectAll("defs *").remove();i.selectAll("defs").each((function(t){var n=e.edge(t);var a=r[n.arrowhead];a(o.Ys(this),n.arrowheadId,n,"arrowhead")}));return i};function N(t){E=t}function $(t,e){var r=t.split("#")[0];return r+"#"+e}function B(t,e){var r=t.edge(e);var n=t.node(e.v);var a=t.node(e.w);var o=r.points.slice(1,r.points.length-1);o.unshift(A(n,o[0]));o.push(A(a,o[o.length-1]));return I(r,o)}function I(t,e){var r=(o.jvg||o.YPS.line)().x((function(t){return t.x})).y((function(t){return t.y}));(r.curve||r.interpolate)(t.curve);return r(e)}function M(t){var e=t.getBBox();var r=t.ownerSVGElement.getScreenCTM().inverse().multiply(t.getScreenCTM()).translate(e.width/2,e.height/2);return{x:r.e,y:r.f}}function R(t,e){var r=t.enter().append("g").attr("class","edgePath").style("opacity",0);r.append("path").attr("class","path").attr("d",(function(t){var r=e.edge(t);var n=e.node(t.v).elem;var a=C.Z(r.points.length).map((function(){return M(n)}));return I(r,a)}));r.append("defs");return r}function Y(t,e){var r=t.exit();h.WR(r,e).style("opacity",0).remove()}var Z=r(62957);var D=function(t,e,r){var n=e.nodes().filter((function(t){return!h.bF(e,t)}));var a=t.selectAll("g.node").data(n,(function(t){return t})).classed("update",true);a.exit().remove();a.enter().append("g").attr("class","node").style("opacity",0);a=t.selectAll("g.node");a.each((function(t){var n=e.node(t);var a=o.Ys(this);h.$p(a,n["class"],(a.classed("update")?"update ":"")+"node");a.select("g.label").remove();var i=a.append("g").attr("class","label");var s=k(i,n);var c=r[n.shape];var d=Z.Z(s.node().getBBox(),"width","height");n.elem=this;if(n.id){a.attr("id",n.id)}if(n.labelId){i.attr("id",n.labelId)}if(l.Z(n,"width")){d.width=n.width}if(l.Z(n,"height")){d.height=n.height}d.width+=n.paddingLeft+n.paddingRight;d.height+=n.paddingTop+n.paddingBottom;i.attr("transform","translate("+(n.paddingLeft-n.paddingRight)/2+","+(n.paddingTop-n.paddingBottom)/2+")");var u=o.Ys(this);u.select(".label-container").remove();var f=c(u,d,n).classed("label-container",true);h.bg(f,n.style);var p=f.node().getBBox();n.width=p.width;n.height=p.height}));var i;if(a.exit){i=a.exit()}else{i=a.selectAll(null)}h.WR(i,e).style("opacity",0).remove();return a};function W(t){D=t}function U(t,e){var r=t.filter((function(){return!o.Ys(this).classed("update")}));function n(t){var r=e.node(t);return"translate("+r.x+","+r.y+")"}r.attr("transform",n);h.WR(t,e).style("opacity",1).attr("transform",n);h.WR(r.selectAll("rect"),e).attr("width",(function(t){return e.node(t).width})).attr("height",(function(t){return e.node(t).height})).attr("x",(function(t){var r=e.node(t);return-r.width/2})).attr("y",(function(t){var r=e.node(t);return-r.height/2}))}function z(t,e){var r=t.filter((function(){return!o.Ys(this).classed("update")}));function n(t){var r=e.edge(t);return l.Z(r,"x")?"translate("+r.x+","+r.y+")":""}r.attr("transform",n);h.WR(t,e).style("opacity",1).attr("transform",n)}function j(t,e){var r=t.filter((function(){return!o.Ys(this).classed("update")}));function n(t){var r=e.node(t);return"translate("+r.x+","+r.y+")"}r.attr("transform",n);h.WR(t,e).style("opacity",1).attr("transform",n)}function O(t,e,r,n){var a=t.x;var o=t.y;var i=a-n.x;var l=o-n.y;var s=Math.sqrt(e*e*l*l+r*r*i*i);var c=Math.abs(e*r*i/s);if(n.x<a){c=-c}var d=Math.abs(e*r*l/s);if(n.y<o){d=-d}return{x:a+c,y:o+d}}function P(t,e,r){return O(t,e,e,r)}function q(t,e,r,n){var a,o,i,l,s,c;var d,h,u,f;var p,g,b;var y,v;a=e.y-t.y;i=t.x-e.x;s=e.x*t.y-t.x*e.y;u=a*r.x+i*r.y+s;f=a*n.x+i*n.y+s;if(u!==0&&f!==0&&V(u,f)){return}o=n.y-r.y;l=r.x-n.x;c=n.x*r.y-r.x*n.y;d=o*t.x+l*t.y+c;h=o*e.x+l*e.y+c;if(d!==0&&h!==0&&V(d,h)){return}p=a*l-o*i;if(p===0){return}g=Math.abs(p/2);b=i*c-l*s;y=b<0?(b-g)/p:(b+g)/p;b=o*s-a*c;v=b<0?(b-g)/p:(b+g)/p;return{x:y,y:v}}function V(t,e){return t*e>0}function X(t,e,r){var n=t.x;var a=t.y;var o=[];var i=Number.POSITIVE_INFINITY;var l=Number.POSITIVE_INFINITY;e.forEach((function(t){i=Math.min(i,t.x);l=Math.min(l,t.y)}));var s=n-t.width/2-i;var c=a-t.height/2-l;for(var d=0;d<e.length;d++){var h=e[d];var u=e[d<e.length-1?d+1:0];var f=q(t,r,{x:s+h.x,y:c+h.y},{x:s+u.x,y:c+u.y});if(f){o.push(f)}}if(!o.length){console.log("NO INTERSECTION FOUND, RETURN NODE CENTER",t);return t}if(o.length>1){o.sort((function(t,e){var n=t.x-r.x;var a=t.y-r.y;var o=Math.sqrt(n*n+a*a);var i=e.x-r.x;var l=e.y-r.y;var s=Math.sqrt(i*i+l*l);return o<s?-1:o===s?0:1}))}return o[0]}function F(t,e){var r=t.x;var n=t.y;var a=e.x-r;var o=e.y-n;var i=t.width/2;var l=t.height/2;var s,c;if(Math.abs(o)*i>Math.abs(a)*l){if(o<0){l=-l}s=o===0?0:l*a/o;c=l}else{if(a<0){i=-i}s=i;c=a===0?0:i*o/a}return{x:r+s,y:n+c}}var G={rect:Q,ellipse:K,circle:J,diamond:tt};function H(t){G=t}function Q(t,e,r){var n=t.insert("rect",":first-child").attr("rx",r.rx).attr("ry",r.ry).attr("x",-e.width/2).attr("y",-e.height/2).attr("width",e.width).attr("height",e.height);r.intersect=function(t){return F(r,t)};return n}function K(t,e,r){var n=e.width/2;var a=e.height/2;var o=t.insert("ellipse",":first-child").attr("x",-e.width/2).attr("y",-e.height/2).attr("rx",n).attr("ry",a);r.intersect=function(t){return O(r,n,a,t)};return o}function J(t,e,r){var n=Math.max(e.width,e.height)/2;var a=t.insert("circle",":first-child").attr("x",-e.width/2).attr("y",-e.height/2).attr("r",n);r.intersect=function(t){return P(r,n,t)};return a}function tt(t,e,r){var n=e.width*Math.SQRT2/2;var a=e.height*Math.SQRT2/2;var o=[{x:0,y:-a},{x:-n,y:0},{x:0,y:a},{x:n,y:0}];var i=t.insert("polygon",":first-child").attr("points",o.map((function(t){return t.x+","+t.y})).join(" "));r.intersect=function(t){return X(r,o,t)};return i}function et(){var t=function(t,e){at(e);var r=it(t,"output");var n=it(r,"clusters");var a=it(r,"edgePaths");var o=S(it(r,"edgeLabels"),e);var i=D(it(r,"nodes"),e,G);(0,d.bK)(e);j(i,e);z(o,e);E(a,e,u);var l=m(n,e);U(l,e);ot(e)};t.createNodes=function(e){if(!arguments.length)return D;W(e);return t};t.createClusters=function(e){if(!arguments.length)return m;_(e);return t};t.createEdgeLabels=function(e){if(!arguments.length)return S;T(e);return t};t.createEdgePaths=function(e){if(!arguments.length)return E;N(e);return t};t.shapes=function(e){if(!arguments.length)return G;H(e);return t};t.arrows=function(e){if(!arguments.length)return u;f(e);return t};return t}var rt={paddingLeft:10,paddingRight:10,paddingTop:10,paddingBottom:10,rx:0,ry:0,shape:"rect"};var nt={arrowhead:"normal",curve:o.c_6};function at(t){t.nodes().forEach((function(e){var r=t.node(e);if(!l.Z(r,"label")&&!t.children(e).length){r.label=e}if(l.Z(r,"paddingX")){s.Z(r,{paddingLeft:r.paddingX,paddingRight:r.paddingX})}if(l.Z(r,"paddingY")){s.Z(r,{paddingTop:r.paddingY,paddingBottom:r.paddingY})}if(l.Z(r,"padding")){s.Z(r,{paddingLeft:r.padding,paddingRight:r.padding,paddingTop:r.padding,paddingBottom:r.padding})}s.Z(r,rt);c.Z(["paddingLeft","paddingRight","paddingTop","paddingBottom"],(function(t){r[t]=Number(r[t])}));if(l.Z(r,"width")){r._prevWidth=r.width}if(l.Z(r,"height")){r._prevHeight=r.height}}));t.edges().forEach((function(e){var r=t.edge(e);if(!l.Z(r,"label")){r.label=""}s.Z(r,nt)}))}function ot(t){c.Z(t.nodes(),(function(e){var r=t.node(e);if(l.Z(r,"_prevWidth")){r.width=r._prevWidth}else{delete r.width}if(l.Z(r,"_prevHeight")){r.height=r._prevHeight}else{delete r.height}delete r._prevWidth;delete r._prevHeight}))}function it(t,e){var r=t.select("g."+e);if(r.empty()){r=t.append("g").attr("class",e)}return r}var lt=r(2386);var st=r(27484);var ct=r(17967);var dt=r(27856);var ht=r(21307);function ut(t,e,r){const n=e.width;const a=e.height;const o=(n+a)*.9;const i=[{x:o/2,y:0},{x:o,y:-o/2},{x:o/2,y:-o},{x:0,y:-o/2}];const l=Tt(t,o,o,i);r.intersect=function(t){return X(r,i,t)};return l}function ft(t,e,r){const n=4;const a=e.height;const o=a/n;const i=e.width+2*o;const l=[{x:o,y:0},{x:i-o,y:0},{x:i,y:-a/2},{x:i-o,y:-a},{x:o,y:-a},{x:0,y:-a/2}];const s=Tt(t,i,a,l);r.intersect=function(t){return X(r,l,t)};return s}function pt(t,e,r){const n=e.width;const a=e.height;const o=[{x:-a/2,y:0},{x:n,y:0},{x:n,y:-a},{x:-a/2,y:-a},{x:0,y:-a/2}];const i=Tt(t,n,a,o);r.intersect=function(t){return X(r,o,t)};return i}function gt(t,e,r){const n=e.width;const a=e.height;const o=[{x:-2*a/6,y:0},{x:n-a/6,y:0},{x:n+2*a/6,y:-a},{x:a/6,y:-a}];const i=Tt(t,n,a,o);r.intersect=function(t){return X(r,o,t)};return i}function bt(t,e,r){const n=e.width;const a=e.height;const o=[{x:2*a/6,y:0},{x:n+a/6,y:0},{x:n-2*a/6,y:-a},{x:-a/6,y:-a}];const i=Tt(t,n,a,o);r.intersect=function(t){return X(r,o,t)};return i}function yt(t,e,r){const n=e.width;const a=e.height;const o=[{x:-2*a/6,y:0},{x:n+2*a/6,y:0},{x:n-a/6,y:-a},{x:a/6,y:-a}];const i=Tt(t,n,a,o);r.intersect=function(t){return X(r,o,t)};return i}function vt(t,e,r){const n=e.width;const a=e.height;const o=[{x:a/6,y:0},{x:n-a/6,y:0},{x:n+2*a/6,y:-a},{x:-2*a/6,y:-a}];const i=Tt(t,n,a,o);r.intersect=function(t){return X(r,o,t)};return i}function wt(t,e,r){const n=e.width;const a=e.height;const o=[{x:0,y:0},{x:n+a/2,y:0},{x:n,y:-a/2},{x:n+a/2,y:-a},{x:0,y:-a}];const i=Tt(t,n,a,o);r.intersect=function(t){return X(r,o,t)};return i}function xt(t,e,r){const n=e.height;const a=e.width+n/4;const o=t.insert("rect",":first-child").attr("rx",n/2).attr("ry",n/2).attr("x",-a/2).attr("y",-n/2).attr("width",a).attr("height",n);r.intersect=function(t){return F(r,t)};return o}function kt(t,e,r){const n=e.width;const a=e.height;const o=[{x:0,y:0},{x:n,y:0},{x:n,y:-a},{x:0,y:-a},{x:0,y:0},{x:-8,y:0},{x:n+8,y:0},{x:n+8,y:-a},{x:-8,y:-a},{x:-8,y:0}];const i=Tt(t,n,a,o);r.intersect=function(t){return X(r,o,t)};return i}function mt(t,e,r){const n=e.width;const a=n/2;const o=a/(2.5+n/50);const i=e.height+o;const l="M 0,"+o+" a "+a+","+o+" 0,0,0 "+n+" 0 a "+a+","+o+" 0,0,0 "+-n+" 0 l 0,"+i+" a "+a+","+o+" 0,0,0 "+n+" 0 l 0,"+-i;const s=t.attr("label-offset-y",o).insert("path",":first-child").attr("d",l).attr("transform","translate("+-n/2+","+-(i/2+o)+")");r.intersect=function(t){const e=F(r,t);const n=e.x-r.x;if(a!=0&&(Math.abs(n)<r.width/2||Math.abs(n)==r.width/2&&Math.abs(e.y-r.y)>r.height/2-o)){let i=o*o*(1-n*n/(a*a));if(i!=0){i=Math.sqrt(i)}i=o-i;if(t.y-r.y>0){i=-i}e.y+=i}return e};return s}function _t(t){t.shapes().question=ut;t.shapes().hexagon=ft;t.shapes().stadium=xt;t.shapes().subroutine=kt;t.shapes().cylinder=mt;t.shapes().rect_left_inv_arrow=pt;t.shapes().lean_right=gt;t.shapes().lean_left=bt;t.shapes().trapezoid=yt;t.shapes().inv_trapezoid=vt;t.shapes().rect_right_inv_arrow=wt}function St(t){t({question:ut});t({hexagon:ft});t({stadium:xt});t({subroutine:kt});t({cylinder:mt});t({rect_left_inv_arrow:pt});t({lean_right:gt});t({lean_left:bt});t({trapezoid:yt});t({inv_trapezoid:vt});t({rect_right_inv_arrow:wt})}function Tt(t,e,r,n){return t.insert("polygon",":first-child").attr("points",n.map((function(t){return t.x+","+t.y})).join(" ")).attr("transform","translate("+-e/2+","+r/2+")")}const Lt={addToRender:_t,addToRenderV2:St};const Ct={};const At=function(t){const e=Object.keys(t);for(const r of e){Ct[r]=t[r]}};const Et=function(t,e,r,n,a,l){const s=!n?(0,o.Ys)(`[id="${r}"]`):n.select(`[id="${r}"]`);const c=!a?document:a;const d=Object.keys(t);d.forEach((function(r){const n=t[r];let a="default";if(n.classes.length>0){a=n.classes.join(" ")}const o=(0,i.k)(n.styles);let d=n.text!==void 0?n.text:n.id;let h;if((0,i.m)((0,i.c)().flowchart.htmlLabels)){const t={label:d.replace(/fa[blrs]?:fa-[\w-]+/g,(t=>`<i class='${t.replace(":"," ")}'></i>`))};h=(0,y.a)(s,t).node();h.parentNode.removeChild(h)}else{const t=c.createElementNS("http://www.w3.org/2000/svg","text");t.setAttribute("style",o.labelStyle.replace("color:","fill:"));const e=d.split(i.e.lineBreakRegex);for(const r of e){const e=c.createElementNS("http://www.w3.org/2000/svg","tspan");e.setAttributeNS("http://www.w3.org/XML/1998/namespace","xml:space","preserve");e.setAttribute("dy","1em");e.setAttribute("x","1");e.textContent=r;t.appendChild(e)}h=t}let u=0;let f="";switch(n.type){case"round":u=5;f="rect";break;case"square":f="rect";break;case"diamond":f="question";break;case"hexagon":f="hexagon";break;case"odd":f="rect_left_inv_arrow";break;case"lean_right":f="lean_right";break;case"lean_left":f="lean_left";break;case"trapezoid":f="trapezoid";break;case"inv_trapezoid":f="inv_trapezoid";break;case"odd_right":f="rect_left_inv_arrow";break;case"circle":f="circle";break;case"ellipse":f="ellipse";break;case"stadium":f="stadium";break;case"subroutine":f="subroutine";break;case"cylinder":f="cylinder";break;case"group":f="rect";break;default:f="rect"}i.l.warn("Adding node",n.id,n.domId);e.setNode(l.db.lookUpDomId(n.id),{labelType:"svg",labelStyle:o.labelStyle,shape:f,label:h,rx:u,ry:u,class:a,style:o.style,id:l.db.lookUpDomId(n.id)})}))};const Nt=function(t,e,r){let n=0;let a;let l;if(t.defaultStyle!==void 0){const e=(0,i.k)(t.defaultStyle);a=e.style;l=e.labelStyle}t.forEach((function(s){n++;const c="L-"+s.start+"-"+s.end;const d="LS-"+s.start;const h="LE-"+s.end;const u={};if(s.type==="arrow_open"){u.arrowhead="none"}else{u.arrowhead="normal"}let f="";let p="";if(s.style!==void 0){const t=(0,i.k)(s.style);f=t.style;p=t.labelStyle}else{switch(s.stroke){case"normal":f="fill:none";if(a!==void 0){f=a}if(l!==void 0){p=l}break;case"dotted":f="fill:none;stroke-width:2px;stroke-dasharray:3;";break;case"thick":f=" stroke-width: 3.5px;fill:none";break}}u.style=f;u.labelStyle=p;if(s.interpolate!==void 0){u.curve=(0,i.n)(s.interpolate,o.c_6)}else if(t.defaultInterpolate!==void 0){u.curve=(0,i.n)(t.defaultInterpolate,o.c_6)}else{u.curve=(0,i.n)(Ct.curve,o.c_6)}if(s.text===void 0){if(s.style!==void 0){u.arrowheadStyle="fill: #333"}}else{u.arrowheadStyle="fill: #333";u.labelpos="c";if((0,i.m)((0,i.c)().flowchart.htmlLabels)){u.labelType="html";u.label=`<span id="L-${c}" class="edgeLabel L-${d}' L-${h}" style="${u.labelStyle}">${s.text.replace(/fa[blrs]?:fa-[\w-]+/g,(t=>`<i class='${t.replace(":"," ")}'></i>`))}</span>`}else{u.labelType="text";u.label=s.text.replace(i.e.lineBreakRegex,"\n");if(s.style===void 0){u.style=u.style||"stroke: #333; stroke-width: 1.5px;fill:none"}u.labelStyle=u.labelStyle.replace("color:","fill:")}}u.id=c;u.class=d+" "+h;u.minlen=s.length||1;e.setEdge(r.db.lookUpDomId(s.start),r.db.lookUpDomId(s.end),u,n)}))};const $t=function(t,e){i.l.info("Extracting classes");return e.db.getClasses()};const Bt=function(t,e,r,n){i.l.info("Drawing flowchart");const{securityLevel:l,flowchart:s}=(0,i.c)();let c;if(l==="sandbox"){c=(0,o.Ys)("#i"+e)}const d=l==="sandbox"?(0,o.Ys)(c.nodes()[0].contentDocument.body):(0,o.Ys)("body");const u=l==="sandbox"?c.nodes()[0].contentDocument:document;let f=n.db.getDirection();if(f===void 0){f="TD"}const p=s.nodeSpacing||50;const g=s.rankSpacing||50;const b=new a.k({multigraph:true,compound:true}).setGraph({rankdir:f,nodesep:p,ranksep:g,marginx:8,marginy:8}).setDefaultEdgeLabel((function(){return{}}));let y;const v=n.db.getSubGraphs();for(let a=v.length-1;a>=0;a--){y=v[a];n.db.addVertex(y.id,y.title,"group",void 0,y.classes)}const w=n.db.getVertices();i.l.warn("Get vertices",w);const x=n.db.getEdges();let k=0;for(k=v.length-1;k>=0;k--){y=v[k];(0,o.td_)("cluster").append("text");for(let t=0;t<y.nodes.length;t++){i.l.warn("Setting subgraph",y.nodes[t],n.db.lookUpDomId(y.nodes[t]),n.db.lookUpDomId(y.id));b.setParent(n.db.lookUpDomId(y.nodes[t]),n.db.lookUpDomId(y.id))}}Et(w,b,e,d,u,n);Nt(x,b,n);const m=new et;Lt.addToRender(m);m.arrows().none=function t(e,r,n,a){const o=e.append("marker").attr("id",r).attr("viewBox","0 0 10 10").attr("refX",9).attr("refY",5).attr("markerUnits","strokeWidth").attr("markerWidth",8).attr("markerHeight",6).attr("orient","auto");const i=o.append("path").attr("d","M 0 0 L 0 0 L 0 0 z");(0,h.bg)(i,n[a+"Style"])};m.arrows().normal=function t(e,r){const n=e.append("marker").attr("id",r).attr("viewBox","0 0 10 10").attr("refX",9).attr("refY",5).attr("markerUnits","strokeWidth").attr("markerWidth",8).attr("markerHeight",6).attr("orient","auto");n.append("path").attr("d","M 0 0 L 10 5 L 0 10 z").attr("class","arrowheadPath").style("stroke-width",1).style("stroke-dasharray","1,0")};const _=d.select(`[id="${e}"]`);const S=d.select("#"+e+" g");m(S,b);S.selectAll("g.node").attr("title",(function(){return n.db.getTooltip(this.id)}));n.db.indexNodes("subGraph"+k);for(k=0;k<v.length;k++){y=v[k];if(y.title!=="undefined"){const t=u.querySelectorAll("#"+e+' [id="'+n.db.lookUpDomId(y.id)+'"] rect');const r=u.querySelectorAll("#"+e+' [id="'+n.db.lookUpDomId(y.id)+'"]');const a=t[0].x.baseVal.value;const i=t[0].y.baseVal.value;const l=t[0].width.baseVal.value;const s=(0,o.Ys)(r[0]);const c=s.select(".label");c.attr("transform",`translate(${a+l/2}, ${i+14})`);c.attr("id",e+"Text");for(let e=0;e<y.classes.length;e++){r[0].classList.add(y.classes[e])}}}if(!s.htmlLabels){const t=u.querySelectorAll('[id="'+e+'"] .edgeLabel .label');for(const e of t){const t=e.getBBox();const r=u.createElementNS("http://www.w3.org/2000/svg","rect");r.setAttribute("rx",0);r.setAttribute("ry",0);r.setAttribute("width",t.width);r.setAttribute("height",t.height);e.insertBefore(r,e.firstChild)}}(0,i.o)(b,_,s.diagramPadding,s.useMaxWidth);const T=Object.keys(w);T.forEach((function(t){const r=w[t];if(r.link){const a=d.select("#"+e+' [id="'+n.db.lookUpDomId(t)+'"]');if(a){const t=u.createElementNS("http://www.w3.org/2000/svg","a");t.setAttributeNS("http://www.w3.org/2000/svg","class",r.classes.join(" "));t.setAttributeNS("http://www.w3.org/2000/svg","href",r.link);t.setAttributeNS("http://www.w3.org/2000/svg","rel","noopener");if(l==="sandbox"){t.setAttributeNS("http://www.w3.org/2000/svg","target","_top")}else if(r.linkTarget){t.setAttributeNS("http://www.w3.org/2000/svg","target",r.linkTarget)}const e=a.insert((function(){return t}),":first-child");const n=a.select(".label-container");if(n){e.append((function(){return n.node()}))}const o=a.select(".label");if(o){e.append((function(){return o.node()}))}}}}))};const It={setConf:At,addVertices:Et,addEdges:Nt,getClasses:$t,draw:Bt};const Mt={parser:n.p,db:n.f,renderer:lt.f,styles:lt.a,init:t=>{if(!t.flowchart){t.flowchart={}}t.flowchart.arrowMarkerAbsolute=t.arrowMarkerAbsolute;It.setConf(t.flowchart);n.f.clear();n.f.setGen("gen-1")}}},2386:(t,e,r)=>{r.d(e,{a:()=>m,f:()=>w});var n=r(67058);var a=r(34596);var o=r(23787);var i=r(29395);var l=r(1497);var s=r(52048);var c=r(37758);const d=(t,e)=>s.Z.lang.round(c.Z.parse(t)[e]);const h=d;var u=r(76538);const f={};const p=function(t){const e=Object.keys(t);for(const r of e){f[r]=t[r]}};const g=function(t,e,r,n,a,i){const s=n.select(`[id="${r}"]`);const c=Object.keys(t);c.forEach((function(r){const n=t[r];let c="default";if(n.classes.length>0){c=n.classes.join(" ")}c=c+" flowchart-label";const d=(0,o.k)(n.styles);let h=n.text!==void 0?n.text:n.id;let u;o.l.info("vertex",n,n.labelType);if(n.labelType==="markdown"){o.l.info("vertex",n,n.labelType)}else{if((0,o.m)((0,o.c)().flowchart.htmlLabels)){const t={label:h.replace(/fa[blrs]?:fa-[\w-]+/g,(t=>`<i class='${t.replace(":"," ")}'></i>`))};u=(0,l.a)(s,t).node();u.parentNode.removeChild(u)}else{const t=a.createElementNS("http://www.w3.org/2000/svg","text");t.setAttribute("style",d.labelStyle.replace("color:","fill:"));const e=h.split(o.e.lineBreakRegex);for(const r of e){const e=a.createElementNS("http://www.w3.org/2000/svg","tspan");e.setAttributeNS("http://www.w3.org/XML/1998/namespace","xml:space","preserve");e.setAttribute("dy","1em");e.setAttribute("x","1");e.textContent=r;t.appendChild(e)}u=t}}let f=0;let p="";switch(n.type){case"round":f=5;p="rect";break;case"square":p="rect";break;case"diamond":p="question";break;case"hexagon":p="hexagon";break;case"odd":p="rect_left_inv_arrow";break;case"lean_right":p="lean_right";break;case"lean_left":p="lean_left";break;case"trapezoid":p="trapezoid";break;case"inv_trapezoid":p="inv_trapezoid";break;case"odd_right":p="rect_left_inv_arrow";break;case"circle":p="circle";break;case"ellipse":p="ellipse";break;case"stadium":p="stadium";break;case"subroutine":p="subroutine";break;case"cylinder":p="cylinder";break;case"group":p="rect";break;case"doublecircle":p="doublecircle";break;default:p="rect"}e.setNode(n.id,{labelStyle:d.labelStyle,shape:p,labelText:h,labelType:n.labelType,rx:f,ry:f,class:c,style:d.style,id:n.id,link:n.link,linkTarget:n.linkTarget,tooltip:i.db.getTooltip(n.id)||"",domId:i.db.lookUpDomId(n.id),haveCallback:n.haveCallback,width:n.type==="group"?500:void 0,dir:n.dir,type:n.type,props:n.props,padding:(0,o.c)().flowchart.padding});o.l.info("setNode",{labelStyle:d.labelStyle,labelType:n.labelType,shape:p,labelText:h,rx:f,ry:f,class:c,style:d.style,id:n.id,domId:i.db.lookUpDomId(n.id),width:n.type==="group"?500:void 0,type:n.type,dir:n.dir,props:n.props,padding:(0,o.c)().flowchart.padding})}))};const b=function(t,e,r){o.l.info("abc78 edges = ",t);let n=0;let i={};let l;let s;if(t.defaultStyle!==void 0){const e=(0,o.k)(t.defaultStyle);l=e.style;s=e.labelStyle}t.forEach((function(r){n++;const c="L-"+r.start+"-"+r.end;if(i[c]===void 0){i[c]=0;o.l.info("abc78 new entry",c,i[c])}else{i[c]++;o.l.info("abc78 new entry",c,i[c])}let d=c+"-"+i[c];o.l.info("abc78 new link id to be used is",c,d,i[c]);const h="LS-"+r.start;const u="LE-"+r.end;const p={style:"",labelStyle:""};p.minlen=r.length||1;if(r.type==="arrow_open"){p.arrowhead="none"}else{p.arrowhead="normal"}p.arrowTypeStart="arrow_open";p.arrowTypeEnd="arrow_open";switch(r.type){case"double_arrow_cross":p.arrowTypeStart="arrow_cross";case"arrow_cross":p.arrowTypeEnd="arrow_cross";break;case"double_arrow_point":p.arrowTypeStart="arrow_point";case"arrow_point":p.arrowTypeEnd="arrow_point";break;case"double_arrow_circle":p.arrowTypeStart="arrow_circle";case"arrow_circle":p.arrowTypeEnd="arrow_circle";break}let g="";let b="";switch(r.stroke){case"normal":g="fill:none;";if(l!==void 0){g=l}if(s!==void 0){b=s}p.thickness="normal";p.pattern="solid";break;case"dotted":p.thickness="normal";p.pattern="dotted";p.style="fill:none;stroke-width:2px;stroke-dasharray:3;";break;case"thick":p.thickness="thick";p.pattern="solid";p.style="stroke-width: 3.5px;fill:none;";break;case"invisible":p.thickness="invisible";p.pattern="solid";p.style="stroke-width: 0;fill:none;";break}if(r.style!==void 0){const t=(0,o.k)(r.style);g=t.style;b=t.labelStyle}p.style=p.style+=g;p.labelStyle=p.labelStyle+=b;if(r.interpolate!==void 0){p.curve=(0,o.n)(r.interpolate,a.c_6)}else if(t.defaultInterpolate!==void 0){p.curve=(0,o.n)(t.defaultInterpolate,a.c_6)}else{p.curve=(0,o.n)(f.curve,a.c_6)}if(r.text===void 0){if(r.style!==void 0){p.arrowheadStyle="fill: #333"}}else{p.arrowheadStyle="fill: #333";p.labelpos="c"}p.labelType=r.labelType;p.label=r.text.replace(o.e.lineBreakRegex,"\n");if(r.style===void 0){p.style=p.style||"stroke: #333; stroke-width: 1.5px;fill:none;"}p.labelStyle=p.labelStyle.replace("color:","fill:");p.id=d;p.classes="flowchart-link "+h+" "+u;e.setEdge(r.start,r.end,p,n)}))};const y=function(t,e){return e.db.getClasses()};const v=async function(t,e,r,l){o.l.info("Drawing flowchart");let s=l.db.getDirection();if(s===void 0){s="TD"}const{securityLevel:c,flowchart:d}=(0,o.c)();const h=d.nodeSpacing||50;const u=d.rankSpacing||50;let f;if(c==="sandbox"){f=(0,a.Ys)("#i"+e)}const p=c==="sandbox"?(0,a.Ys)(f.nodes()[0].contentDocument.body):(0,a.Ys)("body");const y=c==="sandbox"?f.nodes()[0].contentDocument:document;const v=new n.k({multigraph:true,compound:true}).setGraph({rankdir:s,nodesep:h,ranksep:u,marginx:0,marginy:0}).setDefaultEdgeLabel((function(){return{}}));let w;const x=l.db.getSubGraphs();o.l.info("Subgraphs - ",x);for(let n=x.length-1;n>=0;n--){w=x[n];o.l.info("Subgraph - ",w);l.db.addVertex(w.id,{text:w.title,type:w.labelType},"group",void 0,w.classes,w.dir)}const k=l.db.getVertices();const m=l.db.getEdges();o.l.info("Edges",m);let _=0;for(_=x.length-1;_>=0;_--){w=x[_];(0,a.td_)("cluster").append("text");for(let t=0;t<w.nodes.length;t++){o.l.info("Setting up subgraphs",w.nodes[t],w.id);v.setParent(w.nodes[t],w.id)}}g(k,v,e,p,y,l);b(m,v);const S=p.select(`[id="${e}"]`);const T=p.select("#"+e+" g");await(0,i.r)(T,v,["point","circle","cross"],"flowchart",e);o.u.insertTitle(S,"flowchartTitleText",d.titleTopMargin,l.db.getDiagramTitle());(0,o.o)(v,S,d.diagramPadding,d.useMaxWidth);l.db.indexNodes("subGraph"+_);if(!d.htmlLabels){const t=y.querySelectorAll('[id="'+e+'"] .edgeLabel .label');for(const e of t){const t=e.getBBox();const r=y.createElementNS("http://www.w3.org/2000/svg","rect");r.setAttribute("rx",0);r.setAttribute("ry",0);r.setAttribute("width",t.width);r.setAttribute("height",t.height);e.insertBefore(r,e.firstChild)}}const L=Object.keys(k);L.forEach((function(t){const r=k[t];if(r.link){const n=(0,a.Ys)("#"+e+' [id="'+t+'"]');if(n){const t=y.createElementNS("http://www.w3.org/2000/svg","a");t.setAttributeNS("http://www.w3.org/2000/svg","class",r.classes.join(" "));t.setAttributeNS("http://www.w3.org/2000/svg","href",r.link);t.setAttributeNS("http://www.w3.org/2000/svg","rel","noopener");if(c==="sandbox"){t.setAttributeNS("http://www.w3.org/2000/svg","target","_top")}else if(r.linkTarget){t.setAttributeNS("http://www.w3.org/2000/svg","target",r.linkTarget)}const e=n.insert((function(){return t}),":first-child");const a=n.select(".label-container");if(a){e.append((function(){return a.node()}))}const o=n.select(".label");if(o){e.append((function(){return o.node()}))}}}}))};const w={setConf:p,addVertices:g,addEdges:b,getClasses:y,draw:v};const x=(t,e)=>{const r=h;const n=r(t,"r");const a=r(t,"g");const o=r(t,"b");return u.Z(n,a,o,e)};const k=t=>`.label {\n    font-family: ${t.fontFamily};\n    color: ${t.nodeTextColor||t.textColor};\n  }\n  .cluster-label text {\n    fill: ${t.titleColor};\n  }\n  .cluster-label span,p {\n    color: ${t.titleColor};\n  }\n\n  .label text,span,p {\n    fill: ${t.nodeTextColor||t.textColor};\n    color: ${t.nodeTextColor||t.textColor};\n  }\n\n  .node rect,\n  .node circle,\n  .node ellipse,\n  .node polygon,\n  .node path {\n    fill: ${t.mainBkg};\n    stroke: ${t.nodeBorder};\n    stroke-width: 1px;\n  }\n  .flowchart-label text {\n    text-anchor: middle;\n  }\n  // .flowchart-label .text-outer-tspan {\n  //   text-anchor: middle;\n  // }\n  // .flowchart-label .text-inner-tspan {\n  //   text-anchor: start;\n  // }\n\n  .node .label {\n    text-align: center;\n  }\n  .node.clickable {\n    cursor: pointer;\n  }\n\n  .arrowheadPath {\n    fill: ${t.arrowheadColor};\n  }\n\n  .edgePath .path {\n    stroke: ${t.lineColor};\n    stroke-width: 2.0px;\n  }\n\n  .flowchart-link {\n    stroke: ${t.lineColor};\n    fill: none;\n  }\n\n  .edgeLabel {\n    background-color: ${t.edgeLabelBackground};\n    rect {\n      opacity: 0.5;\n      background-color: ${t.edgeLabelBackground};\n      fill: ${t.edgeLabelBackground};\n    }\n    text-align: center;\n  }\n\n  /* For html labels only */\n  .labelBkg {\n    background-color: ${x(t.edgeLabelBackground,.5)};\n    // background-color: \n  }\n\n  .cluster rect {\n    fill: ${t.clusterBkg};\n    stroke: ${t.clusterBorder};\n    stroke-width: 1px;\n  }\n\n  .cluster text {\n    fill: ${t.titleColor};\n  }\n\n  .cluster span,p {\n    color: ${t.titleColor};\n  }\n  /* .cluster div {\n    color: ${t.titleColor};\n  } */\n\n  div.mermaidTooltip {\n    position: absolute;\n    text-align: center;\n    max-width: 200px;\n    padding: 2px;\n    font-family: ${t.fontFamily};\n    font-size: 12px;\n    background: ${t.tertiaryColor};\n    border: 1px solid ${t.border2};\n    border-radius: 2px;\n    pointer-events: none;\n    z-index: 100;\n  }\n\n  .flowchartTitleText {\n    text-anchor: middle;\n    font-size: 18px;\n    fill: ${t.textColor};\n  }\n`;const m=k}}]);
//# sourceMappingURL=1166.e92d29293b8ddfd701cf.js.map?v=e92d29293b8ddfd701cf